import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent
from argparse import Namespace

from xuance.common import Tuple
from xuance.torch.policies.dreamer import DreamerV3Policy
from xuance.torch import Module, Tensor
from xuance.torch.utils.distributions import SymLogDistribution, TwoHotEncodingDistribution, BernoulliSafeMode


def _listify(value, length: int):
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(values) == 1:
        values = values * length
    if len(values) != length:
        raise ValueError(f"Expected length 1 or {length}, got {len(values)}.")
    return values


class SparseLatentHeads(nn.Module):
    """Maps Dreamer recurrent states h_t into multiple sparse latent heads.

    Implements:
        u_t = g_psi(h_t)
        z_t^(ell) = sigma_ell(W_ell u_t + b_ell), ell = 1, ..., L
        z_t = [z_t^(1), ..., z_t^(L)]
    """

    def __init__(
        self,
        recurrent_state_size: int,
        trunk_hidden_size: int,
        trunk_output_size: int,
        num_heads: int,
        head_dim: int,
        activation: str,
        sparsity_mode: str,
        sparsity_topk: int | list[int] | None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sparsity_mode = sparsity_mode
        self.sparsity_topk = _listify(sparsity_topk, num_heads) if sparsity_topk is not None else [0] * num_heads

        self.trunk = nn.Sequential(
            nn.Linear(recurrent_state_size, trunk_hidden_size),
            nn.SiLU(),
            nn.Linear(trunk_hidden_size, trunk_output_size),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(trunk_output_size, head_dim) for _ in range(num_heads)])
        self.activation = self._build_activation(activation)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "identity":
            return nn.Identity()
        return nn.SiLU()

    def _apply_topk_sparsity(self, heads: Tensor) -> Tensor:
        if self.sparsity_mode in ("none", "l1"):
            return heads
        if self.sparsity_mode == "level_topk":
            sparse_heads = []
            for level, topk in enumerate(self.sparsity_topk):
                level_heads = heads[..., level, :]
                if topk is None or topk <= 0 or topk >= self.head_dim:
                    sparse_heads.append(level_heads)
                    continue
                _, indices = torch.topk(level_heads.abs(), k=topk, dim=-1)
                mask = torch.zeros_like(level_heads).scatter_(-1, indices, 1.0)
                sparse_heads.append(level_heads * mask)
            return torch.stack(sparse_heads, dim=-2)
        if self.sparsity_mode == "level_batch_topk":
            sparse_heads = []
            for level, topk in enumerate(self.sparsity_topk):
                level_heads = heads[..., level, :]
                if topk is None or topk <= 0 or topk >= self.head_dim:
                    sparse_heads.append(level_heads)
                    continue
                flat = level_heads.reshape(-1, self.head_dim)
                keep = min(int(topk) * flat.shape[0], flat.numel())
                _, indices = torch.topk(flat.abs().reshape(-1), k=keep, sorted=False)
                mask = torch.zeros_like(flat).reshape(-1).scatter_(-1, indices, 1.0).reshape_as(flat)
                sparse_heads.append((flat * mask).reshape_as(level_heads))
            return torch.stack(sparse_heads, dim=-2)
        if self.sparsity_mode == "global_topk":
            topk = int(sum(max(int(k), 0) for k in self.sparsity_topk))
            flat = heads.reshape(*heads.shape[:-2], self.num_heads * self.head_dim)
            if topk <= 0 or topk >= flat.shape[-1]:
                return heads
            _, indices = torch.topk(flat.abs(), k=topk, dim=-1)
            mask = torch.zeros_like(flat).scatter_(-1, indices, 1.0)
            return (flat * mask).reshape_as(heads)
        raise ValueError(f"Unknown hierarchical_latent.sparsity.mode: {self.sparsity_mode}")

    def forward(self, recurrent_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        original_shape = recurrent_states.shape[:-1]
        flat_h = recurrent_states.reshape(-1, recurrent_states.shape[-1])
        trunk_latent = self.trunk(flat_h)
        heads = torch.stack([self.activation(head(trunk_latent)) for head in self.heads], dim=-2)
        heads = self._apply_topk_sparsity(heads)
        heads = heads.reshape(*original_shape, self.num_heads, self.head_dim)
        flat_heads = heads.reshape(*original_shape, self.num_heads * self.head_dim)
        trunk_latent = trunk_latent.reshape(*original_shape, trunk_latent.shape[-1])
        return trunk_latent, heads, flat_heads


class NestedReconstructionDecoders(nn.Module):
    """Reconstruct h_t from increasingly fine sparse-latent prefixes."""

    def __init__(
        self,
        recurrent_state_size: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        betas: list[float],
        stop_gradient_lower_levels: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.betas = betas
        self.stop_gradient_lower_levels = stop_gradient_lower_levels
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear((level + 1) * head_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, recurrent_state_size),
                )
                for level in range(num_heads)
            ]
        )

    def _prefix_for_level(self, sparse_heads: Tensor, level: int) -> Tensor:
        if not self.stop_gradient_lower_levels or level == 0:
            prefix = sparse_heads[..., :level + 1, :]
        else:
            lower_levels = sparse_heads[..., :level, :].detach()
            current_level = sparse_heads[..., level:level + 1, :]
            prefix = torch.cat((lower_levels, current_level), dim=-2)
        return prefix.reshape(*sparse_heads.shape[:-2], (level + 1) * self.head_dim)

    def forward(self, sparse_heads: Tensor, target_h: Tensor) -> Tuple[Tensor, dict[str, Tensor]]:
        loss = target_h.new_zeros(())
        info = {}
        target = target_h.detach()
        previous_level_loss = None
        for level, decoder in enumerate(self.decoders):
            prefix = self._prefix_for_level(sparse_heads, level)
            reconstructed_h = decoder(prefix)
            level_loss = (target - reconstructed_h).pow(2).mean()
            beta = self.betas[level]
            loss = loss + beta * level_loss
            info[f"hierarchical_latent/recon_loss_level_{level + 1}"] = level_loss.detach()
            if previous_level_loss is not None:
                info[f"hierarchical_latent/recon_marginal_gain_level_{level + 1}"] = (
                    previous_level_loss - level_loss
                ).detach()
            previous_level_loss = level_loss
        info["hierarchical_latent/recon_loss"] = loss.detach()
        return loss, info


class MultiStrideSparseDynamics(nn.Module):
    """Predict future h_{t+Delta_l} from sparse prefixes and action windows."""

    def __init__(
        self,
        recurrent_state_size: int,
        actions_dim: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        strides: list[int],
        alphas: list[float],
        stop_gradient_lower_levels: bool,
        target_stop_gradient: bool,
        action_mode: str,
    ):
        super().__init__()
        self.recurrent_state_size = recurrent_state_size
        self.actions_dim = actions_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.strides = strides
        self.alphas = alphas
        self.stop_gradient_lower_levels = stop_gradient_lower_levels
        self.target_stop_gradient = target_stop_gradient
        self.action_mode = action_mode
        if action_mode == "state_only":
            action_context_dims = [0 for _ in range(num_heads)]
        elif action_mode == "current_action":
            action_context_dims = [actions_dim for _ in range(num_heads)]
        elif action_mode == "subsequence":
            action_context_dims = [stride * actions_dim for stride in strides]
        else:
            raise ValueError(
                "hierarchical_latent.sparse_dynamics.action_mode must be one of "
                "{state_only, current_action, subsequence}."
            )
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear((level + 1) * head_dim + action_context_dims[level], hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, recurrent_state_size),
                )
                for level in range(num_heads)
            ]
        )

    def _action_context(self, actions: Tensor, stride: int) -> Tensor | None:
        if self.action_mode == "state_only":
            return None
        if self.action_mode == "current_action":
            return actions[1:actions.shape[0] - stride + 1]
        return self._action_windows(actions, stride)

    def _prefix_for_level(self, sparse_heads: Tensor, level: int) -> Tensor:
        if not self.stop_gradient_lower_levels or level == 0:
            prefix = sparse_heads[..., :level + 1, :]
        else:
            lower_levels = sparse_heads[..., :level, :].detach()
            current_level = sparse_heads[..., level:level + 1, :]
            prefix = torch.cat((lower_levels, current_level), dim=-2)
        return prefix.reshape(*sparse_heads.shape[:-2], (level + 1) * self.head_dim)

    @staticmethod
    def _action_windows(actions: Tensor, stride: int) -> Tensor:
        # actions[i] is the transition action entering state i, so h_t -> h_{t+d}
        # uses actions[t+1], ..., actions[t+d].
        windows = [actions[start + 1:start + stride + 1] for start in range(actions.shape[0] - stride)]
        return torch.stack(windows, dim=0).transpose(1, 2).reshape(
            actions.shape[0] - stride,
            actions.shape[1],
            stride * actions.shape[-1],
        )

    def forward(
        self,
        sparse_heads: Tensor,
        actions: Tensor,
        target_h: Tensor,
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        loss = target_h.new_zeros(())
        info = {}
        for level, predictor in enumerate(self.predictors):
            stride = self.strides[level]
            if target_h.shape[0] <= stride:
                info[f"hierarchical_latent/sdyn_loss_level_{level + 1}"] = target_h.new_zeros(())
                continue

            prefix = self._prefix_for_level(sparse_heads[:-stride], level)
            action_context = self._action_context(actions, stride)
            predictor_input = prefix if action_context is None else torch.cat((prefix, action_context), dim=-1)
            predicted_h = predictor(predictor_input)
            target_future_h = target_h[stride:]
            if self.target_stop_gradient:
                target_future_h = target_future_h.detach()
            level_loss = (target_future_h - predicted_h).pow(2).mean()
            loss = loss + self.alphas[level] * level_loss
            info[f"hierarchical_latent/sdyn_loss_level_{level + 1}"] = level_loss.detach()

        info["hierarchical_latent/sdyn_loss"] = loss.detach()
        return loss, info


class TemporalContrastiveVICReg(nn.Module):
    """Slow coarse-code contrastive loss plus VICReg-style anti-collapse terms."""

    def __init__(
        self,
        head_dim: int,
        projection_hidden_size: int,
        projection_dim: int,
        projector_type: str,
        temporal_mode: str,
        positive_stride: int,
        temperature: float,
        std_target: float,
        vc_mode: str,
        variance_weight: float,
        covariance_weight: float,
    ):
        super().__init__()
        self.temporal_mode = temporal_mode
        self.positive_stride = positive_stride
        self.temperature = temperature
        self.std_target = std_target
        self.vc_mode = vc_mode
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        if projector_type == "none":
            if projection_dim != head_dim:
                raise ValueError("projection_dim must equal head_dim when projector_type is none.")
            self.projector = nn.Identity()
        elif projector_type == "linear":
            self.projector = nn.Linear(head_dim, projection_dim)
        elif projector_type == "nonlinear":
            self.projector = nn.Sequential(
                nn.Linear(head_dim, projection_hidden_size),
                nn.SiLU(),
                nn.Linear(projection_hidden_size, projection_dim),
            )
        else:
            raise ValueError("projector_type must be one of {none, linear, nonlinear}.")

    @staticmethod
    def _off_diagonal(matrix: Tensor) -> Tensor:
        n, m = matrix.shape
        assert n == m
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _temporal_contrastive_loss(self, projections: Tensor) -> Tensor:
        stride = self.positive_stride
        if projections.shape[0] <= stride:
            return projections.new_zeros(())
        anchors = projections[:-stride].reshape(-1, projections.shape[-1])
        positives = projections[stride:].reshape(-1, projections.shape[-1])
        if anchors.shape[0] <= 1:
            return projections.new_zeros(())
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        logits = anchors @ positives.transpose(0, 1)
        logits = logits / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def _vicreg_loss(self, projections: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        flat = projections.reshape(-1, projections.shape[-1])
        if flat.shape[0] <= 1:
            zero = projections.new_zeros(())
            return zero, zero, zero
        flat = flat - flat.mean(dim=0, keepdim=True)

        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = torch.relu(self.std_target - std).mean()

        covariance = flat.transpose(0, 1) @ flat / (flat.shape[0] - 1)
        covariance_loss = self._off_diagonal(covariance).pow(2).sum() / flat.shape[-1]
        if self.vc_mode == "none":
            vicreg_loss = projections.new_zeros(())
        elif self.vc_mode == "variance":
            vicreg_loss = self.variance_weight * variance_loss
        elif self.vc_mode == "covariance":
            vicreg_loss = self.covariance_weight * covariance_loss
        elif self.vc_mode == "both":
            vicreg_loss = self.variance_weight * variance_loss + self.covariance_weight * covariance_loss
        else:
            raise ValueError("variance_covariance.mode must be one of {none, variance, covariance, both}.")
        return vicreg_loss, variance_loss, covariance_loss

    def _smoothness_loss(self, projections: Tensor) -> Tensor:
        stride = self.positive_stride
        if projections.shape[0] <= stride:
            return projections.new_zeros(())
        return (projections[stride:] - projections[:-stride]).pow(2).mean()

    def forward(self, coarse_head: Tensor) -> Tuple[Tensor, Tensor, dict[str, Tensor]]:
        projections = self.projector(coarse_head)
        if self.temporal_mode == "none":
            temporal_loss = projections.new_zeros(())
        elif self.temporal_mode == "smooth":
            temporal_loss = self._smoothness_loss(projections)
        elif self.temporal_mode == "contrastive":
            temporal_loss = self._temporal_contrastive_loss(projections)
        else:
            raise ValueError("temporal_consistency.mode must be one of {none, smooth, contrastive}.")
        vc_loss, variance_loss, covariance_loss = self._vicreg_loss(projections)
        info = {
            "hierarchical_latent/temp_loss": temporal_loss.detach(),
            "hierarchical_latent/temp_contrastive_loss": (
                temporal_loss.detach() if self.temporal_mode == "contrastive" else projections.new_zeros(())
            ),
            "hierarchical_latent/temp_smooth_loss": (
                temporal_loss.detach() if self.temporal_mode == "smooth" else projections.new_zeros(())
            ),
            "hierarchical_latent/vicreg_loss": vc_loss.detach(),
            "hierarchical_latent/vicreg_variance_loss": variance_loss.detach(),
            "hierarchical_latent/vicreg_covariance_loss": covariance_loss.detach(),
            "hierarchical_latent/coarse_projection_std": projections.reshape(-1, projections.shape[-1]).std(
                dim=0, unbiased=False
            ).mean().detach(),
        }
        return temporal_loss, vc_loss, info


class HierarchicalDreamerPolicy(DreamerV3Policy):
    """Research fork of DreamerV3Policy.

    Put hierarchy-specific actor, imagination, objective, or rollout changes
    here while preserving DreamerV3Policy for baseline runs.
    """

    def __init__(self, model: Module, config: Namespace):
        super(HierarchicalDreamerPolicy, self).__init__(model, config)
        latent_config = self.config.hierarchical_latent
        self.use_hierarchical_latent = latent_config.enabled
        self.hierarchy_training_active = True
        self.hierarchical_integration = latent_config.integration
        self.hierarchical_residual_scale = latent_config.residual_scale
        self.ablation_name = latent_config.get("ablation_name", "full")

        self.sparse_latent_heads = SparseLatentHeads(
            recurrent_state_size=self.recurrent_state_size,
            trunk_hidden_size=latent_config.trunk_hidden_size,
            trunk_output_size=latent_config.trunk_output_size,
            num_heads=latent_config.num_heads,
            head_dim=latent_config.head_dim,
            activation=latent_config.activation,
            sparsity_mode=latent_config.sparsity.mode,
            sparsity_topk=latent_config.sparsity_topk,
        )
        self.sparse_latent_readout = nn.Linear(
            latent_config.num_heads * latent_config.head_dim,
            self.recurrent_state_size,
        )
        betas = list(latent_config.reconstruction.betas)
        if len(betas) == 1:
            betas = betas * latent_config.num_heads
        if len(betas) != latent_config.num_heads:
            raise ValueError("hierarchical_latent.reconstruction.betas must have length 1 or num_heads.")
        self.nested_reconstruction_decoders = NestedReconstructionDecoders(
            recurrent_state_size=self.recurrent_state_size,
            num_heads=latent_config.num_heads,
            head_dim=latent_config.head_dim,
            hidden_size=latent_config.reconstruction.decoder_hidden_size,
            betas=betas,
            stop_gradient_lower_levels=latent_config.reconstruction.stop_gradient_lower_levels,
        )
        strides = list(latent_config.sparse_dynamics.strides)
        alphas = list(latent_config.sparse_dynamics.alphas)
        if len(alphas) == 1:
            alphas = alphas * latent_config.num_heads
        if len(strides) != latent_config.num_heads:
            raise ValueError("hierarchical_latent.sparse_dynamics.strides must have length num_heads.")
        if len(alphas) != latent_config.num_heads:
            raise ValueError("hierarchical_latent.sparse_dynamics.alphas must have length 1 or num_heads.")
        require_strict_strides = latent_config.sparse_dynamics.get("require_strict_decreasing", True)
        if require_strict_strides and (
            strides[-1] != 1 or any(strides[i] <= strides[i + 1] for i in range(len(strides) - 1))
        ):
            raise ValueError("hierarchical_latent.sparse_dynamics.strides must satisfy Delta_1 > ... > Delta_L = 1.")
        self.multi_stride_sparse_dynamics = MultiStrideSparseDynamics(
            recurrent_state_size=self.recurrent_state_size,
            actions_dim=self.actions_dim,
            num_heads=latent_config.num_heads,
            head_dim=latent_config.head_dim,
            hidden_size=latent_config.sparse_dynamics.predictor_hidden_size,
            strides=strides,
            alphas=alphas,
            stop_gradient_lower_levels=latent_config.sparse_dynamics.stop_gradient_lower_levels,
            target_stop_gradient=latent_config.sparse_dynamics.target_stop_gradient,
            action_mode=latent_config.sparse_dynamics.action_mode,
        )
        gammas = list(latent_config.sparsity.gammas)
        if len(gammas) == 1:
            gammas = gammas * latent_config.num_heads
        if len(gammas) != latent_config.num_heads:
            raise ValueError("hierarchical_latent.sparsity.gammas must have length 1 or num_heads.")
        self.sparsity_gammas = torch.as_tensor(gammas, device=self.device).view(1, 1, latent_config.num_heads, 1)
        self.temporal_vicreg = TemporalContrastiveVICReg(
            head_dim=latent_config.head_dim,
            projection_hidden_size=latent_config.temporal_consistency.projection_hidden_size,
            projection_dim=latent_config.temporal_consistency.projection_dim,
            projector_type=latent_config.temporal_consistency.projector_type,
            temporal_mode=latent_config.temporal_consistency.mode,
            positive_stride=latent_config.temporal_consistency.positive_stride,
            temperature=latent_config.temporal_consistency.temperature,
            std_target=latent_config.variance_covariance.std_target,
            vc_mode=latent_config.variance_covariance.mode,
            variance_weight=latent_config.variance_covariance.variance_weight,
            covariance_weight=latent_config.variance_covariance.covariance_weight,
        )
        self._move_hierarchical_modules_to_device()
        self.latest_hierarchical_recon_loss = None
        self.latest_sparse_dynamics_loss = None
        self.latest_temporal_contrastive_loss = None
        self.latest_vicreg_loss = None
        self.latest_sparsity_loss = None
        self.latest_sparse_latent_info = {}

    def _move_hierarchical_modules_to_device(self):
        self.sparse_latent_heads.to(self.device)
        self.sparse_latent_readout.to(self.device)
        self.nested_reconstruction_decoders.to(self.device)
        self.multi_stride_sparse_dynamics.to(self.device)
        self.temporal_vicreg.to(self.device)

    def hierarchical_latent_parameters(self):
        return (
            list(self.sparse_latent_heads.parameters()) +
            list(self.sparse_latent_readout.parameters()) +
            list(self.nested_reconstruction_decoders.parameters()) +
            list(self.multi_stride_sparse_dynamics.parameters()) +
            list(self.temporal_vicreg.parameters())
        )

    def apply_hierarchical_latent(self, recurrent_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.use_hierarchical_latent or not self.hierarchy_training_active:
            zero = recurrent_states.new_zeros(())
            self.latest_hierarchical_recon_loss = zero
            self.latest_sparse_dynamics_loss = zero
            self.latest_temporal_contrastive_loss = zero
            self.latest_vicreg_loss = zero
            self.latest_sparsity_loss = zero
            self.latest_sparse_latent_info = {
                "hierarchical_latent/active": zero.detach(),
                "hierarchical_latent/active_ratio": zero.detach(),
                "hierarchical_latent/mean_abs": zero.detach(),
                "hierarchical_latent/sparsity_loss": zero.detach(),
                "hierarchical_latent/num_heads": zero.detach() + self.config.hierarchical_latent.num_heads,
                "hierarchical_latent/head_dim": zero.detach() + self.config.hierarchical_latent.head_dim,
            }
            return recurrent_states, None

        _, sparse_heads, sparse_flat = self.sparse_latent_heads(recurrent_states)
        recon_loss, recon_info = self.nested_reconstruction_decoders(sparse_heads, recurrent_states)
        self.latest_hierarchical_recon_loss = recon_loss
        sparse_dynamics_loss, sparse_dynamics_info = self.multi_stride_sparse_dynamics(
            sparse_heads, actions, recurrent_states
        )
        self.latest_sparse_dynamics_loss = sparse_dynamics_loss
        temporal_loss, vicreg_loss, regularizer_info = self.temporal_vicreg(sparse_heads[..., 0, :])
        self.latest_temporal_contrastive_loss = temporal_loss
        self.latest_vicreg_loss = vicreg_loss
        sparsity_loss_by_level = (sparse_heads.abs() * self.sparsity_gammas).mean(dim=(0, 1, 3))
        sparsity_loss = sparsity_loss_by_level.sum()
        self.latest_sparsity_loss = sparsity_loss
        delta_h = self.sparse_latent_readout(sparse_flat)
        if self.hierarchical_integration == "replace":
            transformed_states = delta_h
        elif self.hierarchical_integration == "none" or not self.use_hierarchical_latent:
            transformed_states = recurrent_states
        else:
            transformed_states = recurrent_states + self.hierarchical_residual_scale * delta_h

        with torch.no_grad():
            active_mask = (sparse_heads.abs() > 1e-8).float()
            active_ratio = active_mask.mean()
            mean_abs = sparse_heads.abs().mean()
            self.latest_sparse_latent_info = {
                "hierarchical_latent/active": torch.ones_like(active_ratio).detach(),
                "hierarchical_latent/active_ratio": active_ratio.detach(),
                "hierarchical_latent/mean_abs": mean_abs.detach(),
                "hierarchical_latent/sparsity_loss": sparsity_loss.detach(),
                "hierarchical_latent/num_heads": active_ratio.detach() * 0 + self.config.hierarchical_latent.num_heads,
                "hierarchical_latent/head_dim": active_ratio.detach() * 0 + self.config.hierarchical_latent.head_dim,
                **recon_info,
                **sparse_dynamics_info,
                **regularizer_info,
            }
            self.latest_sparse_latent_info.update(self._sparse_usage_diagnostics(sparse_heads, active_mask))
            for level, level_loss in enumerate(sparsity_loss_by_level):
                self.latest_sparse_latent_info[
                    f"hierarchical_latent/sparsity_loss_level_{level + 1}"
                ] = level_loss.detach()
        return transformed_states, sparse_heads

    @staticmethod
    def _effective_rank(covariance: Tensor) -> Tensor:
        eigvals = torch.linalg.eigvalsh(covariance).clamp_min(0)
        eigsum = eigvals.sum()
        if eigsum <= 0:
            return covariance.new_zeros(())
        probs = eigvals / eigsum.clamp_min(1e-8)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return torch.exp(entropy)

    @staticmethod
    def _offdiag_norm(covariance: Tensor) -> Tensor:
        diagonal = torch.diag(torch.diagonal(covariance))
        return (covariance - diagonal).pow(2).mean().sqrt()

    @staticmethod
    def _utilization_entropy(active_mask: Tensor) -> Tensor:
        usage = active_mask.reshape(-1, active_mask.shape[-1]).mean(dim=0)
        total = usage.sum()
        if total <= 0:
            return active_mask.new_zeros(())
        probs = usage / total.clamp_min(1e-8)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return entropy / torch.log(torch.as_tensor(float(active_mask.shape[-1]), device=active_mask.device))

    def _sparse_usage_diagnostics(self, sparse_heads: Tensor, active_mask: Tensor) -> dict[str, Tensor]:
        info = {}
        flat_by_level = sparse_heads.reshape(-1, sparse_heads.shape[-2], sparse_heads.shape[-1])
        active_by_level = active_mask.reshape(-1, active_mask.shape[-2], active_mask.shape[-1])
        dead_threshold = self.config.hierarchical_latent.get("dead_feature_threshold", 1e-6)
        for level in range(flat_by_level.shape[-2]):
            level_values = flat_by_level[:, level, :]
            level_active = active_by_level[:, level, :]
            centered = level_values - level_values.mean(dim=0, keepdim=True)
            variances = centered.var(dim=0, unbiased=False)
            if centered.shape[0] > 1:
                covariance = centered.transpose(0, 1) @ centered / (centered.shape[0] - 1)
                effective_rank = self._effective_rank(covariance)
                offdiag_norm = self._offdiag_norm(covariance)
            else:
                effective_rank = level_values.new_zeros(())
                offdiag_norm = level_values.new_zeros(())
            alive_ratio = (level_active.mean(dim=0) > dead_threshold).float().mean()
            level_prefix = f"hierarchical_latent/level_{level + 1}"
            info[f"{level_prefix}_active_ratio"] = level_active.mean().detach()
            info[f"{level_prefix}_alive_ratio"] = alive_ratio.detach()
            info[f"{level_prefix}_dead_ratio"] = (1.0 - alive_ratio).detach()
            info[f"{level_prefix}_mean_abs"] = level_values.abs().mean().detach()
            info[f"{level_prefix}_min_variance"] = variances.min().detach()
            info[f"{level_prefix}_mean_variance"] = variances.mean().detach()
            info[f"{level_prefix}_effective_rank"] = effective_rank.detach()
            info[f"{level_prefix}_offdiag_cov_norm"] = offdiag_norm.detach()
            info[f"{level_prefix}_utilization_entropy"] = self._utilization_entropy(level_active).detach()
        return info

    def model_forward(
        self,
        obs: Tensor,
        acts: Tensor,
        is_first: Tensor,
    ) -> Tuple[SymLogDistribution, TwoHotEncodingDistribution, Independent, Tensor, Tensor, Tensor, Tensor]:
        recurrent_state = torch.zeros(1, self.batch_size, self.recurrent_state_size, device=self.device)
        recurrent_states = torch.empty(self.seq_len, self.batch_size, self.recurrent_state_size, device=self.device)
        priors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)
        embedded_obs = self.world_model.encoder(obs)

        posterior = torch.zeros(1, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors = torch.empty(self.seq_len, self.batch_size, self.stoch_size, self.disc_size, device=self.device)
        posteriors_logits = torch.empty(self.seq_len, self.batch_size, self.stoch_state_size, device=self.device)
        for i in range(0, self.seq_len):
            recurrent_state, posterior, _, posterior_logits, prior_logits = self.world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                acts[i: i + 1],
                embedded_obs[i: i + 1],
                is_first[i: i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits

        recurrent_states, _ = self.apply_hierarchical_latent(recurrent_states, acts)
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

        reconstructed_obs = self.world_model.observation_model(latent_states)
        po = SymLogDistribution(reconstructed_obs, dims=len(reconstructed_obs.shape[2:]))
        pr = TwoHotEncodingDistribution(self.world_model.reward_model(latent_states), dims=1)
        pc = Independent(BernoulliSafeMode(logits=self.world_model.continue_model(latent_states)), 1)

        priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.stoch_size, self.disc_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], self.stoch_size, self.disc_size)

        return (po, pr, pc, priors_logits, posteriors_logits,
                recurrent_states, posteriors)
