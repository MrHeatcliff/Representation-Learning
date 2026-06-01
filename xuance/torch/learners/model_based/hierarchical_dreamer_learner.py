import torch
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from xuance.torch.utils import kl_div
from xuance.torch.learners.model_based.dreamer_v3_learner import DreamerV3_Learner
from xuance.torch.policies.hierarchical_dreamer import HierarchicalDreamerPolicy
from xuance.common import Tuple, Union
from argparse import Namespace


class HierarchicalDreamer_Learner(DreamerV3_Learner):
    """Research fork of the DreamerV3 learner.

    Add hierarchy-specific losses, optimizers, schedules, or logging here. The
    initial implementation intentionally delegates to DreamerV3_Learner.
    """

    def __init__(
        self,
        config: Namespace,
        policy: HierarchicalDreamerPolicy,
        action_shape: Union[int, Tuple[int, ...]],
        callback,
    ):
        super(HierarchicalDreamer_Learner, self).__init__(config, policy, action_shape, callback)
        self.training_regime = self.config.training_regime
        self._configure_training_regime_optimizer()

    def _configure_training_regime_optimizer(self):
        encoder_params = list(self.policy.world_model.encoder.parameters())
        encoder_param_ids = {id(param) for param in encoder_params}
        world_model_params = [
            param for param in self.policy.world_model.parameters()
            if id(param) not in encoder_param_ids
        ]
        if self.config.harmony:
            world_model_params += [
                self.policy.harmonizer_s1.get_harmony(),
                self.policy.harmonizer_s2.get_harmony(),
                self.policy.harmonizer_s3.get_harmony()
            ]
        hierarchy_params = list(self.policy.hierarchical_latent_parameters())

        if self.training_regime.freeze_encoder:
            for param in encoder_params:
                param.requires_grad_(False)

        base_lr = self.config.learning_rate_model
        encoder_lr = base_lr * self.training_regime.encoder_lr_scale
        hierarchy_lr = base_lr * self.training_regime.hierarchy_lr_scale
        param_groups = []
        if world_model_params:
            param_groups.append({"params": world_model_params, "lr": base_lr, "name": "world_model"})
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
        if hierarchy_params:
            param_groups.append({"params": hierarchy_params, "lr": hierarchy_lr, "name": "hierarchy"})
        self.optimizer["model"] = torch.optim.Adam(param_groups)

    def _in_world_model_only_phase(self):
        return (
            self.training_regime.wm_only_phase1 and
            self.gradient_step < self.training_regime.phase1_gradient_steps
        )

    def _auxiliary_loss_weights(self):
        weights = self.config.hierarchical_latent.loss_weights
        if not self._in_world_model_only_phase():
            return weights
        return type(weights)({
            "hierarchical": 0.0,
            "sparse_dynamics": 0.0,
            "temporal": 0.0,
            "variance_covariance": 0.0,
            "sparsity": 0.0,
        })

    def update(self, **samples):
        if self.gradient_step % self.soft_update_freq == 0:
            self.policy.soft_update(self.tau)

        obs = torch.as_tensor(samples['obs'], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(samples['acts'], device=self.device)
        if not self.is_continuous:
            acts = torch.nn.functional.one_hot(acts.long(), num_classes=self.action_shape).float()
        rews = torch.as_tensor(samples['rews'], device=self.device)
        terms = torch.as_tensor(samples['terms'], device=self.device)
        truncs = torch.as_tensor(samples['truncs'], device=self.device)
        is_first = torch.as_tensor(samples['is_first'], device=self.device)

        is_first[0, :] = torch.ones_like(is_first[0, :])
        acts = torch.cat((torch.zeros_like(acts[:1]), acts[:-1]), 0)
        cont = 1 - terms

        info = self.callback.on_update_start(self.gradient_step,
                                             policy=self.policy, obs=obs, act=acts,
                                             is_first=is_first, rew=rews, termination=terms, truncation=truncs,
                                             cont=cont)

        phase1_active = self._in_world_model_only_phase()
        self.policy.hierarchy_training_active = not phase1_active
        po, pr, pc, priors_logits, posteriors_logits, recurrent_states, posteriors = \
            self.policy.model_forward(obs, acts, is_first)

        observation_loss = -po.log_prob(obs)
        reward_loss = -pr.log_prob(rews)
        dyn_loss = kl_div(
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
        )
        free_nats = torch.full_like(dyn_loss, self.config.world_model.kl_free_nats)
        dyn_loss = torch.maximum(dyn_loss, free_nats)
        repr_loss = kl_div(
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
        )
        repr_loss = torch.maximum(repr_loss, free_nats)

        if pc is not None and cont is not None:
            continue_loss = self.continue_scale_factor * -pc.log_prob(cont)
        else:
            continue_loss = torch.zeros_like(reward_loss)

        if self.config.harmony:
            repr_loss *= self.kl_representation / (self.kl_representation + self.kl_dynamic)
            dyn_loss *= self.kl_dynamic / (self.kl_representation + self.kl_dynamic)
            kl_loss = dyn_loss + repr_loss
            observation_loss = self.policy.harmonizer_s1(observation_loss)
            reward_loss = self.policy.harmonizer_s2(reward_loss)
            kl_loss = self.policy.harmonizer_s3(kl_loss)
        else:
            repr_loss *= self.kl_representation
            dyn_loss *= self.kl_dynamic
            kl_loss = dyn_loss + repr_loss
            kl_loss *= self.kl_regularizer

        hierarchical_loss = self.policy.latest_hierarchical_recon_loss
        if hierarchical_loss is None:
            hierarchical_loss = torch.zeros((), device=self.device)
        sparse_dynamics_loss = self.policy.latest_sparse_dynamics_loss
        if sparse_dynamics_loss is None:
            sparse_dynamics_loss = torch.zeros((), device=self.device)
        temporal_loss = self.policy.latest_temporal_contrastive_loss
        if temporal_loss is None:
            temporal_loss = torch.zeros((), device=self.device)
        vicreg_loss = self.policy.latest_vicreg_loss
        if vicreg_loss is None:
            vicreg_loss = torch.zeros((), device=self.device)
        sparsity_loss = self.policy.latest_sparsity_loss
        if sparsity_loss is None:
            sparsity_loss = torch.zeros((), device=self.device)
        loss_weights = self._auxiliary_loss_weights()
        wm_loss = (kl_loss + observation_loss + reward_loss + continue_loss).mean()
        weighted_hierarchical_loss = loss_weights.hierarchical * hierarchical_loss
        weighted_sparse_dynamics_loss = loss_weights.sparse_dynamics * sparse_dynamics_loss
        weighted_temporal_loss = loss_weights.temporal * temporal_loss
        weighted_vicreg_loss = loss_weights.variance_covariance * vicreg_loss
        weighted_sparsity_loss = loss_weights.sparsity * sparsity_loss
        model_loss = (
            wm_loss +
            weighted_hierarchical_loss +
            weighted_sparse_dynamics_loss +
            weighted_temporal_loss +
            weighted_vicreg_loss +
            weighted_sparsity_loss
        )

        self.optimizer['model'].zero_grad()
        model_loss.backward()
        if self.config.world_model.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.world_model.parameters(), self.config.world_model.clip_gradients)
            torch.nn.utils.clip_grad_norm_(self.policy.hierarchical_latent_parameters(),
                                           self.config.world_model.clip_gradients)
        self.optimizer['model'].step()

        out = self.policy.actor_critic_forward(posteriors, recurrent_states, terms)
        objective, discount, entropy = out['for_actor']
        qv, predicted_target_values, lambda_values = out['for_critic']
        actor_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))

        self.optimizer['actor'].zero_grad()
        actor_loss.backward()
        if self.config.actor.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.actor.clip_gradients)
        self.optimizer['actor'].step()

        critic_loss = -qv.log_prob(lambda_values.detach())
        critic_loss = critic_loss - qv.log_prob(predicted_target_values.detach())
        critic_loss = torch.mean(critic_loss * discount[:-1].squeeze(-1))
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.config.critic.clip_gradients is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.critic.clip_gradients)
        self.optimizer['critic'].step()

        self.gradient_step += 1

        info.update({
            "model_loss/model_loss": model_loss.item(),
            "model_loss/obs_loss": observation_loss.mean().item(),
            "model_loss/rew_loss": reward_loss.mean().item(),
            "model_loss/continue_loss": continue_loss.mean().item(),
            "model_loss/kl_loss": kl_loss.mean().item(),
            "model_loss/world_model_loss": wm_loss.item(),
            "model_loss/hierarchical_recon_loss": hierarchical_loss.item(),
            "model_loss/sparse_dynamics_loss": sparse_dynamics_loss.item(),
            "model_loss/temporal_contrastive_loss": temporal_loss.item(),
            "model_loss/variance_covariance_loss": vicreg_loss.item(),
            "model_loss/sparsity_loss": sparsity_loss.item(),
            "model_loss/weighted_hierarchical_recon_loss": weighted_hierarchical_loss.item(),
            "model_loss/weighted_sparse_dynamics_loss": weighted_sparse_dynamics_loss.item(),
            "model_loss/weighted_temporal_loss": weighted_temporal_loss.item(),
            "model_loss/weighted_variance_covariance_loss": weighted_vicreg_loss.item(),
            "model_loss/weighted_sparsity_loss": weighted_sparsity_loss.item(),
            "training_regime/phase1_active": float(phase1_active),
            "training_regime/encoder_lr": next(
                group["lr"] for group in self.optimizer["model"].param_groups if group.get("name") == "encoder"
            ),
            "training_regime/hierarchy_lr": next(
                group["lr"] for group in self.optimizer["model"].param_groups if group.get("name") == "hierarchy"
            ),

            "actor_loss/actor_loss": actor_loss.item(),
            "actor_loss/reinforce_loss": objective.mean().item(),
            "actor_loss/entropy_loss": entropy.unsqueeze(dim=-1)[:-1].mean().item(),

            "critic_loss/critic_loss": critic_loss.item(),
            "critic_loss/lambda_values": lambda_values.mean().item(),

            "step/gradient_step": self.gradient_step
        })
        if self.config.harmony:
            info.update({'harmonizer/s1': self.policy.harmonizer_s1.get_harmony().item(),
                         'harmonizer/s2': self.policy.harmonizer_s2.get_harmony().item(),
                         'harmonizer/s3': self.policy.harmonizer_s3.get_harmony().item()})

        info.update(self.policy.latest_sparse_latent_info)
        info.update(self.callback.on_update_end(self.gradient_step,
                                                policy=self.policy, info=info,
                                                po=po, pr=pr, pc=pc, priors_logits=priors_logits,
                                                posteriors_logits=posteriors_logits, recurrent_states=recurrent_states,
                                                posteriors=posteriors, observation_loss=observation_loss,
                                                reward_loss=reward_loss, dyn_loss=dyn_loss, free_nats=free_nats,
                                                repr_loss=repr_loss, kl_loss=kl_loss, continue_loss=continue_loss,
                                                model_loss=model_loss, out=out,
                                                actor_loss=actor_loss, critic_loss=critic_loss,
                                                hierarchical_loss=hierarchical_loss,
                                                sparse_dynamics_loss=sparse_dynamics_loss,
                                                temporal_loss=temporal_loss,
                                                vicreg_loss=vicreg_loss,
                                                sparsity_loss=sparsity_loss))
        return info
