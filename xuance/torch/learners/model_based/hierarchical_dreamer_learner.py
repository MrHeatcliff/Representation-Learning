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
