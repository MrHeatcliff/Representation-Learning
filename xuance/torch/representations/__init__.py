from .mlp import Basic_Identical, Basic_MLP
from .cnn import Basic_CNN, AC_CNN_Atari
from .rnn import Basic_RNN
from .vit import Basic_ViT
from .hierarchical_dreamer import HierarchicalDreamerWorldModel

REGISTRY_Representation = {
    "Basic_Identical": Basic_Identical,
    "Basic_MLP": Basic_MLP,
    "Basic_CNN": Basic_CNN,
    "AC_CNN_Atari": AC_CNN_Atari,
    "Basic_RNN": Basic_RNN,
    "Basic_ViT": Basic_ViT,
    "HierarchicalDreamerWorldModel": HierarchicalDreamerWorldModel,
}

__all__ = [
    "REGISTRY_Representation",
    "Basic_Identical", "Basic_MLP",
    "Basic_CNN", "AC_CNN_Atari",
    "Basic_RNN", "Basic_ViT", "HierarchicalDreamerWorldModel"
]
