"""Cell-cell communication network inference (ligand-receptor permutation testing)."""

from .cellcomm import CellCommResult, InteractionResult, infer_cell_communication
from .lr_pairs import LIGAND_RECEPTOR_PAIRS, LR_PAIRS
from .synthetic import generate_demo_dataset

__all__ = [
    "CellCommResult",
    "InteractionResult",
    "infer_cell_communication",
    "LIGAND_RECEPTOR_PAIRS",
    "LR_PAIRS",
    "generate_demo_dataset",
]
