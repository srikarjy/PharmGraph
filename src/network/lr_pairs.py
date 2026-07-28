"""Loads the bundled curated ligand-receptor pair database.

A small, real, well-known subset (~48 pairs spanning cytokine, chemokine,
costimulatory/checkpoint, adhesion, and growth-factor signaling), curated from
CellPhoneDB v4 (Efremova et al. 2020, Nat Protoc) and CellTalkDB/connectomeDB2020
(Cabello-Aguilar et al. 2020, NAR). Not exhaustive -- for production use, point
at the full CellPhoneDB or CellTalkDB databases instead.
"""

import csv
from pathlib import Path
from typing import List, NamedTuple, Tuple

_CSV_PATH = Path(__file__).parent / "resources" / "ligand_receptor_pairs.csv"


class LRPair(NamedTuple):
    ligand: str
    receptor: str
    pathway: str
    source: str


def _load_lr_pairs(csv_path: Path = _CSV_PATH) -> List[LRPair]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            LRPair(row["ligand"], row["receptor"], row["pathway"], row["source"])
            for row in reader
        ]


LR_PAIRS: List[LRPair] = _load_lr_pairs()
LIGAND_RECEPTOR_PAIRS: List[Tuple[str, str]] = [(p.ligand, p.receptor) for p in LR_PAIRS]
