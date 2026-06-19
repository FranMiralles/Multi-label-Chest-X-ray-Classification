from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class OptunaConfig:
    n_trials: int = 5
    n_epochs: int = 8
    patience: int = 3


@dataclass
class TrainingConfig:
    lr: float          = 1e-4
    batch_size: int    = 64
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float    = 0.05
    weight_decay: float = 0.0
    scheduler: str     = 'cosine'
    rotation: int      = 15
    translate: float   = 0.10
    n_epochs: int      = 12
    patience: int      = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        return cls(
            lr=d['lr'],
            batch_size=d['batch_size'],
            asl_gamma_neg=d['asl_gamma_neg'],
            asl_gamma_pos=d['asl_gamma_pos'],
            asl_clip=d['asl_clip'],
            weight_decay=d['weight_decay'],
            scheduler=d['scheduler'],
            rotation=d['rotation'],
            translate=d['translate'],
        )
