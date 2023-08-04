from dataclasses import dataclass
from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class AlgorithmType(Enum):
    DIPOA = 0
    DIHOA = 1


@dataclass
class ScotSettings:
    algorithm: AlgorithmType = AlgorithmType.DIPOA
    time_limit: float = 1e10
    relative_gap: float = 1e-5
    verbose: bool = True
