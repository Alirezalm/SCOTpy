from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class AlgorithmType(Enum):
    DIPOA = 0
    DIHOA = 1