from scotpy_types import ProblemType


def validate_arguments(func):
    def wrapper(self, problem_name: str, rank: int, kappa: int, ptype: ProblemType = ProblemType.CLASSIFICATION):
        # Validate problem_name
        if not isinstance(problem_name, str) or len(problem_name.strip()) == 0:
            raise ValueError("problem_name should be a non-empty string.")

        # Validate rank
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank should be a positive integer.")

        # Validate kappa
        if not isinstance(kappa, int) or kappa < 0:
            raise ValueError("kappa should be a positive integer.")

        # Validate ptype
        if not isinstance(ptype, ProblemType):
            raise ValueError("ptype should be a valid ProblemType enum.")

        # Call the original __init__ method
        func(self, problem_name, rank, kappa, ptype)

    return wrapper
