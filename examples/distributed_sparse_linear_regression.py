from typing import List

from sklearn.datasets import make_regression

from scotpy import (AlgorithmType,
                    ProblemType,
                    ScotModel,
                    ScotPy,
                    ScotSettings
                    )


def main():

    total_nodes = 4

    models: List[ScotModel] = []

    for rank in range(total_nodes):
        dataset, res = make_regression(n_samples=200, n_features=30)

        scp = ScotModel(problem_name="linear_regression", rank=rank,
                        kappa=5, ptype=ProblemType.REGRESSION)

        scp.set_data(dataset, res, normalized_data=True)

        scp.create()

        models.append(scp)

    scot_settings = ScotSettings(
        relative_gap=1e-4,
        time_limit=10000,
        verbose=True,
        algorithm=AlgorithmType.DIPOA
    )

    solver = ScotPy(models, scot_settings)
    solver.run()


if __name__ == '__main__':
    main()
