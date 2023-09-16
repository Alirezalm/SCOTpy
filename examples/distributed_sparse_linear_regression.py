from typing import List

from sklearn.datasets import make_regression

from scotpy.scotpy import (AlgorithmType,
                           ProblemType,
                           ScotModel,
                           ScotPy,
                           ScotSettings
                           )


def main():
    total_nodes = 4

    models: List[ScotModel] = []

    for rank in range(total_nodes):
        dataset, res = make_regression(n_samples=200, n_features=40)

        scp = ScotModel(problem_name="linear_regression", rank=rank,
                        kappa=20, ptype=ProblemType.REGRESSION)

        scp.set_data(dataset, res, normalized_data=True)

        scp.create()

        models.append(scp)

    scot_settings = ScotSettings(
        relative_gap=1e-4,
        time_limit=10000,
        verbose=False,
        algorithm=AlgorithmType.DIHOA,
        ub=1

    )

    solver = ScotPy(models, scot_settings)
    objval, solution, execution_time = solver.run()

    print(f"Optimal Objective Value: {objval}")
    print(f"Optimal Solution: {solution}")
    print(f"Execution Time: {execution_time} seconds")


if __name__ == '__main__':
    main()
