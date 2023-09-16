import json
import os
import pathlib
import shutil
import subprocess
from typing import List

from numpy import ndarray
from sklearn.preprocessing import normalize
from numpy import array
from scotpy_decorators import validate_arguments
from scotpy_types import ProblemType, AlgorithmType, ScotSettings

HOME = os.environ.get("HOME", "")

ROOT = "scotpy"
INPUT = "inputs"
OUTPUT = "output"

WORKING_DIR = os.path.join(HOME, ROOT)

pathlib.Path(WORKING_DIR).mkdir(exist_ok=True)


class ScotModel:
    """
    A class representing a problem model in the Scot system.

    Attributes:
        name (str): The name of the problem.
        rank (int): The rank of the problem.
        ptype (ProblemType): The type of the problem (classification or regression).
        kappa (int): A parameter value for the problem.
        problem_dict (dict): A dictionary representation of the problem model.

    Methods:
        __init__(self, problem_name: str, rank: int, kappa: int, ptype: ProblemType = ProblemType.CLASSIFICATION):
            Constructor method to initialize the ScotModel instance.

        set_data(self, samples: ndarray, response: ndarray, normalized_data: bool = True):
            Set the data for the problem model.

        create(self):
            Create the problem model and save it as a JSON file.

    """

    def __init__(self, problem_name: str, rank: int, kappa: int, ptype: ProblemType = ProblemType.CLASSIFICATION):
        """
        Constructor method to initialize the ScotModel instance.

        Args:
            problem_name (str): The name of the problem.
            rank (int): The rank of the problem.
            kappa (int): A parameter value for the problem.
            ptype (ProblemType, optional): The type of the problem (classification or regression). Defaults to ProblemType.CLASSIFICATION.
        """

        self.name = problem_name
        self.rank = rank
        self.ptype = ptype

        self.__samples = None
        self.__response = None
        self._n_samples = None
        self._n_features = None
        self.kappa = kappa
        self.problem_dict = {}

    def set_data(self, samples: ndarray, response: ndarray, normalized_data: bool = True):
        """
        Set the data for the problem model.

        Args:
            samples (ndarray): The input feature data as a 2D NumPy array.
            response (ndarray): The response variable data as a 1D NumPy array.
            normalized_data (bool, optional): Flag indicating whether the input data is normalized. Defaults to True.
        """
        self.__samples = samples
        self.__response = response

        if normalized_data:
            self.__normalize_data()

        self._n_samples, self._n_features = self.__samples.shape

        self.__response.reshape(self._n_samples, 1)

    def create(self):
        """
        Create the problem model and save it as a JSON file.
        """
        self.__build()
        self.__save_to_json_file()

    def __build(self):
        """
        Build the problem model as a dictionary.
        """
        self.problem_dict["name"] = self.name
        self.problem_dict["version"] = "0.1"

        if self.ptype == ProblemType.CLASSIFICATION:
            self.problem_dict["type"] = "classification"
        elif self.ptype == ProblemType.REGRESSION:
            self.problem_dict["type"] = "regression"
        else:
            raise ValueError("unknown problem")

        self.problem_dict["response"] = [float(x) for x in self.__response]

        self.problem_dict['samples'] = []

        for row_index, row_value in enumerate(self.__samples):
            self.problem_dict["samples"].append([float(x) for x in row_value])

    def __save_to_json_file(self):
        """
        Save the problem model as a JSON file.
        """
        to_json = json.dumps(self.problem_dict,
                             sort_keys=True,
                             indent=4,
                             separators=(',', ': '))

        file_name = f"node_{self.rank}_{self.name}.dist.json"

        pathlib.Path(os.path.join(WORKING_DIR, INPUT)).mkdir(exist_ok=True)

        target = os.path.join(WORKING_DIR, INPUT, file_name)

        with open(target, 'w') as jsonwriter:
            jsonwriter.write(to_json)

        print(target, "created.")

    def __normalize_data(self):
        """
        Normalize the input feature data using L2 normalization.
        """
        self.__samples = normalize(self.__samples, norm='l2')


class ScotPy:

    def __init__(self, models: List[ScotModel], settings: ScotSettings):

        self.models = models
        self.settings = settings
        self.cmd_args = []
        if os.environ.get("SCOT_HOME", "") == "":
            raise ScotPyException(
                "Solver path is unknown. make sure the executable is in the system path.")

        if os.environ.get("GUROBI_HOME", "") == "":
            raise ScotPyException(
                "gurobi path is unknown. make sure GUROBI_HOME is in the system path."
            )

        self.scot_path = os.environ.get("SCOT_HOME", "")
        self.total_size = len(models)

        self.__generate_mpi_cmd()

    def run(self):
        # print(" ".join(self.cmd_args))
        command_return = subprocess.run(self.cmd_args)
        return_code = command_return.returncode
        if return_code > 0:
            raise ScotPyException("SCOT FAILED")

        execution_time, objval, solution = self.process_solver_out()

        return objval, solution, execution_time

    def process_solver_out(self):
        pathlib.Path(os.path.join(WORKING_DIR, OUTPUT)).mkdir(exist_ok=True)
        current_dir = os.path.dirname(os.path.realpath(__file__))
        output_dir = os.path.join(WORKING_DIR, OUTPUT)
        filename = "rank_0_output.json"
        out_file = os.path.join(current_dir, filename)
        try:
            shutil.copy(out_file, os.path.join(output_dir, self.models[0].name + filename))

            for i in range(self.total_size):
                os.remove(f"rank_{i}_output.json")

            with open(os.path.join(output_dir, self.models[0].name + filename)) as file:
                opt_result = json.load(file)
                objval = opt_result["objval"]
                execution_time = opt_result["time"]
                solution = array(opt_result["x"], dtype=float)

        except Exception as exp:
            raise ScotPyException(exp)

        return objval, solution, execution_time

    def __generate_mpi_cmd(self):
        mpi_path = None

        if shutil.which("mpiexec") is not None:
            mpi_path = shutil.which("mpiexec")

        elif shutil.which("mpirun") is not None:
            mpi_path = shutil.which("mpirun")

        else:
            raise ScotPyException("Make sure MPI binaries are installed.")

        input_path = os.path.join(WORKING_DIR, INPUT)
        input_name = self.models[0].name

        n_nonzeros = self.models[0].kappa

        algorithm = self.settings.algorithm

        alg = ""

        if algorithm == AlgorithmType.DIPOA:
            alg = "dipoa"

        elif algorithm == AlgorithmType.DIHOA:
            alg = "dihoa"

        else:
            raise ScotPyException("Unknown algorithm.")

        time_limit = self.settings.time_limit
        r_gap = self.settings.relative_gap
        ub = self.settings.ub

        if self.settings.verbose:
            verbose = "--verbose"

            self.cmd_args = [
                mpi_path,
                "-n",
                str(self.total_size),
                str(os.path.join(self.scot_path, "scot")),
                f"--dir={input_path}",
                f"--input={input_name}",
                f"--nz={n_nonzeros}",
                f"--alg={alg}",
                f"--tlim={time_limit}",
                f"--rgap={r_gap}",
                verbose,
                f"--ub={ub}"
            ]
        else:
            self.cmd_args = [
                mpi_path,
                "-n",
                str(self.total_size),
                str(os.path.join(self.scot_path, "scot")),
                f"--dir={input_path}",
                f"--input={input_name}",
                f"--nz={n_nonzeros}",
                f"--alg={alg}",
                f"--tlim={time_limit}",
                f"--rgap={r_gap}",
                f"--ub={ub}"
            ]


class ScotPyException(Exception):
    pass
