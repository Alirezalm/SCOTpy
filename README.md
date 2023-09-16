# Sparse Convex Optimization Toolbox Python API (SCOTpy)

## Overview
`scotpy` is a lightweight library designed to help you build and solve Sparse Convex Optimization (SCO) problems on computational networks with N nodes. It relies on the availability of the `SCOT` solver executable and associated shared libraries.

For more detailed information about the `SCOT` solver, including compilation instructions and additional details, please visit the [SCOT repository](https://github.com/Alirezalm/scot).

## Installation
### Supported Platforms

**SCOT** is compatible with the following platforms:

1. Ubuntu 20.04 or higher
2. macOS
3. Windows Subsystem for Linux (WSL2)

### Dependencies

To set up **SCOT**, you need the following prerequisites:

1. **SCOT**: Ensure ```SCOT``` is installed on your OS. [Installation Guide](https://github.com/Alirezalm/scot)
2. **Gurobi Optimizer**: Version 10 or higher is required. [Quick Start Guide](https://www.gurobi.com/documentation/quickstart.html).
3. **Message Passing Interface (MPI)**:
   - [OpenMPI](https://www.open-mpi.org/)
   - [MPICH](https://www.mpich.org/)
   - [Microsoft MPI (for Windows)](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
4. See ```requirements.txt``` for other dependencies.

### Usage
 See examples.
