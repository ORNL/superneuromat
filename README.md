# SuperNeuroMAT v1.4

SuperNeuroMAT is a matrix-based simulator for simulating spiking neural networks to be used in neuromorphic computing.

Documentation available at: https://superneuromat.readthedocs.io/en/latest/


## Installation
1. Install using `pip install superneuromat`
2. Update/upgrade using `pip install superneuromat --upgrade`


## Usage
1. In a Python script or on a Python interpreter, do `import superneuromat as snm`
2. The main class can be accessed by `model = snm.NeuromorphicModel()`
3. Refer to docstrings in the source code for the API


## For Development
1. Clone the `superneuromat` repo: `git clone https://github.com/ORNL/superneuromat.git`
2. Install the package as editable inside your virtual environment: `pip install -e superneuromat/`

**Warning:** importing as a relative module (i.e. not installed in your environment) has been deprecated.


## Directory Info
1. `superneuromat`: This contains the source code for `superneuromat`
2. `tests`: This contains unittests for development purposes. Please ignore!


