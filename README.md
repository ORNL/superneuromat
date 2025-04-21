# SuperNeuroMAT v2.0.0

SuperNeuroMAT is a matrix-based simulator for simulating spiking neural networks, which are used in neuromorphic computing. It is one of the fastest, if not the fastest, simlators for simulating spiking neural networks.

Some salient features of SuperNeuroMAT are:
1. Support for leaky integrate and fire neuron model with the following parameters: neuron threshold, neuron leak, and neuron refractory period
2. Support for Spiking-Time-Dependent Plasticity (STDP) synapses with weights and delays
3. No restrictions on connectivity of the neurons, all-to-all connections as well as self connections possible
4. Constant leak supported
5. STDP learning can be configured to turn on/off positive updates and negative updates
6. Excessive synaptic delay can slow down the execution of the simulation, so try to avoid as much as possible
7. Leak refers to the constant amount by which the internal state (membrane potential) of a neuron changes in each time step of the simulation; therefore, zero leak means the neuron fully retains the value in its internal state, and infinite leak means the neuron never retains the value in its internal state
8. STDP implementation is extremely fast
9. The model of neuromorphic computing supported in SuperNeuroMAT is Turing-complete
10. All underlying computational operations are matrix-based and currently supported on CPUs


## Installation
1. Install using `pip install superneuromat`
2. Update/upgrade using `pip install superneuromat --upgrade`


## Usage
1. In a Python script or on a Python interpreter, do `import superneuromat as snm`
2. The main class can be accessed by `snn = snm.SNN()`
3. Refer to docstrings in the source code or on the [readthedocs](https://superneuromat.readthedocs.io/en/latest/) page for the API


## Documentation
Documentation available at: https://superneuromat.readthedocs.io/en/latest/


## Citation
1. Please cite SuperNeuroMAT using:
	```
	@inproceedings{date2023superneuro,
	  title={SuperNeuro: A fast and scalable simulator for neuromorphic computing},
	  author={Date, Prasanna and Gunaratne, Chathika and R. Kulkarni, Shruti and Patton, Robert and Coletti, Mark and Potok, Thomas},
	  booktitle={Proceedings of the 2023 International Conference on Neuromorphic Systems},
	  pages={1--4},
	  year={2023}
	}
	```
2. References for SuperNeuroMAT:
	- [SuperNeuro: A Fast and Scalable Simulator for Neuromorphic Computing](https://dl.acm.org/doi/abs/10.1145/3589737.3606000)
	- [Neuromorphic Computing is Turing-Complete](https://dl.acm.org/doi/abs/10.1145/3546790.3546806)
	- [Computational Complexity of Neuromorphic Algorithms](https://dl.acm.org/doi/abs/10.1145/3477145.3477154)



## For Development
1. Clone the `superneuromat` repo: `git clone https://github.com/ORNL/superneuromat.git`
2. Install the package as editable inside your virtual environment: `pip install -e superneuromat/`

**Warning:** importing as a relative module (i.e. not installed in your environment) has been deprecated.


## Directory Info
1. `superneuromat`: This contains the source code for `superneuromat`
2. `tests`: This contains unittests for development purposes. Please ignore!


