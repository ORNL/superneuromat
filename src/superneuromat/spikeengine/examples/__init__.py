"""Runnable end-to-end examples for the SpikeEngine FPGA accelerator.

Each example ships as an importable module and, for the two main ones, an
equivalent notebook:

  * ``digits_stdp_e2e``   -- sklearn digits: on-chip STDP training (bit-exact
    against SuperNeuroMAT) followed by on-chip rate-readout inference.
  * ``citation_gnn_fpga`` -- citation-graph SNN classifier on hardware, with a
    fixed-point-faithful software reference computed per paper for comparison.

Both accept ``--no-hardware`` to run the software reference alone, so they are
usable without a board attached. The citation example additionally needs the
external ``sgnn-superneuro`` dataset checkout, located via the ``SGNN_REPO``
environment variable; it is not bundled with this package.
"""
