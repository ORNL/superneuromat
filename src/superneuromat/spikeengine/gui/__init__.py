"""SpikeEngine FPGA GUI application (PyQt5) and its host toolkit.

This subpackage bundles the desktop SNN console (`snm_gui`) together with the
host modules it needs (`snm_config`, `snm_boards`, `snm_network`, `snm_driver`,
`snm_presets`, `snm_snn_io`) so the installed `spikeengine` package is a
self-contained launch target -- no external `superneuromat_FPGA1/host` checkout
is required.

The GUI dependencies are optional. Install them with
``pip install "superneuromat[gui]"``. The console entry point below imports
``snm_gui`` lazily so a base installation can give an actionable message
instead of failing at launcher import time with a raw PyQt5 traceback.
"""


def main():
    """Launch the desktop GUI, explaining how to install missing extras."""
    try:
        from .snm_gui import main as gui_main
    except ModuleNotFoundError as exc:
        optional_roots = {"PyQt5", "matplotlib", "yaml", "pyvista", "pyvistaqt"}
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in optional_roots:
            raise SystemExit(
                "SpikeEngine GUI dependencies are not installed. Run: "
                "pip install \"superneuromat[gui]\"") from None
        raise
    return gui_main()
