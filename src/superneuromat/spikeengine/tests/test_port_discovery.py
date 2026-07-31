"""Tests for UART port discovery (audit finding #7, 2026-07-31).

Two defects: SNM_FPGA_PORT was named in three error messages but never read,
and interface 1 was hardcoded even though the same file acknowledged that some
boards use another interface.
"""

from superneuromat.spikeengine import _transport as t


class _Info:
    """Stands in for pyserial's ListPortInfo."""
    def __init__(self, device, iface=None, desc=""):
        self.device = device
        self.description = desc
        self.location = None if iface is None else f"1-1:1.{iface}"
        self.vid, self.pid = 0x0403, 0x6011


def test_snm_fpga_port_env_override_is_honoured(monkeypatch):
    """The variable is advertised in error messages, so it must actually work."""
    monkeypatch.setenv("SNM_FPGA_PORT", "COM_OVERRIDE")
    assert t.autodetect_port(required=True) == "COM_OVERRIDE"


def test_env_override_wins_without_any_board_present(monkeypatch):
    """The override must short-circuit detection, not merely reorder it --
    otherwise it is useless in exactly the case it exists for (a board whose
    VID:PID is not in the table)."""
    monkeypatch.setenv("SNM_FPGA_PORT", "/dev/ttyCUSTOM")
    monkeypatch.setattr(t, "_board_port_infos", list)
    assert t.autodetect_port(required=True) == "/dev/ttyCUSTOM"


def test_no_env_override_falls_through_to_detection(monkeypatch):
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    monkeypatch.setattr(t, "_board_port_infos", lambda: [_Info("COM9")])
    assert t.autodetect_port(required=True) == "COM9"


def test_default_prefers_ftdi_interface_1(monkeypatch):
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    infos = [_Info("COM1", iface=0), _Info("COM2", iface=1)]
    monkeypatch.setattr(t, "_board_port_infos", lambda: infos)
    assert t.autodetect_port(required=True) == "COM2"


def test_board_can_override_the_uart_interface(monkeypatch):
    """A board whose runtime UART is not on interface 1 must be able to say
    so; previously interface 1 was hardcoded for every board."""
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    infos = [_Info("COM1", iface=0), _Info("COM2", iface=1),
             _Info("COM3", iface=2)]
    monkeypatch.setattr(t, "_board_port_infos", lambda: infos)
    assert t.autodetect_port(required=True, prefer_interface=2) == "COM3"
    assert t.autodetect_port(required=True, prefer_interface=0) == "COM1"


def test_duplicate_preferred_interfaces_are_probed_in_preference_order(monkeypatch):
    """Interface numbers repeat across physical USB bridges. If two boards
    both expose channel D/interface 3, auto-detection must probe rather than
    blindly selecting the first interface-3 node."""
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    infos = [_Info("COM5", iface=1), _Info("COM7", iface=3),
             _Info("COM13", iface=3)]
    monkeypatch.setattr(t, "_board_port_infos", lambda: infos)
    attempts = []

    def _probe(port, baud):
        attempts.append((port, baud))
        return port == "COM13" and baud == 4_000_000

    monkeypatch.setattr(t, "_probe_port", _probe)
    assert t.autodetect_port(required=True, prefer_interface=3) == "COM13"
    assert [port for port, _ in attempts[:4]] == [
        "COM7", "COM7", "COM7", "COM13"]


def test_transport_baud_prevents_wrong_speed_probe_false_positive(monkeypatch):
    """A debug/JTAG channel may answer a status-shaped frame at another baud.
    It must not win detection when the runtime will reopen it at 4 Mbaud."""
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    infos = [_Info("COM7", iface=3), _Info("COM13", iface=3)]
    monkeypatch.setattr(t, "_board_port_infos", lambda: infos)
    attempts = []

    def _probe(port, baud):
        attempts.append((port, baud))
        return ((port, baud) == ("COM7", 115_200)
                or (port, baud) == ("COM13", 4_000_000))

    monkeypatch.setattr(t, "_probe_port", _probe)
    assert t.autodetect_port(
        required=True, prefer_interface=3, baud=4_000_000) == "COM13"
    assert attempts == [("COM7", 4_000_000), ("COM13", 4_000_000)]


def test_known_interface_does_not_fall_through_to_another_board(monkeypatch):
    """If the requested board is offline, report that fact instead of finding
    a responsive UART on a different interface/physical board."""
    monkeypatch.delenv("SNM_FPGA_PORT", raising=False)
    infos = [_Info("COM5", iface=1), _Info("COM13", iface=3)]
    monkeypatch.setattr(t, "_board_port_infos", lambda: infos)
    attempts = []

    def _probe(port, baud):
        attempts.append((port, baud))
        return port == "COM13"

    monkeypatch.setattr(t, "_probe_port", _probe)
    assert t.autodetect_port(
        required=True, prefer_interface=1, baud=4_000_000) == "COM5"
    # A unique metadata match is selected directly; critically, COM13 is never
    # considered as a fallback for this board.
    assert attempts == []


def test_boards_expose_the_interface_field():
    """StdpBoard must carry the field connect() plumbs through."""
    from superneuromat.spikeengine import BOARDS
    for b in BOARDS.values():
        assert hasattr(b, "uart_interface_index")


def test_connect_passes_the_board_interface_to_the_device(monkeypatch):
    """End-to-end wiring: catalogue -> connect() -> device -> transport."""
    import superneuromat.spikeengine as se

    seen = {}

    class _Fake:
        def __init__(self, port, baud, **kw):
            seen.update(kw)

    monkeypatch.setattr(se, "InferEngineStdpDevice", _Fake)
    se.connect(port="COM_TEST", board="zcu104")
    assert "prefer_interface" in seen, "board interface not plumbed to the device"
    assert seen["stdp_window"] == se.get_board("zcu104").stdp_window
