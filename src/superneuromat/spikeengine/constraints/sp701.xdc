# SuperNeuroMAT3 FPGA - Xilinx Spartan-7 SP701 (xc7s100fgga676-2) constraints.
#
# Pins are from the OFFICIAL SP701 Master XDC (rdf0510, rev2p0, AMD 2024) - NOT invented.
# Top module for this board: spikeengine_sp701_top (board wrapper) which turns the
# 200 MHz differential SYSCLK into the core's 100 MHz via IBUFDS + MMCM, and inverts the
# active-HIGH CPU_RESET into the core wrapper's active-high reset.
#
# STATUS: build-ready, but NOT yet hardware-validated on a physical SP701 (no board on
# hand at authoring time). Verify on silicon before treating as a validated bitstream.

## ---- System clock: 200 MHz LVDS differential (SiT9102 oscillator, Bank 33 2.5V) ------
## The MMCM in the wrapper divides this to the core's 100 MHz; only the 200 MHz input
## clock is declared here (Vivado derives the 100 MHz generated clock from the MMCM).
set_property -dict { PACKAGE_PIN AE8 IOSTANDARD LVDS_25 } [get_ports { SYSCLK_P }]
set_property -dict { PACKAGE_PIN AE7 IOSTANDARD LVDS_25 } [get_ports { SYSCLK_N }]
create_clock -add -name sys_clk_pin -period 5.000 -waveform {0 2.5} [get_ports { SYSCLK_P }]

## ---- CPU_RESET push-button (active-high, Bank 13 1.8V) -----------------------------
set_property -dict { PACKAGE_PIN AE15 IOSTANDARD LVCMOS18 } [get_ports { CPU_RESET }]

## ---- USB-UART via onboard USB-to-UART bridge (Bank 14 3.3V) ------------------------
## FPGA uart_rx is an input (host -> FPGA); uart_tx is an output (FPGA -> host).
set_property -dict { PACKAGE_PIN Y22 IOSTANDARD LVCMOS33 } [get_ports { uart_rx }]
set_property -dict { PACKAGE_PIN Y21 IOSTANDARD LVCMOS33 } [get_ports { uart_tx }]

## ---- Spike LEDs led[0..7] -> GPIO_LED0..7 (Bank 15 3.3V) ----------------------------
## The core drives 16 spike LEDs; SP701 exposes 8, so led[15:8] are left unconnected.
set_property -dict { PACKAGE_PIN J25 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN M24 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN L24 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN K25 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]
set_property -dict { PACKAGE_PIN K26 IOSTANDARD LVCMOS33 } [get_ports { led[4] }]
set_property -dict { PACKAGE_PIN M25 IOSTANDARD LVCMOS33 } [get_ports { led[5] }]
set_property -dict { PACKAGE_PIN L25 IOSTANDARD LVCMOS33 } [get_ports { led[6] }]
set_property -dict { PACKAGE_PIN H22 IOSTANDARD LVCMOS33 } [get_ports { led[7] }]

## ---- Async I/O false paths (reset button + UART are async to the core clock) --------
set_false_path -from [get_ports {CPU_RESET uart_rx}]
set_false_path -to   [get_ports {uart_tx led[*]}]

## ---- high-fanout power-up reset: relax recovery/removal pessimism --------------------
## Mirror of the Basys-3 lever: the synchronized reset (rst_sync_reg[1], inside the
## generic top instance u_top) fans out to every async CLR; it is a power-up-only path,
## safe to treat as a false path so it does not bound the core clock.
set_false_path -from [get_cells u_top/rst_sync_reg[1]]
