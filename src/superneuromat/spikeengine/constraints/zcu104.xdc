# SuperNeuroMAT3 FPGA - Xilinx Zynq UltraScale+ ZCU104 (xczu7ev-ffvc1156-2-e).
#
# Pins are from the OFFICIAL ZCU104 Rev1.0 Master XDC (rdf0438) - NOT invented.
# PS/PL decision = Option A (PL-only): the ZCU104 exposes a PL clock (CLK_125) and a
# PL-routed UART (UART2, via the onboard USB-UART), so the design runs entirely in the
# PL over UART -- no PS/AXI needed. Top module: spikeengine_zcu104_top (board wrapper:
# IBUFDS + MMCME4 125MHz->100MHz; active-high GPIO_PB_SW0 reset).
#
# STATUS: build-ready, NOT yet hardware-validated on a physical ZCU104. Verify on silicon
# before treating as a validated bitstream.

## ---- System clock: 125 MHz LVDS differential (Bank 28, 1.8V) -------------------------
## The MMCM in the wrapper divides this to the core's 100 MHz; only the 125 MHz input
## clock is declared here (Vivado derives the 100 MHz generated clock from the MMCM).
set_property -dict { PACKAGE_PIN F23 IOSTANDARD LVDS } [get_ports { CLK_125_P }]
set_property -dict { PACKAGE_PIN E23 IOSTANDARD LVDS } [get_ports { CLK_125_N }]
create_clock -add -name sys_clk_pin -period 8.000 -waveform {0 4.0} [get_ports { CLK_125_P }]

## ---- Reset push-button GPIO_PB_SW0 (active-HIGH, Bank 88 3.3V) -----------------------
set_property -dict { PACKAGE_PIN B4 IOSTANDARD LVCMOS33 } [get_ports { GPIO_PB_SW0 }]

## ---- USB-UART via PL-routed UART2 (Bank 28 1.8V) ------------------------------------
## FPGA uart_rx = UART2_TXD_FPGA_RXD (host->FPGA input); uart_tx = UART2_RXD_FPGA_TXD.
set_property -dict { PACKAGE_PIN A20 IOSTANDARD LVCMOS18 } [get_ports { uart_rx }]
set_property -dict { PACKAGE_PIN C19 IOSTANDARD LVCMOS18 } [get_ports { uart_tx }]

## ---- Spike LEDs led[0..3] -> GPIO_LED_0..3_LS (Bank 88 3.3V) ------------------------
## The core drives 16 spike LEDs; ZCU104 exposes 4 PL LEDs, so led[15:4] are unused.
set_property -dict { PACKAGE_PIN D5 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN D6 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN A5 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN B5 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]

## ---- Async I/O false paths (reset button + UART are async to the core clock) --------
set_false_path -from [get_ports {GPIO_PB_SW0 uart_rx}]
set_false_path -to   [get_ports {uart_tx led[*]}]

## ---- high-fanout power-up reset: relax recovery/removal pessimism --------------------
set_false_path -from [get_cells u_top/rst_sync_reg[1]]
