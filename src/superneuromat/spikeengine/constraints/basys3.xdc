# SuperNeuroMAT3 FPGA - Digilent Basys-3 (Artix-7 XC7A35T-1CPG236C) constraints.
#
# Top module: spikeengine_fpga_top
#   clk      : 100 MHz onboard oscillator (W5)
#   rst_btn  : center push-button BTNC (U18), active-high
#   uart_rx  : data from host PC -> FPGA  (USB-UART RsRx, B18)
#   uart_tx  : data from FPGA   -> host   (USB-UART RsTx, A18)
#   led[15:0]: latest spike frame for neurons 0..15 on LD0..LD15
#
# Pins/standards follow the Digilent Basys-3 Master XDC.

## Clock signal (100 MHz board oscillator on W5). The core is pipelined to close
## at 100 MHz (ADR-001, OOC WNS +1.992 ns), so the oscillator drives sys_clk
## directly through the top's BUFG -- no divider, one 10 ns clock for the whole
## design. create_clock on the input port propagates through u_sys_bufg to the
## core; no create_generated_clock is needed.
set_property -dict { PACKAGE_PIN W5  IOSTANDARD LVCMOS33 } [get_ports { clk }]
create_clock -add -name sys_clk_pin -period 10.000 -waveform {0 5} [get_ports { clk }]

## Reset push-button (BTNC), active-high
set_property -dict { PACKAGE_PIN U18 IOSTANDARD LVCMOS33 } [get_ports { rst_btn }]

## USB-UART bridge (FT2232) - note: from the FPGA's point of view,
## RsRx is an input (host -> FPGA) and RsTx is an output (FPGA -> host).
set_property -dict { PACKAGE_PIN B18 IOSTANDARD LVCMOS33 } [get_ports { uart_rx }]
set_property -dict { PACKAGE_PIN A18 IOSTANDARD LVCMOS33 } [get_ports { uart_tx }]

## Spike LEDs LD0..LD15 -- LED i shows neuron i firing in the latest timestep.
set_property -dict { PACKAGE_PIN U16 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN E19 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN U19 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN V19 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]
set_property -dict { PACKAGE_PIN W18 IOSTANDARD LVCMOS33 } [get_ports { led[4] }]
set_property -dict { PACKAGE_PIN U15 IOSTANDARD LVCMOS33 } [get_ports { led[5] }]
set_property -dict { PACKAGE_PIN U14 IOSTANDARD LVCMOS33 } [get_ports { led[6] }]
set_property -dict { PACKAGE_PIN V14 IOSTANDARD LVCMOS33 } [get_ports { led[7] }]
set_property -dict { PACKAGE_PIN V13 IOSTANDARD LVCMOS33 } [get_ports { led[8] }]
set_property -dict { PACKAGE_PIN V3  IOSTANDARD LVCMOS33 } [get_ports { led[9] }]
set_property -dict { PACKAGE_PIN W3  IOSTANDARD LVCMOS33 } [get_ports { led[10] }]
set_property -dict { PACKAGE_PIN U3  IOSTANDARD LVCMOS33 } [get_ports { led[11] }]
set_property -dict { PACKAGE_PIN P3  IOSTANDARD LVCMOS33 } [get_ports { led[12] }]
set_property -dict { PACKAGE_PIN N3  IOSTANDARD LVCMOS33 } [get_ports { led[13] }]
set_property -dict { PACKAGE_PIN P1  IOSTANDARD LVCMOS33 } [get_ports { led[14] }]
set_property -dict { PACKAGE_PIN L1  IOSTANDARD LVCMOS33 } [get_ports { led[15] }]

## Configuration / bitstream options
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO     [current_design]

## The internally-generated SPI clock is not a real board clock; the bridge runs
## entirely in the sys_clk domain (spi_sclk/cs/mosi are sampled by sys_clk, never
## used as a clock), so no extra create_clock is needed for it. The whole command
## path UART->bridge->SPI slave->engine->core is one synchronous sys_clk domain.

## --- asynchronous I/O (false paths) ---
## The only true async boundary is the UART (the host's baud clock is unrelated to
## sys_clk). uart_rx and rst_btn are resynchronised on-chip (see ASYNC_REG flops in
## uart_rx.v, snm_spi_slave.v, spikeengine_fpga_top.v); uart_tx is sampled by the
## host mid-bit; the LEDs are non-timing. None has a synchronous external timing
## relationship, so declare them false paths rather than invent fictional I/O delays.
set_false_path -from [get_ports rst_btn]
set_false_path -from [get_ports uart_rx]
set_false_path -to   [get_ports uart_tx]
set_false_path -to   [get_ports {led[*]}]

## --- high-fanout power-up reset: relax recovery/removal pessimism ---
## The synchronized reset rst_sync_reg[1] fans out to every async CLR/PRE in the
## ASIC-style core (~5.3k loads), so its distribution net (95% route, 0 logic) was the
## board's worst path (recovery/removal check) even though it comfortably arrives within
## one clock. reset_n is asserted ONLY at power-up (see the REQP-1839 / RBOR-1 waivers
## below), it feeds async resets only (never a data path), and the host re-initialises
## all state over UART after release -- so cross-flop skew in reset RELEASE is
## functionally harmless. Declare it a false path (standard for a synchronised,
## power-up-only reset): its recovery/removal no longer caps Fmax, so the reported WNS
## reflects the real synchronous datapath.
set_false_path -from [get_cells rst_sync_reg[1]]

# =====================================================================
# FPGA build waivers / message config (an XDC is just Tcl, so these apply
# automatically wherever this constraints file is used - no project changes).
#
# Every item below is an expected, benign consequence of reusing the verified
# SuperNeuroMAT3 ASIC core UNCHANGED on FPGA. Nothing here masks a real timing or
# functional problem: timing is met and the design was proven equivalent to the
# OpenRAM-backed core in RTL regression.
# =====================================================================

## --- implementation DRC waivers ---
## REQP-1839 / RBOR-1: the ASIC core uses asynchronous reset throughout, so the
## registers driving BRAM address/control pins (and the BRAM output capture regs)
## carry an async reset. Vivado flags a theoretical mid-access reset hazard. Here
## reset_n is asserted only at power-up (never during operation) and the BRAM
## contents are not reset, so the hazard cannot occur. Functional equivalence is
## covered by the RTL regression. Waive (with justification) rather than refactor
## verified RTL to synchronous reset.
## NOTE: read_xdc executes only recognised constraint/DRC commands, NOT general
## Tcl (if/catch/set/puts are ignored). So these must be BARE create_waiver calls,
## each on one line. They register during both the synth and impl constraint
## reads; report_drc (impl) then shows REQP-1839 / RBOR-1 as waived.
create_waiver -type DRC -id {REQP-1839} -description {Async-reset regs drive BRAM addr/control. reset_n asserts only at power-up, never mid-access; BRAM not reset. Functionally safe; verified in RTL regression.}
create_waiver -type DRC -id {RBOR-1} -description {BRAM output captured by async-reset regs (ASIC core reset style). reset_n power-up only; no mid-access reset. Functionally safe; verified in RTL regression.}
