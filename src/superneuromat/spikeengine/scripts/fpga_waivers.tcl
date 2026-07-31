# FPGA build waivers / message config for SuperNeuroMAT3 FPGA board targets.
#
# Every item below is an expected, benign consequence of reusing the verified
# SuperNeuroMAT3 ASIC core UNCHANGED on an FPGA. Each is documented so the build
# log stays clean without editing verified RTL. Nothing here masks a real timing
# or functional problem: timing is met and the design was proven equivalent to
# the OpenRAM-backed core in RTL regression.
#
# Safe to source at any stage: the synth message suppressions always apply; the
# DRC waivers only run once a design exists (post-synth). This makes the file
# usable as BOTH a synthesis and an implementation pre-hook.
#
# Batch flow: build_fpga.tcl sources it before synth and after synth.
# GUI flow:
#   Settings -> Synthesis      -> tcl.pre = scripts/fpga_waivers.tcl
#   Settings -> Implementation -> tcl.pre = scripts/fpga_waivers.tcl

# ---- synthesis message suppressions (apply always) ----
# (The DFT scan ports were removed from the RTL, so the scan-related 8-3848 /
#  8-7129 warnings no longer occur.)
# 8-6014: dead/combinational temp registers optimised away for this config.
# 8-7129: a few remaining unused ports (e.g. the BRAM wrapper's reset_n, kept for
#         drop-in compatibility with the OpenRAM wrapper port list).
set_msg_config -id {Synth 8-6014} -suppress
set_msg_config -id {Synth 8-7129} -suppress

# ---- implementation DRC waivers (only once a design is loaded) ----
# REQP-1839 / RBOR-1: the ASIC core uses asynchronous reset throughout, so the
# registers driving BRAM address/control pins (and the BRAM output capture regs)
# carry an async reset. Vivado flags a theoretical mid-access reset hazard. In
# this design reset_n is asserted only at power-up (never during operation) and
# the BRAM contents are not reset, so the hazard cannot occur. Functional
# equivalence is covered by the RTL regression. Waive with justification rather
# than refactor verified RTL to synchronous reset.
if {[current_design -quiet] ne ""} {
    if {[llength [get_waivers -quiet -filter {ID == REQP-1839}]] == 0} {
        create_waiver -type DRC -id {REQP-1839} \
            -description {Async-reset regs drive BRAM addr/control. reset_n asserts only at power-up, never mid-access; BRAM not reset. Functionally safe; verified in RTL regression.}
    }
    if {[llength [get_waivers -quiet -filter {ID == RBOR-1}]] == 0} {
        create_waiver -type DRC -id {RBOR-1} \
            -description {BRAM output captured by async-reset regs (ASIC core reset style). reset_n power-up only; no mid-access reset. Functionally safe; verified in RTL regression.}
    }
}
