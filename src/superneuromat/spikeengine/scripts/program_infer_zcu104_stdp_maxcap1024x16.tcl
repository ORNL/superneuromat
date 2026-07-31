# Vivado batch JTAG programmer for the parallel-lane INFERENCE engine on ZCU104
# (snm_infer_zcu104_top: N=1024, K=16 lanes, S=1,048,576 synapses, built_unverified
# as of 2026-07-12 -- this script performs its FIRST hardware bring-up).
#
# Downloads vivado_build/zcu104_infer/snm_infer_zcu104.bit to the Zynq UltraScale+
# PL over the board's onboard USB-JTAG. Volatile load only (lost on power cycle).
#
# Run (board connected via USB, powered on):
#   vivado -mode batch -source scripts/program_infer_zcu104.tcl
#
# Optionally point at a different bitstream:
#   vivado -mode batch -source scripts/program_infer_zcu104.tcl -tclargs path/to.bit

set script_dir  [file dirname [file normalize [info script]]]
set src         [file dirname $script_dir]
set default_bit [file normalize $src/../bitstreams/zcu104/snm_infer_zcu104_stdp_maxcap1024x16.bit]

set bit $default_bit
if {[llength $argv] >= 1} {
    set bit [file normalize [lindex $argv 0]]
}
if {![file exists $bit]} {
    puts "ERROR: bitstream not found: $bit"
    puts "       build it first: vivado -mode batch -source scripts/build_infer_zcu104.tcl"
    exit 1
}
puts "Programming bitstream: $bit"

# ---- connect to the hardware server / target ----
open_hw_manager
connect_hw_server -allow_non_jtag
if {[catch {get_hw_targets} targets] || [llength $targets] == 0} {
    puts "ERROR: no JTAG hardware targets found."
    puts "       Check the USB-JTAG cable (Digilent JTAG-SMT2-NC on ZCU104), board"
    puts "       power, and that no other tool (e.g. a Vivado GUI hw_manager) is"
    puts "       holding the target open."
    close_hw_manager
    exit 1
}
# ---- search EVERY JTAG target for the xczu7ev PL device, not just the first
# target (2026-07-14: with two boards connected on separate USB-JTAG
# interfaces, each shows up as its OWN hw_target -- picking only
# [lindex $targets 0] would silently program whichever board happened to
# enumerate first, which is exactly what caused a real ZCU104/SP701 bitstream
# mismatch error earlier this session). The JTAG chain on ZCU104 itself also
# typically enumerates the PS debug tap ("arm_dap...") ahead of the PL device
# within a single target -- filter for xczu7ev specifically, not "*dap*". ----
set dev ""
set matched_target ""
foreach t $targets {
    current_hw_target $t
    open_hw_target
    foreach d [get_hw_devices] {
        if {[string match -nocase "xczu7*" $d] && ![string match -nocase "*dap*" $d]} {
            set dev $d; set matched_target $t; break
        }
    }
    if {$dev ne ""} { break }
    close_hw_target
}
if {$dev eq ""} {
    puts "ERROR: no xczu7ev device found on any of [llength $targets] JTAG target(s)."
    puts "       Targets checked: $targets"
    puts "       Check the ZCU104 board is connected/powered."
    close_hw_manager
    exit 1
}
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
puts "Target device: $dev (on hw_target $matched_target)"

# ---- program ----
set_property PROGRAM.FILE [list $bit] $dev
program_hw_devices $dev
refresh_hw_device $dev

puts "PROGRAM_DONE device=$dev"
close_hw_target
close_hw_manager

# --------------------------------------------------------------------------
# Persistent (QSPI-flash) programming is NOT set up for this image -- this is
# a first-bring-up volatile load only, matching the project's policy of
# validating on volatile load before ever touching flash. See NOTES.md.
# --------------------------------------------------------------------------
