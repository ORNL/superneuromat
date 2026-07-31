# Vivado batch JTAG programmer for the STDP-capable parallel-lane engine on
# Basys3 (2026-07-21, npu_stdp_dev experiment -- FIRST hardware bring-up for
# this experiment). New sibling of program_infer_basys3.tcl (that file
# untouched), same volatile-load-only JTAG flow, pointed at the STDP
# bitstream (tiny N_MAX=4/NUM_LANES=2 network, chosen deliberately small for
# a first-ever hardware attempt).
#
# Run (board connected via USB, powered on):
#   vivado -mode batch -source scripts/program_infer_basys3_stdp.tcl

set script_dir  [file dirname [file normalize [info script]]]
set src         [file dirname $script_dir]
set default_bit [file normalize $src/../bitstreams/basys3/snm_infer_basys3_stdp_cap256x8.bit]

set bit $default_bit
if {[llength $argv] >= 1} {
    set bit [file normalize [lindex $argv 0]]
}
if {![file exists $bit]} {
    puts "ERROR: bitstream not found: $bit"
    puts "       build it first: vivado -mode batch -source scripts/build_infer_basys3_stdp.tcl"
    exit 1
}
puts "Programming bitstream: $bit"

open_hw_manager
connect_hw_server -allow_non_jtag
if {[catch {get_hw_targets} targets] || [llength $targets] == 0} {
    puts "ERROR: no JTAG hardware targets found."
    puts "       Check the USB cable, board power, and that no other tool"
    puts "       (e.g. a Vivado GUI hw_manager) is holding the target open."
    close_hw_manager
    exit 1
}

set dev ""
set matched_target ""
foreach t $targets {
    current_hw_target $t
    open_hw_target
    foreach d [get_hw_devices] {
        if {[string match -nocase "xc7a35t*" $d]} { set dev $d; set matched_target $t; break }
    }
    if {$dev ne ""} { break }
    close_hw_target
}
if {$dev eq ""} {
    puts "ERROR: no xc7a35t device found on any of [llength $targets] JTAG target(s)."
    puts "       Targets checked: $targets"
    puts "       Check the Basys3 board is connected/powered."
    close_hw_manager
    exit 1
}
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
puts "Target device: $dev (on hw_target $matched_target)"

set_property PROGRAM.FILE [list $bit] $dev
program_hw_devices $dev
refresh_hw_device $dev

puts "PROGRAM_DONE device=$dev"
close_hw_target
close_hw_manager

# --------------------------------------------------------------------------
# Volatile load only (lost on power cycle) -- matches project policy of
# validating on volatile load before ever touching flash, and this being a
# first bring-up for this specific STDP experiment.
# --------------------------------------------------------------------------
