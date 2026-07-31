
# PARAMETRIC SP701 STDP build (2026-07-29). Wide fixed-point (W16/D24), reads
# N_MAX / NUM_LANES / SYN_CAP_PER_LANE / SPIKE_MON_BASE / SYNTH_ONLY from
# -tclargs. Local build path + copy-back (OneDrive lock workaround). Same
# board-specific fix recipe as build_infer_sp701_stdp_maxcap352x8.tcl
# (replication on cfg_* nets, 5-stage neuron pipe, deep gather pipe). NOTE:
# the sp701 top was fixed (2026-07-29) to actually declare+forward WEIGHT_W/
# DATA_W/SPIKE_MON_BASE/SYN_CAP_PER_LANE -- previously these -generics were
# silently dropped and the "wide16" bitstream was really an 8-bit build.
#
#   vivado -mode batch -source build_infer_sp701_stdp_custom.tcl \
#          -tclargs <N_MAX> <NUM_LANES> <SYN_CAP_PER_LANE> [SPIKE_MON_BASE] [SYNTH_ONLY]

set script_dir [file dirname [file normalize [info script]]]
set src        [file dirname $script_dir]
set part       xc7s100fgga676-2
set top        snm_infer_sp701_top_stdp

set N_MAX     [expr {[llength $argv] >= 1 ? [lindex $argv 0] : 256}]
set NUM_LANES [expr {[llength $argv] >= 2 ? [lindex $argv 1] : 8}]
set LOCAL_D   [expr {($N_MAX + $NUM_LANES - 1) / $NUM_LANES}]
set SYN_CAP   [expr {[llength $argv] >= 3 ? [lindex $argv 2] : $LOCAL_D * $N_MAX}]
set SPIKE_MON [expr {[llength $argv] >= 4 ? [lindex $argv 3] : 64}]
set SYNTH_ONLY [expr {[llength $argv] >= 5 ? [lindex $argv 4] : 0}]

set tag "N${N_MAX}_K${NUM_LANES}_cap${SYN_CAP}_w16"
puts "CUSTOM_BUILD board=sp701 tag=$tag N_MAX=$N_MAX NUM_LANES=$NUM_LANES SYN_CAP_PER_LANE=$SYN_CAP LOCAL_D=$LOCAL_D synth_only=$SYNTH_ONLY"

# Local scratch build dir, kept off OneDrive (see module comment above).
# Override with SPIKEENGINE_BUILD_DIR; defaults under TEMP, not a
# machine-specific path.
if {[info exists env(SPIKEENGINE_BUILD_DIR)]} {
    set build_root $env(SPIKEENGINE_BUILD_DIR)
} else {
    set build_root [file join $env(TEMP) spikeengine_build]
}
set local_outdir [file normalize $build_root/sp701_custom_$tag]
file mkdir $local_outdir
set final_outdir [file normalize $build_root/results/sp701_custom_$tag]
file mkdir $final_outdir
set outdir $local_outdir
set bit_name "snm_infer_sp701_stdp_${tag}.bit"

set rtl_files {
    rtl/snm_bram_fifo.v rtl/snm_gather_lane_stdp.v rtl/snm_neuron_update_lane.v
    rtl/snm_infer_lane_stdp.v rtl/snm_infer_multilane_stdp.v rtl/snm_infer_cmd_ctrl_stdp.v
    rtl/snm_infer_engine_stdp.v rtl/snm_spi_slave.v rtl/snm_infer_top_stdp.v
    rtl/fpga/uart_rx.v rtl/fpga/uart_tx.v rtl/fpga/uart_to_spi_master.v
    rtl/fpga/snm_infer_fpga_top_stdp.v rtl/fpga/snm_infer_sp701_top_stdp.v
}
foreach f $rtl_files { read_verilog -sv [list $src/$f] }
read_xdc [list $src/constraints/sp701.xdc]
source $script_dir/fpga_waivers.tcl

synth_design -top $top -part $part -flatten_hierarchy rebuilt \
    -include_dirs [list $src/rtl] \
    -generic N_MAX=$N_MAX -generic NUM_LANES=$NUM_LANES -generic STDP_WINDOW=5 \
    -generic SPIKE_MON_BASE=$SPIKE_MON -generic WEIGHT_W=16 -generic DATA_W=24 \
    -generic SYN_CAP_PER_LANE=$SYN_CAP \
    -verilog_define SYNTHESIS=1 -verilog_define SNM_INFER_NEURON_PIPE_5STAGE=1 \
    -verilog_define SNM_INFER_GATHER_PIPE_DEEP=1
source $script_dir/fpga_waivers.tcl

report_utilization -file $outdir/post_synth_util.rpt
if {$SYNTH_ONLY} {
    set brams [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
    set luts  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
    set ffs   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
    puts "SYNTH_CHECK board=sp701 N=$N_MAX K=$NUM_LANES CAP=$SYN_CAP => brams=$brams lutcells=$luts ffs=$ffs"
    return
}
write_checkpoint  -force $outdir/post_synth.dcp
report_timing_summary -file $outdir/post_synth_timing.rpt

place_design -directive ExtraTimingOpt
set hfn [get_nets -hier -filter \
    {NAME =~ *u_ctrl/cfg_param_* || NAME =~ *u_ctrl/cfg_syn_* || NAME =~ *u_ctrl/cfg_dptr_* || NAME =~ *u_ctrl/cfg_stdp_*}]
if {[llength $hfn]} { phys_opt_design -force_replication_on_nets $hfn }
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design
write_checkpoint  -force $outdir/post_route.dcp
report_utilization    -file $outdir/post_route_util.rpt
report_timing_summary -file $outdir/post_route_timing.rpt
report_drc            -file $outdir/post_route_drc.rpt

set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set luts  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set brams [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts "TIMING board=sp701_$tag part=$part wns=$wns ns luts=$luts ffs=$ffs brams=$brams"
write_bitstream -force $outdir/$bit_name
foreach f [glob -nocomplain $outdir/*] { file copy -force $f $final_outdir }
puts "BUILD_DONE board=sp701_$tag part=$part bitstream=$final_outdir/$bit_name outdir=$final_outdir"
