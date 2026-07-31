
# Full zcu104 dataset build WITH Option A (neuron state -> BRAM). Reads the
# experiment RTL. Local build path + copy-back. Board recipe from the zcu104
# custom script (maxThreads 1, replication on cfg_* nets, URAM, route Explore).
# tclargs: <N_MAX> <NUM_LANES> <SYN_CAP_PER_LANE> <label>
set script_dir [file dirname [file normalize [info script]]]
# 2026-07-31: was `$script_dir/rtl`, i.e. scripts/rtl -- which does not exist.
# rtl/ is a SIBLING of scripts/, so every read_verilog below resolved to a
# missing path and this script could not run at all as shipped.
set pkg_root [file dirname $script_dir]
set src   $pkg_root/rtl
set part  xczu7ev-ffvc1156-2-e
set top   snm_infer_zcu104_top_stdp
set_param general.maxThreads 1

set N     [lindex $argv 0]
set K     [lindex $argv 1]
set CAP   [lindex $argv 2]
set label [lindex $argv 3]
set tag   "zcu104_${label}_N${N}_K${K}_bramA"
puts "DS_BUILD tag=$tag N=$N K=$K cap=$CAP"

# Local scratch build dir (kept off OneDrive -- Vivado's long-path/file-lock
# issues under a synced OneDrive folder are the reason for the copy-back
# pattern here). Override with SPIKEENGINE_BUILD_DIR; defaults to under TEMP,
# not a machine-specific path.
if {[info exists env(SPIKEENGINE_BUILD_DIR)]} {
    set build_root $env(SPIKEENGINE_BUILD_DIR)
} else {
    set build_root [file join $env(TEMP) spikeengine_build]
}
set outdir [file normalize $build_root/$tag]
file mkdir $outdir
# 2026-07-31: results are copied back next to the scratch build, NOT into
# $pkg_root/../vivado_build (which is inside site-packages once installed --
# same defect fixed in the other build scripts).
set final_outdir [file normalize $build_root/results/$tag]
file mkdir $final_outdir
set bit_name "snm_infer_${tag}.bit"

set rtl_files [list \
    snm_bram_fifo.v snm_gather_lane_stdp.v snm_neuron_update_lane.v \
    snm_infer_lane_stdp.v snm_infer_multilane_stdp.v snm_infer_cmd_ctrl_stdp.v \
    snm_infer_engine_stdp.v snm_spi_slave.v snm_infer_top_stdp.v \
    fpga/uart_rx.v fpga/uart_tx.v fpga/uart_to_spi_master.v \
    fpga/snm_infer_fpga_top_stdp.v fpga/snm_infer_zcu104_top_stdp.v]
foreach f $rtl_files { read_verilog -sv [list $src/$f] }
# 2026-07-31: was ../../../spikeengine_pkg/src/spikeengine/constraints/ -- the
# PRE-MERGE standalone layout, which now lives under legacy/ and is not
# shipped. Use the constraints inside this package.
read_xdc [list [file normalize $pkg_root/constraints/zcu104.xdc]]

synth_design -top $top -part $part -flatten_hierarchy rebuilt -include_dirs [list $src] \
    -generic N_MAX=$N -generic NUM_LANES=$K -generic STDP_WINDOW=5 -generic SPIKE_MON_BASE=64 \
    -generic WEIGHT_W=16 -generic DATA_W=24 -generic SYN_CAP_PER_LANE=$CAP \
    -verilog_define SYNTHESIS=1 -verilog_define SNM_INFER_NEURON_PIPE_5STAGE=1 \
    -verilog_define SNM_SYN_MEM_ULTRA=1 -verilog_define SNM_INFER_GATHER_PIPE_DEEP=1 \
    -verilog_define SNM_NEURON_STATE_BRAM=1
report_utilization -file $outdir/post_synth_util.rpt
set slut [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
puts "DS_SYNTH tag=$tag lutcells=$slut"

place_design -directive ExtraTimingOpt
set hfn [get_nets -hier -filter {NAME =~ *u_ctrl/cfg_param_* || NAME =~ *u_ctrl/cfg_syn_* || NAME =~ *u_ctrl/cfg_dptr_* || NAME =~ *u_ctrl/cfg_stdp_*}]
if {[llength $hfn]} { phys_opt_design -force_replication_on_nets $hfn }
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
phys_opt_design
report_utilization    -file $outdir/post_route_util.rpt
report_timing_summary -file $outdir/post_route_timing.rpt
report_drc            -file $outdir/post_route_drc.rpt
set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set luts [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set ur   [llength [get_cells -hier -filter {REF_NAME =~ URAM*}]]
set br   [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts "DS_ROUTE tag=$tag wns=$wns luts=$luts ffs=$ffs uram=$ur bram=$br"
write_bitstream -force $outdir/$bit_name
foreach f [glob -nocomplain $outdir/*] { file copy -force $f $final_outdir }
puts "DS_DONE tag=$tag bitstream=$final_outdir/$bit_name"
