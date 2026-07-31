
# WIDE FIXED-POINT (WEIGHT_W=16/DATA_W=24) max-capacity STDP build for SP701
# (2026-07-28). Same N_MAX/NUM_LANES/pipeline config as
# build_infer_sp701_stdp_maxcap352x8.tcl (that build's 8-bit synapse entries
# closed at only WNS +0.003ns with 106/120 BRAM tiles -- razor-thin even
# before widening). Widening ENTRY_W from WEIGHT_W(8)+SRC_W(9)=17 bits to
# WEIGHT_W(16)+SRC_W(9)=25 bits (~47% bigger) is a real risk to BOTH timing
# and BRAM budget here, unlike Basys3's much smaller N=256/K=8 config where
# the same widening had headroom to absorb. Building anyway (user's explicit
# call, 2026-07-28) to get real Vivado numbers rather than guess -- if BRAM
# overflows or timing goes negative, the reports below will show it plainly.
#
# MAX-CAPACITY STDP build for SP701 (2026-07-22, npu_stdp_dev experiment).
# N_MAX=352, NUM_LANES=8 -> LOCAL_D=44, SLIGHTLY SMALLER than the already
# timing-closed cap256x8 config's LOCAL_D=32 (WNS +0.169ns there) -- picked
# deliberately, since LOCAL_D (not raw N_MAX or NUM_LANES) is this project's
# own established dominant driver of critical-path depth (see NOTES.md's
# lane-count sweep at N=256 fixed: LOCAL_D is what matters, more lanes at
# fixed LOCAL_D was fine on the smaller Basys3 board too when LUT budget
# allowed it). SP701's xc7s100 has ~3x Basys3's LUT budget, so 16 lanes'
# replicated logic is expected to fit even though it didn't on Basys3.
#
# BRAM sizing: SYN_CAP_PER_LANE = LOCAL_D*N_MAX = 28*448 = 12,544 (dense
# worst case). ENTRY_W = WEIGHT_W(8) + SRC_W(ceil(log2(448))=9) = 17 bits.
# bits/lane = 12544*17 = 213,248, under 6 RAMB36 tiles (36864 bits each) ->
# 6*16 = 96 tiles, 80% of SP701's 120-tile budget (vs cap256x8's 34/120=28%)
# -- deliberately more aggressive since this build's whole point is finding
# the real capacity ceiling, not preserving Basys3-level margin.
#
#   vivado -mode batch -source scripts/build_infer_sp701_stdp_maxcap352x8.tcl

set script_dir [file dirname [file normalize [info script]]]
set src        [file dirname $script_dir]
set part       xc7s100fgga676-2
set top        snm_infer_sp701_top_stdp

# Output dir OUTSIDE OneDrive (2026-07-22): the OneDrive-hosted vivado_build/
# path intermittently fails at write_checkpoint with a spurious [Common
# 17-1293] "not writable" under cloud-sync file-lock load (same issue the
# ZCU104 STDP build already worked around). Build local, copy back at the end.
# Local scratch build dir, kept off OneDrive (see module comment above).
# Override with SPIKEENGINE_BUILD_DIR; defaults under TEMP, not a
# machine-specific path.
if {[info exists env(SPIKEENGINE_BUILD_DIR)]} {
    set build_root $env(SPIKEENGINE_BUILD_DIR)
} else {
    set build_root [file join $env(TEMP) spikeengine_build]
}
set local_outdir [file normalize $build_root/sp701_infer_stdp_maxcap352x8_wide16]
file mkdir $local_outdir
# 2026-07-31: results are copied back under the scratch build root, NOT
# into $src/../vivado_build -- that resolves inside site-packages once
# installed (read-only installs fail; Windows path length re-breaks).
set final_outdir [file normalize $build_root/results/sp701_infer_stdp_maxcap352x8_wide16]
file mkdir $final_outdir
set outdir $local_outdir

set bit_name "snm_infer_sp701_stdp_maxcap352x8_wide16.bit"

set rtl_files {
    rtl/snm_bram_fifo.v
    rtl/snm_gather_lane_stdp.v
    rtl/snm_neuron_update_lane.v
    rtl/snm_infer_lane_stdp.v
    rtl/snm_infer_multilane_stdp.v
    rtl/snm_infer_cmd_ctrl_stdp.v
    rtl/snm_infer_engine_stdp.v
    rtl/snm_spi_slave.v
    rtl/snm_infer_top_stdp.v
    rtl/fpga/uart_rx.v
    rtl/fpga/uart_tx.v
    rtl/fpga/uart_to_spi_master.v
    rtl/fpga/snm_infer_fpga_top_stdp.v
    rtl/fpga/snm_infer_sp701_top_stdp.v
}

foreach f $rtl_files {
    puts "read_verilog $f"
    read_verilog -sv [list $src/$f]
}
read_xdc [list $src/constraints/sp701.xdc]

source $script_dir/fpga_waivers.tcl

# 2026-07-24: this build previously closed with real margin (WNS +0.120ns,
# 106/120 BRAM) without the deep-pipe define. Rebuilding against the current
# session's RTL (SUMW_W widened, the STDP II=1 pipeline's extra registers,
# etc.) eroded that to WNS +0.003ns -- same critical path signature as the
# Basys3 STDP build hit (syn_row_reg -> accum_reg, spike_frozen mux,
# MUXF7x2/MUXF8x2), just not yet negative here. Adding
# SNM_INFER_GATHER_PIPE_DEEP now, proactively, using the same already-proven
# fix (490x WNS improvement on Basys3) rather than waiting for a future RTL
# change to push this the last few ps negative.
synth_design -top $top -part $part -flatten_hierarchy rebuilt \
    -include_dirs [list $src/rtl] \
    -generic N_MAX=352 -generic NUM_LANES=8 -generic STDP_WINDOW=5 -generic SPIKE_MON_BASE=64 \
    -generic WEIGHT_W=16 -generic DATA_W=24 \
    -verilog_define SYNTHESIS=1 -verilog_define SNM_INFER_NEURON_PIPE_5STAGE=1 \
    -verilog_define SNM_INFER_GATHER_PIPE_DEEP=1

source $script_dir/fpga_waivers.tcl

write_checkpoint  -force $outdir/post_synth.dcp
report_utilization -file $outdir/post_synth_util.rpt
report_timing_summary -file $outdir/post_synth_timing.rpt

# NOTE (2026-07-22): standalone `opt_design` deliberately OMITTED -- see the
# vivado-2025-2-optdesign-crash-workaround memory note / every Basys3 STDP
# build script this session. place_design's own internal optimization is a
# proven-safe substitute.
place_design -directive ExtraTimingOpt

# Same broadcast-fanout mitigation the inference-only SP701 build needed at
# NUM_LANES=8 (build_infer_sp701.tcl), extended with the STDP config-bus
# nets (cfg_stdp_*) -- applied proactively since the underlying cause (one
# physical config controller broadcasting to all lanes) is identical.
set hfn [get_nets -hier -filter \
    {NAME =~ *u_ctrl/cfg_param_* || NAME =~ *u_ctrl/cfg_syn_* || NAME =~ *u_ctrl/cfg_dptr_* || NAME =~ *u_ctrl/cfg_stdp_*}]
if {[llength $hfn]} {
    puts "SP701_STDP: forcing replication on [llength $hfn] high-fanout config-bus net(s)"
    phys_opt_design -force_replication_on_nets $hfn
}

phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design
write_checkpoint  -force $outdir/post_route.dcp
report_utilization        -file $outdir/post_route_util.rpt
report_timing_summary     -file $outdir/post_route_timing.rpt
report_drc                -file $outdir/post_route_drc.rpt
report_timing -delay_type max -max_paths 20 -sort_by slack -nworst 1 \
    -input_pins -file $outdir/post_route_worst_paths.rpt
report_power       -file $outdir/post_route_power.rpt
report_methodology -file $outdir/post_route_methodology.rpt

set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set luts  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set brams [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts "TIMING board=sp701_infer_stdp_maxcap352x8_wide16 part=$part wns=$wns ns luts=$luts ffs=$ffs brams=$brams"

write_bitstream -force $outdir/$bit_name

# Copy local build results back under the OneDrive-tracked vivado_build/ tree.
foreach f [glob -nocomplain $outdir/*] {
    file copy -force $f $final_outdir
}

puts "BUILD_DONE board=sp701_infer_stdp_maxcap352x8_wide16 part=$part bitstream=$final_outdir/$bit_name outdir=$final_outdir"
