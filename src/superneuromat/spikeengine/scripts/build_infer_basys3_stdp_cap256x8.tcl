
# Max-practical-capacity STDP build for Basys3 (2026-07-21, npu_stdp_dev
# experiment). N_MAX=128, NUM_LANES=8 -> LOCAL_D=16, dense worst-case
# SYN_CAP_PER_LANE=LOCAL_D*N_MAX=2048 (snm_infer_engine_stdp.v always sizes
# for dense worst case, same convention as the inference-only engine --
# there is no separate sparse-capacity override at the board-top level).
#
# Resource justification (done BEFORE this build, not after a failure):
# ENTRY_W = WEIGHT_W(8) + SRC_W(ceil(log2(128))=7) = 15 bits. bits/lane =
# SYN_CAP_PER_LANE * ENTRY_W = 2048*15 = 30720 bits, under one RAMB36E1's
# 36864-bit capacity -> 1 tile per read port; STDP duplicates the read port
# (see NOTES.md progress update 3/6) -> 2 tiles/lane -> 8 lanes * 2 = 16
# BRAM36 tiles total, well inside Basys3's 50-tile budget. LOCAL_D=16 is
# HALF the LOCAL_D=32 the inference-only engine needed pipeline-stage fixes
# to barely close timing at (WNS +0.599ns, N_MAX=256/NUM_LANES=8) -- picked
# deliberately to keep real margin against STDP's added per-lane logic
# depth (T_ISSUE/T_WAIT_READ/T_UPDATE, saturating add, merged read/write
# muxing) on top of an already-razor-thin historical baseline.
#
#   vivado -mode batch -source scripts/build_infer_basys3_stdp_maxcap.tcl

set script_dir [file dirname [file normalize [info script]]]
set src        [file dirname $script_dir]
set part       xc7a35tcpg236-1
set top        snm_infer_fpga_top_stdp

# Scratch build dir OUTSIDE the package (2026-07-31). This was the only build
# script still writing to $src/../vivado_build/, which resolves INSIDE
# site-packages once installed: checkpoints, the full post_route_*.rpt set and
# the .bit all landed in the installed package. That fails outright on a
# read-only install and re-opens the Windows path-length problem, and basys3
# is the DEFAULT board, so it was the path most users would hit. build.py
# redirects only Vivado's cwd/-tempDir -- the Tcl picks outdir itself -- so
# the fix has to be here. Matches every other build script.
if {[info exists env(SPIKEENGINE_BUILD_DIR)]} {
    set build_root $env(SPIKEENGINE_BUILD_DIR)
} else {
    set build_root [file join $env(TEMP) spikeengine_build]
}
set outdir     [file normalize $build_root/basys3_infer_stdp_cap256x8]
file mkdir $outdir

set bit_name "snm_infer_basys3_stdp_cap256x8.bit"

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
}

foreach f $rtl_files {
    puts "read_verilog $f"
    read_verilog -sv [list $src/$f]
}
read_xdc [list $src/constraints/basys3.xdc]

source $script_dir/fpga_waivers.tcl

# SPIKE_MON_BASE=64 (2026-07-25): the board's 16 LEDs show spike_out starting at
# neuron 64 -- i.e. the OUTPUT-class neurons of the 64-input/10-output digits
# classifier (LED[0..9] = classes 0..9), instead of the default input/pixel
# layer. Purely a monitor-mux choice; no effect on compute. Harmless for any
# other >=80-neuron network run on this bitstream (just a different LED window).
# WEIGHT_W=16 / DATA_W=24 (2026-07-27): wider fixed-point so on-chip STDP
# training can represent the working digits recipe's sub-LSB taps (0.002 etc.)
# at host FRAC_BITS=11 -- WEIGHT_W=16 holds the fine taps + weights up to ~11
# real (max seen ~11.3), and DATA_W=24 gives the membrane the headroom the
# recipe's dynamic range needs (threshold 99 / teacher 100 at FRAC=11 =
# ~203k raw). 8-bit/16-bit defaults stay for other builds; this build selects
# the wide datapath explicitly. Cost: synapse entry is WEIGHT_W+SRC_W = 24 bits
# (was 15), so ~1.6x synapse BRAM -- confirm it still fits xc7a35t below.
synth_design -top $top -part $part -flatten_hierarchy rebuilt \
    -include_dirs [list $src/rtl] \
    -generic N_MAX=256 -generic NUM_LANES=8 -generic STDP_WINDOW=5 -generic SPIKE_MON_BASE=64 \
    -generic WEIGHT_W=16 -generic DATA_W=24 \
    -verilog_define SYNTHESIS=1 -verilog_define SNM_INFER_NEURON_PIPE_5STAGE=1 \
    -verilog_define SNM_INFER_GATHER_PIPE_DEEP=1

source $script_dir/fpga_waivers.tcl

write_checkpoint  -force $outdir/post_synth.dcp
report_utilization -file $outdir/post_synth_util.rpt
report_timing_summary -file $outdir/post_synth_timing.rpt

# NOTE (2026-07-22): the standalone `opt_design` call is deliberately OMITTED.
# Vivado 2025.2 crashes reproducibly with [Synth 20-411] at this exact call on
# this design (confirmed 3x, always at the same point, always after synth/DRC
# completed cleanly -- see the vivado-2025-2-optdesign-crash-workaround memory
# note). place_design performs its own internal optimization pass, so skipping
# the standalone opt_design is a proven-safe workaround (real result: WNS
# +0.111-0.335ns across three separate builds with it omitted), not a quality
# shortcut.
place_design -directive ExtraTimingOpt

# Same broadcast-fanout mitigation build_infer_basys3.tcl already needed at
# this LOCAL_D scale -- apply proactively.
set hfn [get_nets -hier -filter \
    {NAME =~ *u_ctrl/cfg_param_* || NAME =~ *u_ctrl/cfg_syn_* || NAME =~ *u_ctrl/cfg_dptr_* || NAME =~ *u_ctrl/cfg_stdp_*}]
if {[llength $hfn]} {
    puts "BASYS3_STDP_MAXCAP: forcing replication on [llength $hfn] high-fanout config-bus net(s)"
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

set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set luts  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set brams [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts "TIMING board=basys3_infer_stdp_cap256x8 part=$part wns=$wns ns luts=$luts ffs=$ffs brams=$brams"

write_bitstream -force $outdir/$bit_name

puts "BUILD_DONE board=basys3_infer_stdp_cap256x8 part=$part bitstream=$outdir/$bit_name outdir=$outdir"
