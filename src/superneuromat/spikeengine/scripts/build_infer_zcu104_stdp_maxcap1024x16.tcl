
# WIDE FIXED-POINT (WEIGHT_W=16/DATA_W=24) max-capacity STDP build for ZCU104
# (2026-07-28). Same N_MAX/NUM_LANES/URAM-packing config as
# build_infer_zcu104_stdp_maxcap1024x16.tcl. REAL RISK: that build's URAM
# packing (MEM_NATIVE_W=72, ENTRY_W=WEIGHT_W(8)+SRC_W(10)=18 bits ->
# PACK_FACTOR=4 entries/row -> 64/96 URAM tiles) was sized for 8-bit weights.
# Widening to WEIGHT_W(16)+SRC_W(10)=26 bits drops PACK_FACTOR to
# floor(72/26)=2 entries/row -- roughly DOUBLING rows needed per lane, which
# could push URAM usage well past the 96-tile budget. Building anyway (user's
# explicit call, 2026-07-28) to get real Vivado numbers rather than guess --
# report_utilization below will show the actual URAM count if it fits at all.
#
# MAX-CAPACITY STDP build for ZCU104 (2026-07-22, npu_stdp_dev experiment).
# N_MAX=1024, NUM_LANES=16 -- ZCU104's own NATIVE inference-only target scale
# (see snm_infer_zcu104_top.v's own sizing comment), now with STDP. LOCAL_D=64.
#
# SNM_SYN_MEM_ULTRA=1 (URAM) IS defined here, unlike the smaller cap256x8
# build: at this scale, plain block RAM alone doesn't fit -- SYN_CAP_PER_LANE
# = LOCAL_D*N_MAX = 64*1024 = 65,536 (dense worst case), ENTRY_W =
# WEIGHT_W(8)+SRC_W(ceil(log2(1024))=10) = 18 bits, bits/lane = 65536*18 =
# 1,179,648, which alone needs 32 RAMB36 tiles/lane * 16 lanes = 512 tiles --
# over ZCU104's 312-tile budget. Packing into URAM instead (MEM_NATIVE_W=72,
# PACK_FACTOR=72/18=4 entries/row) drops this to 16,384 rows/lane / 4096
# rows-per-URAM-tile = 4 URAM tiles/lane * 16 lanes = 64 tiles, 67% of
# ZCU104's 96-tile URAM budget -- feasible, unlike the plain-BRAM path.
#
#   vivado -mode batch -source scripts/build_infer_zcu104_stdp_maxcap1024x16.tcl

set script_dir [file dirname [file normalize [info script]]]
set src        [file dirname $script_dir]
set part       xczu7ev-ffvc1156-2-e
set top        snm_infer_zcu104_top_stdp

# Inherited caution from build_infer_zcu104.tcl: that build hit a reproducible
# empty [Synth 20-411] opt_design crash at the FULL N_MAX=1024/NUM_LANES=16
# scale when cmd_ctrl/engine was combined with snm_spi_slave, fixed by forcing
# single-threaded optimization. This STDP build is much smaller
# (N_MAX=256/NUM_LANES=8) and OMITS the standalone opt_design call entirely
# (see the Basys3 STDP scripts' note), so the original trigger may not even
# apply -- kept anyway as cheap insurance since it costs wall-clock time only,
# not correctness or margin.
set_param general.maxThreads 1

# Output dir deliberately OUTSIDE OneDrive (2026-07-22): two consecutive
# attempts at the OneDrive-hosted vivado_build/zcu104_infer_stdp_maxcap1024x16
# path failed at the very first write_checkpoint with a spurious [Common
# 17-1293] "already exists, is a directory, but is not writable" on a
# freshly-created, empty, independently-verified-writable folder -- matches
# this project's known OneDrive cloud-sync file-lock behavior (already
# documented for hw_server .bit programming; same root cause hits Vivado's
# own checkpoint writer here). Building to a local, non-synced path sidesteps
# it; reports/bitstream are copied back under vivado_build/ at the end.
# Local scratch build dir, kept off OneDrive (see module comment above).
# Override with SPIKEENGINE_BUILD_DIR; defaults under TEMP, not a
# machine-specific path.
if {[info exists env(SPIKEENGINE_BUILD_DIR)]} {
    set build_root $env(SPIKEENGINE_BUILD_DIR)
} else {
    set build_root [file join $env(TEMP) spikeengine_build]
}
set local_outdir [file normalize $build_root/zcu104_infer_stdp_maxcap1024x16_wide16]
file mkdir $local_outdir
# 2026-07-31: removed a dead `set outdir $src/../vivado_build/...` + mkdir that
# was immediately overwritten by $local_outdir on the next line. Harmless as
# written, but it CREATED that directory inside the package (site-packages
# once installed) and was a standing invitation to "fix" the override by
# deleting the wrong line. Results are copied back to $final_outdir below.
set outdir $local_outdir

set bit_name "snm_infer_zcu104_stdp_maxcap1024x16_wide16.bit"

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
    rtl/fpga/snm_infer_zcu104_top_stdp.v
}

foreach f $rtl_files {
    puts "read_verilog $f"
    read_verilog -sv [list $src/$f]
}
read_xdc [list $src/constraints/zcu104.xdc]

source $script_dir/fpga_waivers.tcl

# 2026-07-24: this build previously closed with huge margin (WNS +1.502ns,
# no deep-pipe define needed) at 67% URAM / 34% LUT -- plenty of fabric to
# spare, so this was never a congestion problem the way SP701's K=16 was.
# Rebuilding against the current session's RTL (SUMW_W widened, the STDP
# II=1 pipeline's extra registers, the port-B history fix, etc.) eroded that
# margin to WNS +0.053ns -- same critical path signature Basys3 and SP701
# both hit (syn_row_reg -> accum_reg, spike_frozen mux, MUXF7x2/MUXF8x2,
# real post-route data confirmed). Adding SNM_INFER_GATHER_PIPE_DEEP now,
# same already-proven fix (490x WNS improvement on Basys3, 148x on SP701).
synth_design -top $top -part $part -flatten_hierarchy rebuilt \
    -include_dirs [list $src/rtl] \
    -generic N_MAX=1024 -generic NUM_LANES=16 -generic STDP_WINDOW=5 -generic SPIKE_MON_BASE=64 \
    -generic WEIGHT_W=16 -generic DATA_W=24 \
    -verilog_define SYNTHESIS=1 -verilog_define SNM_INFER_NEURON_PIPE_5STAGE=1 \
    -verilog_define SNM_SYN_MEM_ULTRA=1 -verilog_define SNM_INFER_GATHER_PIPE_DEEP=1

source $script_dir/fpga_waivers.tcl

write_checkpoint  -force $outdir/post_synth.dcp
report_utilization -file $outdir/post_synth_util.rpt
report_timing_summary -file $outdir/post_synth_timing.rpt

# NOTE (2026-07-22): standalone `opt_design` deliberately OMITTED -- see the
# vivado-2025-2-optdesign-crash-workaround memory note / every Basys3 STDP
# build script this session.
place_design -directive ExtraTimingOpt

set hfn [get_nets -hier -filter \
    {NAME =~ *u_ctrl/cfg_param_* || NAME =~ *u_ctrl/cfg_syn_* || NAME =~ *u_ctrl/cfg_dptr_* || NAME =~ *u_ctrl/cfg_stdp_*}]
if {[llength $hfn]} {
    puts "ZCU104_STDP: forcing replication on [llength $hfn] high-fanout config-bus net(s)"
    phys_opt_design -force_replication_on_nets $hfn
}

phys_opt_design -directive AggressiveExplore
route_design -directive Explore
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
set urams [llength [get_cells -hier -filter {REF_NAME =~ URAM*}]]
puts "TIMING board=zcu104_infer_stdp_maxcap1024x16_wide16 part=$part wns=$wns ns luts=$luts ffs=$ffs brams=$brams urams=$urams"

write_bitstream -force $outdir/$bit_name

# Copy the local build results back under the OneDrive-tracked vivado_build/
# tree so they're versioned/discoverable like every other board's build.
# 2026-07-31: results are copied back under the scratch build root, NOT
# into $src/../vivado_build -- that resolves inside site-packages once
# installed (read-only installs fail; Windows path length re-breaks).
set final_outdir [file normalize $build_root/results/zcu104_infer_stdp_maxcap1024x16_wide16]
file mkdir $final_outdir
foreach f [glob -nocomplain $outdir/*] {
    file copy -force $f $final_outdir
}

puts "BUILD_DONE board=zcu104_infer_stdp_maxcap1024x16_wide16 part=$part bitstream=$final_outdir/$bit_name outdir=$final_outdir"
