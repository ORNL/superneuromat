`timescale 1us/1ns

// STDP-capable single-lane pipeline (2026-07-21, npu_stdp_dev experiment).
// Mirrors snm_infer_lane.v's structure exactly (gather -> neuron_update chain,
// tick_start/tick_busy/tick_done sequencing) but chains snm_gather_lane_stdp
// instead of snm_gather_lane, and adds a THIRD phase: after tick_done (both
// gather and neuron-update have produced this tick's spike_out), the caller
// pulses stdp_start to walk STDP write-backs for whichever local destinations
// just fired. Kept as its own file (not a modification of snm_infer_lane.v)
// for the same reason snm_gather_lane_stdp.v is its own file: the inference-
// only module stays byte-for-byte untouched.
module snm_infer_lane_stdp #(
    parameter integer LOCAL_D     = 32,
    parameter integer SYN_CAP     = 4096,
    parameter integer SRC_W       = 16,
    parameter integer WEIGHT_W    = 8,
    parameter integer ACCUM_W     = 32,
    parameter integer DATA_W      = 16,
    parameter integer REF_W       = 8,
    parameter integer STDP_WINDOW = 5,
    parameter integer DPTR_W      = $clog2(SYN_CAP + 1)
)(
    input  wire clk,
    input  wire reset_n,

    // ---- synapse config load (passthrough to snm_gather_lane_stdp) ----
    input  wire                     cfg_dptr_we,
    input  wire [$clog2(LOCAL_D+1)-1:0] cfg_dptr_idx,
    input  wire [DPTR_W-1:0]        cfg_dptr_wdata,
    input  wire                     cfg_syn_we,
    input  wire [DPTR_W-1:0]        cfg_syn_idx,
    input  wire [SRC_W-1:0]         cfg_syn_src,
    input  wire signed [WEIGHT_W-1:0] cfg_syn_weight,

    // ---- synapse weight readback (passthrough) ----
    input  wire                     cfg_syn_re,
    output wire                     cfg_syn_rd_valid,
    output wire [SRC_W-1:0]         cfg_syn_rd_src,
    output wire signed [WEIGHT_W-1:0] cfg_syn_rd_weight,

    // ---- STDP parameter config load (passthrough) ----
    input  wire                       cfg_stdp_we,
    input  wire [$clog2(STDP_WINDOW):0] cfg_stdp_idx,
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_apos,
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_aneg,
    input  wire                       stdp_global_enable,
    input  wire [7:0] stdp_win_eff,

    // ---- neuron param config load (passthrough) ----
    input  wire                          cfg_param_we,
    input  wire [2:0]                    cfg_param_field,
    input  wire [$clog2(LOCAL_D+1)-1:0]  cfg_param_idx,
    input  wire signed [DATA_W-1:0]      cfg_param_threshold,
    input  wire [DATA_W-1:0]             cfg_param_leak,
    input  wire signed [DATA_W-1:0]      cfg_param_reset_state,
    input  wire [REF_W-1:0]              cfg_param_refrac_period,
    input  wire                          cfg_param_input_enable,

    // ---- per-tick external-input load (passthrough) ----
    input  wire                          in_we,
    input  wire [$clog2(LOCAL_D+1)-1:0]  in_idx,
    input  wire signed [DATA_W-1:0]      in_value,
    input  wire                          input_valid,

    // ---- per-tick run (gather+neuron_update, identical timing to snm_infer_lane) ----
    input  wire                     tick_start,
    output wire                     tick_busy,
    output reg                      tick_done,
    input  wire [(1<<SRC_W)-1:0]    spike_frozen,

    // ---- results ----
    output wire [LOCAL_D-1:0]                spike_out,
    output wire signed [LOCAL_D*DATA_W-1:0]  vmem_out_flat,
    output reg  [31:0]              cycles_this_run,

    // ---- STDP phase (new): caller pulses stdp_start AFTER tick_done, once this
    // tick's spike_out is valid. ----
    input  wire                     stdp_start,
    output wire                     stdp_busy,
    output reg                      stdp_done,

    // ---- history refresh passthrough (2026-07-24: source-indexed BRAM,
    // replaces src_hist_flat -- see snm_gather_lane_stdp.v's own comment). ----
    input  wire [(1<<SRC_W)-1:0]    spike_out_global,
    input  wire                     hist_refresh_start,
    output wire                     hist_refresh_done,
    output reg [31:0]               stdp_cycles_this_run
);
    wire gather_busy, gather_done;
    wire signed [LOCAL_D*ACCUM_W-1:0] accum_flat;
    wire [31:0] gather_cycles;
    wire stdp_busy_w, stdp_done_w;
    wire [31:0] stdp_cycles_w;

    snm_gather_lane_stdp #(
        .LOCAL_D(LOCAL_D), .SYN_CAP(SYN_CAP), .SRC_W(SRC_W),
        .WEIGHT_W(WEIGHT_W), .ACCUM_W(ACCUM_W), .STDP_WINDOW(STDP_WINDOW)
    ) u_gather (
        .clk(clk), .reset_n(reset_n),
        .cfg_dptr_we(cfg_dptr_we), .cfg_dptr_idx(cfg_dptr_idx), .cfg_dptr_wdata(cfg_dptr_wdata),
        .cfg_syn_we(cfg_syn_we), .cfg_syn_idx(cfg_syn_idx),
        .cfg_syn_src(cfg_syn_src), .cfg_syn_weight(cfg_syn_weight),
        .cfg_syn_re(cfg_syn_re), .cfg_syn_rd_valid(cfg_syn_rd_valid),
        .cfg_syn_rd_src(cfg_syn_rd_src), .cfg_syn_rd_weight(cfg_syn_rd_weight),
        .cfg_stdp_we(cfg_stdp_we), .cfg_stdp_idx(cfg_stdp_idx),
        .cfg_stdp_apos(cfg_stdp_apos), .cfg_stdp_aneg(cfg_stdp_aneg),
        .stdp_global_enable(stdp_global_enable),
        .stdp_win_eff(stdp_win_eff),
        .run_start(tick_start),
        .run_busy(gather_busy),
        .run_done(gather_done),
        .spike_frozen(spike_frozen),
        .accum_out_flat(accum_flat),
        .cycles_this_run(gather_cycles),
        .stdp_start(stdp_start),
        .stdp_busy(stdp_busy_w),
        .stdp_done(stdp_done_w),
        .spike_out_local(spike_out),
        .spike_out_global(spike_out_global),
        .hist_refresh_start(hist_refresh_start),
        .hist_refresh_done(hist_refresh_done),
        .stdp_cycles_this_run(stdp_cycles_w)
    );
    assign stdp_busy = stdp_busy_w;

    wire neuron_busy, neuron_done;

    snm_neuron_update_lane #(
        .LOCAL_D(LOCAL_D), .DATA_W(DATA_W), .ACCUM_W(ACCUM_W), .REF_W(REF_W)
    ) u_neuron (
        .clk(clk), .reset_n(reset_n),
        .cfg_param_we(cfg_param_we), .cfg_param_field(cfg_param_field), .cfg_param_idx(cfg_param_idx),
        .cfg_param_threshold(cfg_param_threshold), .cfg_param_leak(cfg_param_leak),
        .cfg_param_reset_state(cfg_param_reset_state), .cfg_param_refrac_period(cfg_param_refrac_period),
        .cfg_param_input_enable(cfg_param_input_enable),
        .in_we(in_we), .in_idx(in_idx), .in_value(in_value), .input_valid(input_valid),
        .start(gather_done),
        .busy(neuron_busy),
        .done(neuron_done),
        .accum_in_flat(accum_flat),
        .spike_out(spike_out),
        .vmem_out_flat(vmem_out_flat)
    );

    // tick_busy: identical "latched, spans the whole tick" fix snm_infer_lane.v
    // documents (gather->neuron handoff glitch).
    reg busy_r;
    assign tick_busy = busy_r;

    // Synchronous reset (2026-07-24, future-ASIC methodology fix -- see
    // snm_gather_lane_stdp.v's always-block comment for the full rationale).
    always @(posedge clk) begin
        if (!reset_n) begin
            tick_done <= 1'b0;
            stdp_done <= 1'b0;
            busy_r <= 1'b0;
            cycles_this_run <= 32'd0;
            stdp_cycles_this_run <= 32'd0;
        end else begin
            tick_done <= 1'b0;
            stdp_done <= 1'b0;
            if (tick_start) begin
                busy_r <= 1'b1;
                cycles_this_run <= 32'd0;
            end else if (busy_r) begin
                cycles_this_run <= cycles_this_run + 1'b1;
            end
            if (neuron_done) begin
                busy_r <= 1'b0;
                tick_done <= 1'b1;
            end
            if (stdp_start) stdp_cycles_this_run <= 32'd0;
            else if (stdp_busy_w) stdp_cycles_this_run <= stdp_cycles_this_run + 1'b1;
            if (stdp_done_w) stdp_done <= 1'b1;
        end
    end
endmodule
