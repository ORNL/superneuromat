`timescale 1us/1ns

// Command-bus-driven STDP-capable engine (2026-07-21, npu_stdp_dev
// experiment): snm_infer_cmd_ctrl_stdp + snm_infer_multilane_stdp wired
// together, mirroring snm_infer_engine.v's role for the inference-only
// engine exactly (drop-in board-top boundary exposing only the command
// bus + busy/irq). vmem_out_flat is deliberately left unconnected, same
// reasoning as snm_infer_engine.v's own comment on this (wide unpacked-
// array internal wires at N_MAX=1024 have tripped real Vivado opt_design
// crashes before; nothing on the command-bus path reads it).
module snm_infer_engine_stdp #(
    parameter integer N_MAX     = 1024,
    parameter integer NUM_LANES = 16,
    parameter integer DATA_W    = 16,
    parameter integer WEIGHT_W  = 8,
    parameter integer ACCUM_W   = 32,
    parameter integer REF_W     = 8,
    parameter integer STDP_WINDOW = 5,
    parameter integer SPIKE_MON_BASE = 0,  // LED[0] = this neuron id (see cmd_ctrl)
    // SYN_CAP_PER_LANE decoupled from N_MAX (2026-07-29). Default expression is
    // IDENTICAL to the previous hardcoded localparam, so any caller that does
    // NOT override it synthesizes to the exact same design as before (proven by
    // sim parity + real dense-rebuild regression). Override with a smaller value
    // for SPARSE graphs to fit many more neurons in the same BRAM/URAM budget
    // than dense N^2 worst-case sizing allows. Host-side capacity validation
    // (spikeengine.capacity.validate_network_fits) MUST guard loads against it
    // -- the RTL has no write-side bounds check.
    parameter integer SYN_CAP_PER_LANE = ((N_MAX + NUM_LANES - 1) / NUM_LANES) * N_MAX
)(
    input  wire        clk,
    input  wire        reset_n,
    input  wire        cmd_valid,
    output wire        cmd_ready,
    input  wire [7:0]  cmd_opcode,
    input  wire [7:0]  cmd_sel,
    input  wire [15:0] cmd_addr,
    input  wire [31:0] cmd_wdata,
    output wire        rsp_valid,
    input  wire        rsp_ready,
    output wire [7:0]  rsp_status,
    output wire [31:0] rsp_rdata,
    output wire        busy,
    output wire        irq,
    output wire [15:0] spike_mon
);
    localparam integer N_MAX_W = $clog2(N_MAX);
    localparam integer LOCAL_D = (N_MAX + NUM_LANES - 1) / NUM_LANES;
    localparam integer DPTR_W  = $clog2(SYN_CAP_PER_LANE + 1);
    localparam integer LANE_W  = (NUM_LANES < 2) ? 1 : $clog2(NUM_LANES);
    localparam integer LDPTR_W = $clog2(LOCAL_D + 1);
    localparam integer STDP_IDX_W = $clog2(STDP_WINDOW) + 1;

    wire [LANE_W-1:0]  cfg_lane;
    wire               cfg_dptr_we;
    wire [LDPTR_W-1:0] cfg_dptr_idx;
    wire [DPTR_W-1:0]  cfg_dptr_wdata;
    wire               cfg_syn_we;
    wire [DPTR_W-1:0]  cfg_syn_idx;
    wire [N_MAX_W-1:0] cfg_syn_src;
    wire signed [WEIGHT_W-1:0] cfg_syn_weight;
    wire               cfg_syn_re;
    wire               cfg_syn_rd_valid;
    wire [N_MAX_W-1:0] cfg_syn_rd_src;
    wire signed [WEIGHT_W-1:0] cfg_syn_rd_weight;
    wire               cfg_stdp_we;
    wire [STDP_IDX_W-1:0] cfg_stdp_idx;
    wire signed [WEIGHT_W-1:0] cfg_stdp_apos;
    wire signed [WEIGHT_W-1:0] cfg_stdp_aneg;
    wire               stdp_global_enable;
    wire               cfg_param_we;
    wire [2:0]         cfg_param_field;
    wire [LDPTR_W-1:0] cfg_param_idx;
    wire signed [DATA_W-1:0] cfg_param_threshold;
    wire [DATA_W-1:0]  cfg_param_leak;
    wire signed [DATA_W-1:0] cfg_param_reset_state;
    wire [REF_W-1:0]   cfg_param_refrac_period;
    wire               cfg_param_input_enable;
    wire               in_we;
    wire [LANE_W-1:0]  in_lane;
    wire [LDPTR_W-1:0] in_idx;
    wire signed [DATA_W-1:0] in_value;
    wire               input_valid;
    wire               spike_load;
    wire [N_MAX-1:0]   spike_in;
    wire               tick_start;
    wire               tick_busy, tick_done;
    wire [N_MAX-1:0]   spike_out;
    wire [31:0]        tick_cycles;
    wire [31:0]        stdp_tick_cycles;

    // ---- engine soft-reset (2026-07-25): cmd_ctrl pulses soft_reset on
    // OP_ENGINE_RESET; ANDing ~soft_reset into the multilane engine's reset_n
    // synchronously clears all engine datapath state without a bitstream
    // reload (cmd_ctrl keeps the ungated reset_n so it survives to answer the
    // command). See snm_infer_cmd_ctrl_stdp.v's soft_reset port comment. ----
    wire               soft_reset;
    wire               engine_reset_n = reset_n & ~soft_reset;

    snm_infer_cmd_ctrl_stdp #(
        .N_MAX(N_MAX), .NUM_LANES(NUM_LANES), .DATA_W(DATA_W),
        .WEIGHT_W(WEIGHT_W), .ACCUM_W(ACCUM_W), .REF_W(REF_W), .STDP_WINDOW(STDP_WINDOW),
        .SPIKE_MON_BASE(SPIKE_MON_BASE),
        // explicitly wired so cmd_ctrl's DPTR_W stays consistent with the lane
        // engine when SYN_CAP_PER_LANE is overridden (see param comment above).
        .SYN_CAP_PER_LANE(SYN_CAP_PER_LANE)
    ) u_ctrl (
        .clk(clk), .reset_n(reset_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_opcode(cmd_opcode),
        .cmd_sel(cmd_sel), .cmd_addr(cmd_addr), .cmd_wdata(cmd_wdata),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status),
        .rsp_rdata(rsp_rdata), .busy(busy), .irq(irq), .spike_mon(spike_mon),
        .soft_reset(soft_reset),
        .cfg_lane(cfg_lane), .cfg_dptr_we(cfg_dptr_we), .cfg_dptr_idx(cfg_dptr_idx),
        .cfg_dptr_wdata(cfg_dptr_wdata), .cfg_syn_we(cfg_syn_we), .cfg_syn_idx(cfg_syn_idx),
        .cfg_syn_src(cfg_syn_src), .cfg_syn_weight(cfg_syn_weight),
        .cfg_syn_re(cfg_syn_re), .cfg_syn_rd_valid(cfg_syn_rd_valid),
        .cfg_syn_rd_src(cfg_syn_rd_src), .cfg_syn_rd_weight(cfg_syn_rd_weight),
        .cfg_stdp_we(cfg_stdp_we), .cfg_stdp_idx(cfg_stdp_idx),
        .cfg_stdp_apos(cfg_stdp_apos), .cfg_stdp_aneg(cfg_stdp_aneg),
        .stdp_global_enable(stdp_global_enable),
        .cfg_param_we(cfg_param_we), .cfg_param_field(cfg_param_field), .cfg_param_idx(cfg_param_idx),
        .cfg_param_threshold(cfg_param_threshold), .cfg_param_leak(cfg_param_leak),
        .cfg_param_reset_state(cfg_param_reset_state), .cfg_param_refrac_period(cfg_param_refrac_period),
        .cfg_param_input_enable(cfg_param_input_enable),
        .in_we(in_we), .in_lane(in_lane), .in_idx(in_idx), .in_value(in_value),
        .input_valid(input_valid), .spike_load(spike_load), .spike_in(spike_in),
        .tick_start(tick_start), .tick_busy(tick_busy), .tick_done(tick_done),
        .tick_cycles(tick_cycles), .stdp_tick_cycles(stdp_tick_cycles),
        .spike_out(spike_out)
    );

    snm_infer_multilane_stdp #(
        .NUM_LANES(NUM_LANES), .N_MAX(N_MAX), .WEIGHT_W(WEIGHT_W),
        .ACCUM_W(ACCUM_W), .DATA_W(DATA_W), .REF_W(REF_W), .STDP_WINDOW(STDP_WINDOW),
        .SYN_CAP_PER_LANE(SYN_CAP_PER_LANE)
    ) u_engine (
        .clk(clk), .reset_n(engine_reset_n),
        .cfg_lane(cfg_lane),
        .cfg_dptr_we(cfg_dptr_we), .cfg_dptr_idx(cfg_dptr_idx), .cfg_dptr_wdata(cfg_dptr_wdata),
        .cfg_syn_we(cfg_syn_we), .cfg_syn_idx(cfg_syn_idx),
        .cfg_syn_src(cfg_syn_src), .cfg_syn_weight(cfg_syn_weight),
        .cfg_syn_re(cfg_syn_re), .cfg_syn_rd_valid(cfg_syn_rd_valid),
        .cfg_syn_rd_src(cfg_syn_rd_src), .cfg_syn_rd_weight(cfg_syn_rd_weight),
        .cfg_stdp_we(cfg_stdp_we), .cfg_stdp_idx(cfg_stdp_idx),
        .cfg_stdp_apos(cfg_stdp_apos), .cfg_stdp_aneg(cfg_stdp_aneg),
        .stdp_global_enable(stdp_global_enable),
        .cfg_param_we(cfg_param_we), .cfg_param_field(cfg_param_field), .cfg_param_idx(cfg_param_idx),
        .cfg_param_threshold(cfg_param_threshold), .cfg_param_leak(cfg_param_leak),
        .cfg_param_reset_state(cfg_param_reset_state), .cfg_param_refrac_period(cfg_param_refrac_period),
        .cfg_param_input_enable(cfg_param_input_enable),
        .in_we(in_we), .in_lane(in_lane), .in_idx(in_idx),
        .in_value(in_value), .input_valid(input_valid),
        .spike_load(spike_load), .spike_in(spike_in),
        .tick_start(tick_start), .tick_busy(tick_busy), .tick_done(tick_done),
        .tick_cycles(tick_cycles), .stdp_tick_cycles(stdp_tick_cycles),
        .spike_out(spike_out), .vmem_out_flat()
    );
endmodule
