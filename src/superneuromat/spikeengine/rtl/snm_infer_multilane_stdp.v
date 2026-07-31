`timescale 1us/1ns

// STDP-capable K-lane inference+learning engine (2026-07-21, npu_stdp_dev
// experiment). Mirrors snm_infer_multilane.v's structure (K lanes under a
// bulk-synchronous tick barrier, per-lane-replicated frozen-spike scheme,
// destination partitioning n -> lane n%NUM_LANES / local n/NUM_LANES) but
// chains snm_infer_lane_stdp instead of snm_infer_lane, and adds:
//   1. A STDP_WINDOW-deep, per-GLOBAL-neuron spike-history shift register
//      (does not exist in the inference-only design at all -- inference only
//      ever needs the CURRENT tick's spike vector). Replicated to every lane,
//      same treatment spike_frozen already gets.
//   2. A second barrier phase: after ALL lanes report tick_done (this tick's
//      spike_out is valid for every lane), pulse stdp_start on every lane
//      using the PRE-this-tick history window, wait for all lanes' stdp_done,
//      THEN shift this tick's own spike_out into the window (becomes "1 tick
//      ago" for the NEXT tick) -- ordering matters: STDP for tick T must see
//      history strictly BEFORE tick T, matching snm_network.py predict_
//      weights()'s "hist[j] = spikes j+1 ticks ago" semantics exactly.
//
// spike_out/vmem_out_flat readback assembly is IDENTICAL to snm_infer_multilane.v.
module snm_infer_multilane_stdp #(
    parameter integer NUM_LANES = 16,
    parameter integer N_MAX     = 1024,
    parameter integer N_MAX_W   = $clog2(N_MAX),
    parameter integer LOCAL_D   = (N_MAX + NUM_LANES - 1) / NUM_LANES,
    parameter integer SYN_CAP_PER_LANE = 65536,
    parameter integer WEIGHT_W  = 8,
    parameter integer ACCUM_W   = 32,
    parameter integer DATA_W    = 16,
    parameter integer REF_W     = 8,
    parameter integer STDP_WINDOW = 5,
    parameter integer DPTR_W    = $clog2(SYN_CAP_PER_LANE + 1),
    parameter integer LANE_W    = (NUM_LANES < 2) ? 1 : $clog2(NUM_LANES),
    parameter integer LDPTR_W   = $clog2(LOCAL_D + 1)
)(
    input  wire                     clk,
    input  wire                     reset_n,

    // ---- shared, lane-selected synapse config load ----
    input  wire [LANE_W-1:0]        cfg_lane,
    input  wire                     cfg_dptr_we,
    input  wire [LDPTR_W-1:0]       cfg_dptr_idx,
    input  wire [DPTR_W-1:0]        cfg_dptr_wdata,
    input  wire                     cfg_syn_we,
    input  wire [DPTR_W-1:0]        cfg_syn_idx,
    input  wire [N_MAX_W-1:0]       cfg_syn_src,
    input  wire signed [WEIGHT_W-1:0] cfg_syn_weight,

    // ---- shared, lane-selected synapse weight readback ----
    input  wire                     cfg_syn_re,
    output wire                     cfg_syn_rd_valid,
    output wire [N_MAX_W-1:0]       cfg_syn_rd_src,
    output wire signed [WEIGHT_W-1:0] cfg_syn_rd_weight,

    // ---- shared, lane-selected STDP param config load ----
    input  wire                       cfg_stdp_we,
    input  wire [$clog2(STDP_WINDOW):0] cfg_stdp_idx,
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_apos,
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_aneg,
    input  wire                       stdp_global_enable,

    // ---- shared, lane-selected neuron-param config load ----
    input  wire                          cfg_param_we,
    input  wire [2:0]                    cfg_param_field,
    input  wire [LDPTR_W-1:0]            cfg_param_idx,
    input  wire signed [DATA_W-1:0]      cfg_param_threshold,
    input  wire [DATA_W-1:0]             cfg_param_leak,
    input  wire signed [DATA_W-1:0]      cfg_param_reset_state,
    input  wire [REF_W-1:0]              cfg_param_refrac_period,
    input  wire                          cfg_param_input_enable,

    // ---- shared, lane-selected per-tick external-input load ----
    input  wire                          in_we,
    input  wire [LANE_W-1:0]             in_lane,
    input  wire [LDPTR_W-1:0]            in_idx,
    input  wire signed [DATA_W-1:0]      in_value,
    input  wire                          input_valid,

    // ---- per-tick run: ONE pulse advances a full tick (inference + STDP) ----
    input  wire                     spike_load,
    input  wire [N_MAX-1:0]         spike_in,
    input  wire                     tick_start,
    output wire                     tick_busy,
    output reg                      tick_done,       // pulses once inference+STDP both settle
    output reg  [31:0]              tick_cycles,     // inference-phase cycles (matches snm_infer_multilane.v)
    output reg  [31:0]              stdp_tick_cycles,// STDP-phase cycles for this tick

    // ---- results: global neuron id -> fired bit / post-tick membrane ----
    output wire [N_MAX-1:0]          spike_out,
    output wire signed [N_MAX*DATA_W-1:0] vmem_out_flat
);
    // Effective STDP window depth = superneuromat's t = min(stdp_time_steps,
    // len(spike_train)-1), i.e. min(STDP_WINDOW, tick_index). 0 during the very
    // first tick (=> no STDP update at all), then +1 per tick, saturating at
    // STDP_WINDOW once the history window is fully populated.
    reg [7:0] stdp_win_eff;
    // The lane's spike_frozen / spike_out_global ports are sized to the source
    // ADDRESS space (1<<SRC_W = 1<<N_MAX_W), which is >= N_MAX and STRICTLY
    // greater whenever N_MAX is not a power of two (e.g. N_MAX=448 -> 512).
    // Connecting the N_MAX-wide spike vectors directly triggers a port-width
    // padding warning per lane on those boards. Widening the internal spike
    // signals to SPK_W here makes the port connections exact-width (silencing
    // the warnings) and is functionally identical: source ids only ever range
    // over [0, N_MAX), so the extra high bits are always 0 and never a valid
    // lookup index. (2026-07-24 cleanup.)
    localparam integer SPK_W = 1 << N_MAX_W;
    reg [SPK_W-1:0] spike_frozen [0:NUM_LANES-1];
    integer li;
    always @(posedge clk) begin
        if (spike_load)
            for (li = 0; li < NUM_LANES; li = li + 1)
                spike_frozen[li] <= spike_in;   // zero-extends N_MAX -> SPK_W
    end

    // ---- spike-history refresh (2026-07-24: source-indexed BRAM, replaces
    // the old flat [STDP_WINDOW][N_MAX] combinationally-muxed hist/
    // src_hist_flat -- see snm_gather_lane_stdp.v's own comment for the full
    // root cause and rationale). Each lane now owns its own history memory
    // and refreshes it locally from spike_out (already assembled below by
    // g_readback -- a plain wire, no extra storage needed here) once per
    // tick, broadcast via hist_refresh_start_r exactly like spike_frozen's
    // own "registered input, replicated to every lane" treatment. Only
    // control/status wires live at this level now; no bulk history storage.
    wire [NUM_LANES-1:0] lane_hist_refresh_done;
    reg  hist_refresh_start_r;

    // SPK_W-wide zero-extension of the assembled spike_out (see spike_frozen's
    // comment above) so the .spike_out_global port connection is exact-width.
    wire [SPK_W-1:0] spike_out_ext;
    assign spike_out_ext = spike_out;

    wire [NUM_LANES-1:0] lane_tick_busy, lane_tick_done;
    wire [NUM_LANES-1:0] lane_stdp_busy, lane_stdp_done;
    wire [LOCAL_D-1:0] lane_spike [0:NUM_LANES-1];
    wire signed [LOCAL_D*DATA_W-1:0] lane_vmem_flat [0:NUM_LANES-1];
    wire [31:0] lane_cycles [0:NUM_LANES-1];
    wire [31:0] lane_stdp_cycles [0:NUM_LANES-1];
    reg stdp_start_r;

    // ---- synapse readback per-lane results (2026-07-25): every lane's
    // gather module answers cfg_syn_re independently (each has its own
    // syn_row), but only the cfg_lane-selected lane's answer is real -- same
    // "broadcast request masked by cfg_lane, mux the response back by
    // cfg_lane" pattern the write-side cfg_syn_we strobe already uses in the
    // generate block below, just for a read instead of a write.
    wire [NUM_LANES-1:0] lane_syn_rd_valid;
    wire [N_MAX_W-1:0] lane_syn_rd_src [0:NUM_LANES-1];
    wire signed [WEIGHT_W-1:0] lane_syn_rd_weight [0:NUM_LANES-1];
    assign cfg_syn_rd_valid  = lane_syn_rd_valid[cfg_lane];
    assign cfg_syn_rd_src    = lane_syn_rd_src[cfg_lane];
    assign cfg_syn_rd_weight = lane_syn_rd_weight[cfg_lane];

    genvar g;
    generate
        for (g = 0; g < NUM_LANES; g = g + 1) begin : g_lane
            snm_infer_lane_stdp #(
                .LOCAL_D(LOCAL_D), .SYN_CAP(SYN_CAP_PER_LANE), .SRC_W(N_MAX_W),
                .WEIGHT_W(WEIGHT_W), .ACCUM_W(ACCUM_W), .DATA_W(DATA_W), .REF_W(REF_W),
                .STDP_WINDOW(STDP_WINDOW)
            ) u_lane (
                .clk(clk), .reset_n(reset_n),
                .cfg_dptr_we(cfg_dptr_we && (cfg_lane == g[LANE_W-1:0])),
                .cfg_dptr_idx(cfg_dptr_idx),
                .cfg_dptr_wdata(cfg_dptr_wdata),
                .cfg_syn_we(cfg_syn_we && (cfg_lane == g[LANE_W-1:0])),
                .cfg_syn_idx(cfg_syn_idx),
                .cfg_syn_src(cfg_syn_src),
                .cfg_syn_weight(cfg_syn_weight),
                .cfg_syn_re(cfg_syn_re && (cfg_lane == g[LANE_W-1:0])),
                .cfg_syn_rd_valid(lane_syn_rd_valid[g]),
                .cfg_syn_rd_src(lane_syn_rd_src[g]),
                .cfg_syn_rd_weight(lane_syn_rd_weight[g]),
                .cfg_stdp_we(cfg_stdp_we && (cfg_lane == g[LANE_W-1:0])),
                .cfg_stdp_idx(cfg_stdp_idx),
                .cfg_stdp_apos(cfg_stdp_apos),
                .cfg_stdp_aneg(cfg_stdp_aneg),
                .stdp_global_enable(stdp_global_enable),
                .stdp_win_eff(stdp_win_eff),
                .cfg_param_we(cfg_param_we && (cfg_lane == g[LANE_W-1:0])),
                .cfg_param_field(cfg_param_field),
                .cfg_param_idx(cfg_param_idx),
                .cfg_param_threshold(cfg_param_threshold),
                .cfg_param_leak(cfg_param_leak),
                .cfg_param_reset_state(cfg_param_reset_state),
                .cfg_param_refrac_period(cfg_param_refrac_period),
                .cfg_param_input_enable(cfg_param_input_enable),
                .in_we(in_we && (in_lane == g[LANE_W-1:0])),
                .in_idx(in_idx),
                .in_value(in_value),
                .input_valid(input_valid),
                .tick_start(tick_start),
                .tick_busy(lane_tick_busy[g]),
                .tick_done(lane_tick_done[g]),
                .spike_frozen(spike_frozen[g]),
                .spike_out(lane_spike[g]),
                .vmem_out_flat(lane_vmem_flat[g]),
                .cycles_this_run(lane_cycles[g]),
                .stdp_start(stdp_start_r),
                .stdp_busy(lane_stdp_busy[g]),
                .stdp_done(lane_stdp_done[g]),
                .spike_out_global(spike_out_ext),
                .hist_refresh_start(hist_refresh_start_r),
                .hist_refresh_done(lane_hist_refresh_done[g]),
                .stdp_cycles_this_run(lane_stdp_cycles[g])
            );
        end
    endgenerate

    genvar r;
    generate
        for (r = 0; r < N_MAX; r = r + 1) begin : g_readback
            assign spike_out[r] = lane_spike[r % NUM_LANES][r / NUM_LANES];
            assign vmem_out_flat[r*DATA_W +: DATA_W] =
                lane_vmem_flat[r % NUM_LANES][(r / NUM_LANES) * DATA_W +: DATA_W];
        end
    endgenerate

    // ---- three-phase barrier: inference phase (existing pattern), STDP
    // phase, then refresh every lane's history memory using THIS tick's
    // spike_out (2026-07-24: B_SHIFT now waits for each lane's OWN sequential
    // BRAM refresh -- see snm_gather_lane_stdp.v -- instead of doing a single
    // central shift here in one cycle; ~(1<<N_MAX_W) cycles instead of 1, a
    // ~1% addition to the dense-worst-case tick budget). ----
    localparam [1:0] B_IDLE  = 2'd0;
    localparam [1:0] B_INFER = 2'd1;
    localparam [1:0] B_STDP  = 2'd2;
    localparam [1:0] B_SHIFT = 2'd3;

    reg [1:0] bstate;
    reg [NUM_LANES-1:0] infer_seen, stdp_seen, hist_refresh_seen;
    wire [NUM_LANES-1:0] infer_seen_next = infer_seen | lane_tick_done;
    wire [NUM_LANES-1:0] stdp_seen_next  = stdp_seen  | lane_stdp_done;
    wire [NUM_LANES-1:0] hist_refresh_seen_next = hist_refresh_seen | lane_hist_refresh_done;

    assign tick_busy = (bstate != B_IDLE);

    // Synchronous reset (2026-07-24, future-ASIC methodology fix -- see
    // snm_gather_lane_stdp.v's always-block comment for the full rationale).
    always @(posedge clk) begin
        if (!reset_n) begin
            bstate <= B_IDLE;
            infer_seen <= {NUM_LANES{1'b0}};
            stdp_seen  <= {NUM_LANES{1'b0}};
            hist_refresh_seen <= {NUM_LANES{1'b0}};
            tick_done  <= 1'b0;
            tick_cycles <= 32'd0;
            stdp_tick_cycles <= 32'd0;
            stdp_start_r <= 1'b0;
            hist_refresh_start_r <= 1'b0;
            stdp_win_eff <= 0;
        end else begin
            tick_done    <= 1'b0;
            stdp_start_r <= 1'b0;
            hist_refresh_start_r <= 1'b0;

            case (bstate)
                B_IDLE: begin
                    if (tick_start) begin
                        infer_seen  <= {NUM_LANES{1'b0}};
                        tick_cycles <= 32'd1;
                        bstate      <= B_INFER;
                    end
                end
                B_INFER: begin
                    tick_cycles <= tick_cycles + 1'b1;
                    infer_seen  <= infer_seen_next;
                    if (&infer_seen_next) begin
                        stdp_seen        <= {NUM_LANES{1'b0}};
                        stdp_tick_cycles <= 32'd1;
                        stdp_start_r     <= 1'b1;   // pulse: all lanes' spike_out now valid
                        bstate           <= B_STDP;
                    end
                end
                B_STDP: begin
                    stdp_tick_cycles <= stdp_tick_cycles + 1'b1;
                    stdp_seen        <= stdp_seen_next;
                    if (&stdp_seen_next) begin
                        hist_refresh_seen    <= {NUM_LANES{1'b0}};
                        hist_refresh_start_r <= 1'b1;  // pulse: begin every lane's history refresh
                        bstate <= B_SHIFT;
                    end
                end
                B_SHIFT: begin
                    stdp_tick_cycles  <= stdp_tick_cycles + 1'b1;
                    hist_refresh_seen <= hist_refresh_seen_next;
                    if (&hist_refresh_seen_next) begin
                        // grow the effective window for the NEXT tick (saturating)
                        if (stdp_win_eff < STDP_WINDOW[7:0])
                            stdp_win_eff <= stdp_win_eff + 1'b1;
                        tick_done <= 1'b1;
                        bstate    <= B_IDLE;
                    end
                end
                default: bstate <= B_IDLE;
            endcase
        end
    end
endmodule
