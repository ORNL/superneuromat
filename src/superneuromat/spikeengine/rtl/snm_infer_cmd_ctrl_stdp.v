`timescale 1us/1ns

// STDP-capable command controller (2026-07-21, npu_stdp_dev experiment).
// New sibling of snm_infer_cmd_ctrl.v (that file stays byte-for-byte
// untouched, same convention as every other file in this experiment) --
// targets snm_infer_multilane_stdp instead of snm_infer_multilane.
//
// snm_infer_multilane_stdp exposes a SINGLE tick_start that drives BOTH the
// inference phase and the STDP phase internally (its own two-phase barrier
// FSM), with tick_done pulsing only once both have settled -- so the run/
// tick sequencing here (OP_RUN_START -> S_TICK_PRE -> S_TICK_WAIT ->
// S_TICK_CAP) is IDENTICAL to snm_infer_cmd_ctrl.v, unchanged. The only new
// material is config-write passthrough for the STDP parameter table
// (Apos/Aneg per window slot) and the global STDP enable flag, plus a
// readback path for stdp_tick_cycles (useful for the same real-cycle-cost
// profiling NOTES.md's progress update 4 did via raw testbench signals --
// now available over the wire without needing simulation hierarchy access).
module snm_infer_cmd_ctrl_stdp #(
    parameter integer N_MAX     = 1024,
    parameter integer NUM_LANES = 16,
    parameter integer DATA_W    = 16,
    parameter integer WEIGHT_W  = 8,
    parameter integer ACCUM_W   = 32,
    parameter integer REF_W     = 8,
    parameter integer STDP_WINDOW = 5,
    // Which neuron id the low LED bit (spike_mon[0]) reflects. Default 0 keeps
    // the historical behaviour (LEDs = neurons 0..15). Set to the first OUTPUT
    // neuron id (e.g. 64 for the 64-input/10-output digits net) so the board's
    // LEDs visually show the classifier's OUTPUT-class firing instead of the
    // input/pixel layer. (2026-07-25)
    parameter integer SPIKE_MON_BASE = 0,
    parameter integer N_MAX_W = $clog2(N_MAX),
    parameter integer LOCAL_D = (N_MAX + NUM_LANES - 1) / NUM_LANES,
    parameter integer SYN_CAP_PER_LANE = LOCAL_D * N_MAX,
    parameter integer DPTR_W  = $clog2(SYN_CAP_PER_LANE + 1),
    parameter integer LANE_W  = (NUM_LANES < 2) ? 1 : $clog2(NUM_LANES),
    parameter integer LDPTR_W = $clog2(LOCAL_D + 1),
    parameter integer STDP_IDX_W = $clog2(STDP_WINDOW) + 1
)(
    input  wire        clk,
    input  wire        reset_n,

    // ---- internal command bus (front-end compatible) ----
    input  wire        cmd_valid,
    output reg         cmd_ready,
    input  wire [7:0]  cmd_opcode,
    input  wire [7:0]  cmd_sel,
    input  wire [15:0] cmd_addr,
    input  wire [31:0] cmd_wdata,
    output reg         rsp_valid,
    input  wire        rsp_ready,
    output reg  [7:0]  rsp_status,
    output reg  [31:0] rsp_rdata,

    output wire        busy,
    output wire        irq,
    output wire [15:0] spike_mon,

    // ---- engine soft-reset (2026-07-25): pulsed high for one cycle on
    // OP_ENGINE_RESET. The engine top ANDs ~soft_reset into the multilane
    // engine's reset_n, synchronously clearing all engine datapath state
    // (input_value[]/vmem/refractory counters/stdp_win_eff/pipeline/FSMs)
    // WITHOUT a bitstream reload -- the state that a config reload alone can
    // NOT clear (see the hardware inter-run divergence writeup). cmd_ctrl is
    // deliberately NOT reset by it (it must stay alive to answer the command);
    // cur_spikes IS cleared here in step, so the post-reset spike_in feedback
    // starts clean too. The history memory is intentionally not force-cleared
    // (it is never bulk-reset, for BRAM inference) -- stdp_win_eff resetting to
    // 0 gates it out of use until it has been refreshed with fresh post-reset
    // spikes, so stale history self-flushes within the warm-up window. ----
    output reg         soft_reset,

    // ---- engine config/run/readback ports ----
    output reg  [LANE_W-1:0]        cfg_lane,
    output reg                      cfg_dptr_we,
    output reg  [LDPTR_W-1:0]       cfg_dptr_idx,
    output reg  [DPTR_W-1:0]        cfg_dptr_wdata,
    output reg                      cfg_syn_we,
    output reg  [DPTR_W-1:0]        cfg_syn_idx,
    output reg  [N_MAX_W-1:0]       cfg_syn_src,
    output reg  signed [WEIGHT_W-1:0] cfg_syn_weight,
    // ---- synapse weight readback (2026-07-25) ----
    output reg                      cfg_syn_re,
    input  wire                     cfg_syn_rd_valid,
    input  wire [N_MAX_W-1:0]       cfg_syn_rd_src,
    input  wire signed [WEIGHT_W-1:0] cfg_syn_rd_weight,
    output reg                      cfg_stdp_we,
    output reg  [STDP_IDX_W-1:0]    cfg_stdp_idx,
    output reg  signed [WEIGHT_W-1:0] cfg_stdp_apos,
    output reg  signed [WEIGHT_W-1:0] cfg_stdp_aneg,
    output reg                      stdp_global_enable,
    output reg                      cfg_param_we,
    output reg  [2:0]               cfg_param_field,
    output reg  [LDPTR_W-1:0]       cfg_param_idx,
    output reg  signed [DATA_W-1:0] cfg_param_threshold,
    output reg  [DATA_W-1:0]        cfg_param_leak,
    output reg  signed [DATA_W-1:0] cfg_param_reset_state,
    output reg  [REF_W-1:0]         cfg_param_refrac_period,
    output reg                      cfg_param_input_enable,
    output reg                      in_we,
    output reg  [LANE_W-1:0]        in_lane,
    output reg  [LDPTR_W-1:0]       in_idx,
    output reg  signed [DATA_W-1:0] in_value,
    output reg                      input_valid,
    output reg                      spike_load,
    output reg  [N_MAX-1:0]         spike_in,
    output reg                      tick_start,
    input  wire                     tick_busy,
    input  wire                     tick_done,
    // Real inference-phase cycle count for the last tick (2026-07-22 addition
    // -- this was NOT exposed before; total real per-tick time could only be
    // reported as stdp cycles [real hardware] + inference cycles [simulation
    // cross-validated, not directly measured]. OP_READ_STATUS with sel=1 now
    // returns this directly off the chip, so the WHOLE per-tick timing number
    // can be a real hardware measurement, not half-simulated).
    input  wire [31:0]              tick_cycles,
    input  wire [31:0]              stdp_tick_cycles,
    input  wire [N_MAX-1:0]         spike_out
);
    localparam integer SPIKE_WORDS = (N_MAX + 31) / 32;
    localparam integer SPIKE_WORD_MAX = (SPIKE_WORDS < 2) ? 0 : ($clog2(SPIKE_WORDS) - 1);

    // opcodes -- identical set to snm_infer_cmd_ctrl.v, no new opcode needed
    // (tick_start already runs inference+STDP as one unit at the engine level)
    localparam [7:0] OP_WRITE_CONFIG       = 8'h01;
    // Read counterpart of OP_WRITE_CONFIG (2026-07-25). Only IL_SYN is
    // supported so far (host needs to read back hardware-learned weights
    // after on-chip STDP training, not just infer them via determinism) --
    // other selectors return ST_INVALID_SEL, same pattern OP_WRITE_CONFIG's
    // own default case uses. Addressing (cur_lane/IL_SYN_ADDR_HI/addr_r) is
    // IDENTICAL to an IL_SYN write: set lane (IL_SET_LANE) and high address
    // bits (IL_SYN_ADDR_HI) exactly as before a write, then issue this with
    // cmd_addr = the low bits of the synapse index.
    localparam [7:0] OP_READ_CONFIG        = 8'h02;
    localparam [7:0] OP_RUN_START          = 8'h05;
    localparam [7:0] OP_READ_STATUS        = 8'h09;
    localparam [7:0] OP_READ_OUTPUT        = 8'h0a;
    localparam [7:0] OP_INPUT_VECTOR_WRITE = 8'h0c;
    localparam [7:0] OP_CLEAR_ERROR        = 8'h10;
    localparam [7:0] OP_ENGINE_RESET       = 8'h11;  // soft-reset all engine datapath state (no bitstream reload)

    // inference config selectors -- same codes as snm_infer_cmd_ctrl.v
    localparam [7:0] IL_SET_LANE     = 8'h1b;
    localparam [7:0] IL_DPTR         = 8'h0d;
    localparam [7:0] IL_SYN          = 8'h0e;
    localparam [7:0] IL_THRESHOLD    = 8'h06;
    localparam [7:0] IL_LEAK         = 8'h07;
    localparam [7:0] IL_RESET_STATE  = 8'h08;
    localparam [7:0] IL_REFRAC       = 8'h09;
    localparam [7:0] IL_INPUT_ENABLE = 8'h0a;
    localparam [7:0] IL_INPUT_VALUE  = 8'h0f;
    localparam [7:0] IL_INPUT_VEC_VAL= 8'h1a;
    localparam [7:0] IL_SYN_ADDR_HI  = 8'h19;
    // NEW STDP selectors (picked from the unused range above the existing
    // IL_* codes, none of which reach 0x1c+)
    localparam [7:0] IL_STDP_TABLE   = 8'h1c;  // addr=window idx, wdata={aneg[15:8],apos[7:0]}
    localparam [7:0] IL_STDP_ENABLE  = 8'h1d;  // wdata[0] = stdp_global_enable

    localparam [2:0] PF_THR=3'd0, PF_LEAK=3'd1, PF_RST=3'd2, PF_RP=3'd3, PF_IEN=3'd4;

    localparam [7:0] ST_OK=8'h00, ST_BUSY=8'h01, ST_INVALID_OP=8'h02,
                     ST_INVALID_SEL=8'h03, ST_ADDR_RANGE=8'h04;

    reg [7:0]  op_r;
    reg [7:0]  sel_r;
    reg [15:0] addr_r;
    reg [31:0] wdata_r;

    reg [LANE_W-1:0] cur_lane;
    reg [15:0]       syn_addr_hi;
    wire [31:0] syn_idx_full = {syn_addr_hi, addr_r};
    reg signed [DATA_W-1:0] vec_value;
    reg               have_input;
    reg [N_MAX-1:0]   cur_spikes;
    reg               err_flag;
    reg [31:0]        last_stdp_cycles;   // latched at S_TICK_CAP for OP_READ_STATUS-adjacent readback
    reg [31:0]        last_tick_cycles;   // inference-phase cycles, same latch point

    wire [SPIKE_WORDS*32-1:0] spike_padded = {{(SPIKE_WORDS*32 - N_MAX){1'b0}}, cur_spikes};

    // spike_padded is N_MAX zero-extended to a whole number of 32-bit words, so
    // slicing [SPIKE_MON_BASE +: 16] is always in range even when
    // SPIKE_MON_BASE+16 exceeds N_MAX (the high bits read as 0). This also
    // covers the N_MAX < 16 case the old generate block special-cased.
    assign spike_mon = spike_padded[SPIKE_MON_BASE +: 16];

    localparam [3:0] S_IDLE      = 4'd0,
                     S_DECODE    = 4'd1,
                     S_VEC_EXPAND= 4'd2,
                     S_TICK_PRE  = 4'd3,
                     S_TICK_WAIT = 4'd4,
                     S_TICK_CAP  = 4'd5,
                     S_RESP      = 4'd6,
                     S_SYN_RD_WAIT = 4'd7;  // waiting for cfg_syn_rd_valid (2-cycle BRAM read latency)
    reg [3:0] state;

    reg [31:0] vec_mask;
    reg [15:0] vec_word;
    reg [5:0]  vec_bit;

    assign busy = (state != S_IDLE) || tick_busy;
    assign irq  = rsp_valid;

    integer vn;

    task clear_strobes;
        begin
            cfg_dptr_we <= 1'b0;
            cfg_syn_we  <= 1'b0;
            cfg_syn_re  <= 1'b0;
            cfg_stdp_we <= 1'b0;
            cfg_param_we<= 1'b0;
            in_we       <= 1'b0;
            spike_load  <= 1'b0;
            tick_start  <= 1'b0;
            soft_reset  <= 1'b0;   // one-cycle pulse, deasserted every cycle unless re-driven
        end
    endtask

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cmd_ready <= 1'b1;
            rsp_valid <= 1'b0;
            rsp_status<= ST_OK;
            rsp_rdata <= 32'd0;
            cur_lane  <= {LANE_W{1'b0}};
            syn_addr_hi <= 16'd0;
            vec_value <= {DATA_W{1'b0}};
            have_input<= 1'b0;
            cur_spikes<= {N_MAX{1'b0}};
            soft_reset <= 1'b0;
            err_flag  <= 1'b0;
            last_stdp_cycles <= 32'd0;
            last_tick_cycles <= 32'd0;
            state     <= S_IDLE;
            cfg_lane  <= {LANE_W{1'b0}};
            cfg_dptr_idx <= {LDPTR_W{1'b0}};
            cfg_dptr_wdata <= {DPTR_W{1'b0}};
            cfg_syn_idx <= {DPTR_W{1'b0}};
            cfg_syn_src <= {N_MAX_W{1'b0}};
            cfg_syn_weight <= {WEIGHT_W{1'b0}};
            cfg_stdp_idx <= {STDP_IDX_W{1'b0}};
            cfg_stdp_apos <= {WEIGHT_W{1'b0}};
            cfg_stdp_aneg <= {WEIGHT_W{1'b0}};
            stdp_global_enable <= 1'b0;
            cfg_param_field <= 3'd0;
            cfg_param_idx <= {LDPTR_W{1'b0}};
            cfg_param_threshold <= {DATA_W{1'b0}};
            cfg_param_leak <= {DATA_W{1'b0}};
            cfg_param_reset_state <= {DATA_W{1'b0}};
            cfg_param_refrac_period <= {REF_W{1'b0}};
            cfg_param_input_enable <= 1'b0;
            in_lane <= {LANE_W{1'b0}};
            in_idx  <= {LDPTR_W{1'b0}};
            in_value<= {DATA_W{1'b0}};
            input_valid <= 1'b0;
            spike_in <= {N_MAX{1'b0}};
            cfg_dptr_we <= 1'b0; cfg_syn_we <= 1'b0; cfg_syn_re <= 1'b0; cfg_stdp_we <= 1'b0; cfg_param_we <= 1'b0;
            in_we <= 1'b0; spike_load <= 1'b0; tick_start <= 1'b0;
            vec_mask <= 32'd0; vec_word <= 16'd0; vec_bit <= 6'd0;
            op_r <= 8'd0; sel_r <= 8'd0; addr_r <= 16'd0; wdata_r <= 32'd0;
        end else begin
            clear_strobes;

            case (state)
                S_IDLE: begin
                    cmd_ready <= 1'b1;
                    if (cmd_valid && cmd_ready) begin
                        op_r    <= cmd_opcode;
                        sel_r   <= cmd_sel;
                        addr_r  <= cmd_addr;
                        wdata_r <= cmd_wdata;
                        cmd_ready <= 1'b0;
                        state <= S_DECODE;
                    end
                end

                S_DECODE: begin
                    rsp_status <= ST_OK;
                    rsp_rdata  <= 32'd0;
                    case (op_r)
                        OP_WRITE_CONFIG: begin
                            case (sel_r)
                                IL_SET_LANE: begin
                                    cur_lane <= wdata_r[LANE_W-1:0];
                                    state <= S_RESP;
                                end
                                IL_DPTR: begin
                                    cfg_lane <= cur_lane;
                                    cfg_dptr_idx <= addr_r[LDPTR_W-1:0];
                                    cfg_dptr_wdata <= wdata_r[DPTR_W-1:0];
                                    cfg_dptr_we <= 1'b1;
                                    state <= S_RESP;
                                end
                                IL_SYN_ADDR_HI: begin
                                    syn_addr_hi <= wdata_r[15:0];
                                    state <= S_RESP;
                                end
                                IL_SYN: begin
                                    // 2026-07-27: field offsets are WEIGHT_W-
                                    // parametric (was a hardcoded 8, which
                                    // overlapped src+weight once WEIGHT_W grew
                                    // past 8). weight occupies wdata[WEIGHT_W-1:0],
                                    // src the next N_MAX_W bits above it. At
                                    // WEIGHT_W=16/N_MAX_W=8 that is weight[15:0]
                                    // + src[23:16] = 24 bits, still inside the
                                    // 32-bit command word.
                                    cfg_lane <= cur_lane;
                                    cfg_syn_idx <= syn_idx_full[DPTR_W-1:0];
                                    cfg_syn_src <= wdata_r[WEIGHT_W +: N_MAX_W];
                                    cfg_syn_weight <= wdata_r[WEIGHT_W-1:0];
                                    cfg_syn_we <= 1'b1;
                                    state <= S_RESP;
                                end
                                IL_STDP_TABLE: begin
                                    // addr = window slot index (0..STDP_WINDOW-1);
                                    // Apos=wdata[WEIGHT_W-1:0], Aneg=wdata[2*WEIGHT_W-1:WEIGHT_W]
                                    // -- same "pack narrow fields into one
                                    // 32-bit word" convention IL_SYN uses, and
                                    // likewise WEIGHT_W-parametric (was a
                                    // hardcoded 8-bit split). At WEIGHT_W=16
                                    // that is apos[15:0]+aneg[31:16], filling
                                    // the whole 32-bit word exactly.
                                    cfg_lane <= cur_lane;
                                    cfg_stdp_idx  <= addr_r[STDP_IDX_W-1:0];
                                    cfg_stdp_apos <= wdata_r[WEIGHT_W-1:0];
                                    cfg_stdp_aneg <= wdata_r[WEIGHT_W +: WEIGHT_W];
                                    cfg_stdp_we   <= 1'b1;
                                    state <= S_RESP;
                                end
                                IL_STDP_ENABLE: begin
                                    stdp_global_enable <= wdata_r[0];
                                    state <= S_RESP;
                                end
                                IL_THRESHOLD, IL_LEAK, IL_RESET_STATE,
                                IL_REFRAC, IL_INPUT_ENABLE: begin
                                    cfg_lane <= cur_lane;
                                    cfg_param_idx <= addr_r[LDPTR_W-1:0];
                                    cfg_param_threshold   <= wdata_r[DATA_W-1:0];
                                    cfg_param_leak        <= wdata_r[DATA_W-1:0];
                                    cfg_param_reset_state <= wdata_r[DATA_W-1:0];
                                    cfg_param_refrac_period<= wdata_r[REF_W-1:0];
                                    cfg_param_input_enable<= wdata_r[0];
                                    cfg_param_field <=
                                        (sel_r==IL_THRESHOLD)   ? PF_THR :
                                        (sel_r==IL_LEAK)        ? PF_LEAK:
                                        (sel_r==IL_RESET_STATE) ? PF_RST :
                                        (sel_r==IL_REFRAC)      ? PF_RP  : PF_IEN;
                                    cfg_param_we <= 1'b1;
                                    state <= S_RESP;
                                end
                                IL_INPUT_VALUE: begin
                                    in_lane <= cur_lane;
                                    in_idx  <= addr_r[LDPTR_W-1:0];
                                    in_value<= wdata_r[DATA_W-1:0];
                                    in_we   <= 1'b1;
                                    have_input <= 1'b1;
                                    state <= S_RESP;
                                end
                                IL_INPUT_VEC_VAL: begin
                                    vec_value <= wdata_r[DATA_W-1:0];
                                    state <= S_RESP;
                                end
                                default: begin
                                    rsp_status <= ST_INVALID_SEL;
                                    state <= S_RESP;
                                end
                            endcase
                        end

                        OP_READ_CONFIG: begin
                            case (sel_r)
                                IL_SYN: begin
                                    cfg_lane <= cur_lane;
                                    cfg_syn_idx <= syn_idx_full[DPTR_W-1:0];
                                    cfg_syn_re  <= 1'b1;
                                    state <= S_SYN_RD_WAIT;
                                end
                                default: begin
                                    rsp_status <= ST_INVALID_SEL;
                                    state <= S_RESP;
                                end
                            endcase
                        end

                        OP_INPUT_VECTOR_WRITE: begin
                            if (addr_r >= SPIKE_WORDS[15:0]) begin
                                rsp_status <= ST_ADDR_RANGE;
                                state <= S_RESP;
                            end else begin
                                vec_mask <= wdata_r;
                                vec_word <= addr_r;
                                vec_bit  <= 6'd0;
                                have_input <= 1'b1;
                                state <= S_VEC_EXPAND;
                            end
                        end

                        OP_RUN_START: begin
                            spike_in   <= cur_spikes;
                            spike_load <= 1'b1;
                            input_valid<= have_input;
                            state <= S_TICK_PRE;
                        end

                        OP_READ_OUTPUT: begin
                            if (addr_r >= SPIKE_WORDS[15:0]) begin
                                rsp_status <= ST_ADDR_RANGE;
                            end else begin
                                rsp_rdata <= spike_padded[addr_r[SPIKE_WORD_MAX:0]*32 +: 32];
                            end
                            state <= S_RESP;
                        end

                        OP_READ_STATUS: begin
                            // sel_r sub-selects which status word comes back
                            // (2026-07-22 addition): sel=0 is the original
                            // status+stdp-cycles word (unchanged, for backward
                            // compat); sel=1/2 return the FULL 32-bit real
                            // inference/STDP cycle counters directly, so a
                            // total real per-tick time can be reported from
                            // hardware measurements alone, no simulation
                            // cross-reference needed.
                            if (sel_r == 8'd1)
                                rsp_rdata <= last_tick_cycles;
                            else if (sel_r == 8'd2)
                                rsp_rdata <= last_stdp_cycles;
                            else
                                // bit1 (was reserved-0 in the inference-only
                                // controller) carries stdp_global_enable so the
                                // host can read back what it last configured;
                                // rdata[31:16] carries the last tick's STDP-
                                // phase cycle count, truncated to 16 bits (see
                                // sel=2 above for the full 32-bit value).
                                rsp_rdata <= {last_stdp_cycles[15:0], 8'd0, 4'd0,
                                              stdp_global_enable, err_flag, tick_busy, 1'b0};
                            state <= S_RESP;
                        end

                        OP_CLEAR_ERROR: begin
                            err_flag <= 1'b0;
                            state <= S_RESP;
                        end

                        OP_ENGINE_RESET: begin
                            // Pulse the engine soft-reset for one cycle (the
                            // engine's synchronous reset only needs a single
                            // clocked edge to clear every datapath register),
                            // and clear cur_spikes so the post-reset spike_in
                            // feedback starts from all-zero. Weights/params are
                            // reloaded by config after this, so they are not the
                            // concern here -- this clears the NON-config-
                            // resettable state (see the soft_reset port comment).
                            soft_reset <= 1'b1;
                            cur_spikes <= {N_MAX{1'b0}};
                            state <= S_RESP;
                        end

                        default: begin
                            rsp_status <= ST_INVALID_OP;
                            err_flag <= 1'b1;
                            state <= S_RESP;
                        end
                    endcase
                end

                S_VEC_EXPAND: begin
                    if (vec_bit >= 6'd32) begin
                        state <= S_RESP;
                    end else begin
                        vn = vec_word*32 + vec_bit;
                        if (vec_mask[vec_bit] && (vn < N_MAX)) begin
                            in_lane <= vn % NUM_LANES;
                            in_idx  <= vn / NUM_LANES;
                            in_value<= vec_value;
                            in_we   <= 1'b1;
                        end
                        vec_bit <= vec_bit + 1'b1;
                    end
                end

                S_TICK_PRE: begin
                    tick_start <= 1'b1;
                    state <= S_TICK_WAIT;
                end

                S_TICK_WAIT: begin
                    if (tick_done) state <= S_TICK_CAP;
                end

                S_TICK_CAP: begin
                    cur_spikes  <= spike_out;
                    last_stdp_cycles <= stdp_tick_cycles;
                    last_tick_cycles <= tick_cycles;
                    input_valid <= 1'b0;
                    have_input  <= 1'b0;
                    state <= S_RESP;
                end

                S_SYN_RD_WAIT: begin
                    // cfg_syn_re was pulsed one cycle ago (clear_strobes
                    // already deasserted it this cycle); wait for the gather
                    // module's shared-BRAM-read pipeline (2-cycle latency,
                    // see snm_gather_lane_stdp.v) to present cfg_syn_rd_valid.
                    // rdata packing mirrors an IL_SYN WRITE's wdata exactly
                    // (weight in [WEIGHT_W-1:0], src in [WEIGHT_W +: N_MAX_W],
                    // WEIGHT_W-parametric -- see the IL_SYN write comment) so
                    // the host can round-trip a read result straight back into
                    // a write without re-packing it.
                    if (cfg_syn_rd_valid) begin
                        rsp_rdata[WEIGHT_W-1:0]        <= cfg_syn_rd_weight;
                        rsp_rdata[WEIGHT_W +: N_MAX_W]  <= cfg_syn_rd_src;
                        state <= S_RESP;
                    end
                end

                S_RESP: begin
                    rsp_valid <= 1'b1;
                    if (rsp_valid && rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
