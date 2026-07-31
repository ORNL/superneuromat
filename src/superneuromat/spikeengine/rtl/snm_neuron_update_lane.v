`timescale 1us/1ns

// The generated config header is included HERE (not just in the top) because
// SNM_INFER_NEURON_PIPE_4STAGE below is an `ifdef, and Vivado compiles each
// file as its own compilation unit -- a define made in another file's
// `include is invisible (same lesson already learned for snm_sram8_16k.v and
// snm_gather_lane.v).
`include "snm_config.vh"

// Neuron-update stage for one gather lane (2026-07-11, end-to-end pipeline step).
//
// Consumes a lane's private synapse accumulators (snm_gather_lane's accum_out_flat)
// and walks LOCAL_D neurons applying the SAME membrane update math as the existing
// serial core (spikeengine_core.v: leak_toward_reset + saturating_fold + strict
// `> threshold` spike decision + refractory hold), so this module's output is meant
// to be bit-exact against that reference for the propagation+neuron-update pipeline.
//
// PIPELINED 3-STAGE WALK (deep-review fix, 2026-07-11): the first version computed
// state-read -> leak -> 33-bit saturating fold -> threshold compare -> writeback in
// ONE cycle per neuron. That is the exact chain spikeengine_core.v documents as
// "a single-cycle ~21x CARRY4 path (Fmax ~59 MHz)" -- which that core had to split
// across ST_TICK_UPDATE_LEAK / ACCUM_WAIT / DECIDE to close 100 MHz -- and this
// module had additionally put a LOCAL_D:1 x ACCUM_W accumulator mux in front of it.
// Functionally identical in simulation (sim has no timing), but a near-certain
// 100 MHz timing failure on silicon-targeted synthesis. Restructured into the same
// proven 3-stage split, walking one neuron per cycle with 2 extra cycles of drain:
//   stage A: read per-neuron state/params (the muxes) + leak_toward_reset
//   stage B: saturating fold of the leaked vmem with the synapse accumulator
//   stage C: refractory / threshold decision + state writeback
// No forwarding is needed: each neuron is visited exactly once per tick, and a
// neuron's stage-C write lands >= 2 indices behind the neuron being read at stage
// A, so no same-address read-after-write can occur within a tick.
//
// Scope (matches the inference-only decision, see NOTES.md): external per-tick
// INPUT (tick_input_value/tick_input_enable in the core) is intentionally NOT
// wired here -- this stage covers the guaranteed 1ms inference path for a purely
// recurrent network. Adding external input is a follow-on (one more per-neuron
// config array + the core's tick_base mux between stages A and B).
//
// Per-neuron parameters (threshold, leak, reset_state, refrac_period) are
// CONFIG-LOADED, matching the core's per-neuron CTRL_PARAM_BASE row (these are
// not global constants in the reference either). vmem/refrac_count are RUNTIME
// STATE that persists across ticks (reset to 0 only on reset_n).
module snm_neuron_update_lane #(
    parameter integer LOCAL_D  = 32,
    parameter integer DATA_W   = 16,
    parameter integer ACCUM_W  = 32,
    parameter integer REF_W    = 8
)(
    input  wire clk,
    input  wire reset_n,

    // ---- per-neuron config load ----
    // cfg_param_field selects WHICH field cfg_param_we writes, so a field-at-a-time
    // host protocol (OP_WRITE_CONFIG carries one L_CFG_* field per command) can load
    // params without the controller shadowing them. PF_ALL (7) writes every field at
    // once from all five data ports (the bulk-load path the testbenches use).
    input  wire                          cfg_param_we,
    input  wire [2:0]                    cfg_param_field, // 0=thr 1=leak 2=reset 3=refrac 4=in_en 7=ALL
    input  wire [$clog2(LOCAL_D+1)-1:0]  cfg_param_idx,
    input  wire signed [DATA_W-1:0]      cfg_param_threshold,
    input  wire [DATA_W-1:0]             cfg_param_leak,
    input  wire signed [DATA_W-1:0]      cfg_param_reset_state,
    input  wire [REF_W-1:0]              cfg_param_refrac_period,
    input  wire                          cfg_param_input_enable,  // persistent: does this neuron accept external input

    // ---- per-tick external-input load (per-neuron value; write before start) ----
    // Matches the core's tick_input_value (a per-neuron, per-tick stimulus stored
    // in the neuron's state row; protocol v3 loads it as an "input vector"). A
    // single input_valid, sampled at start, gates whether external input applies
    // at all this tick -- exactly the core's (input_valid && input_enable) guard.
    input  wire                          in_we,
    input  wire [$clog2(LOCAL_D+1)-1:0]  in_idx,
    input  wire signed [DATA_W-1:0]      in_value,
    input  wire                          input_valid,

    // ---- per-tick run ----
    input  wire                          start,      // pulse: gather lane's run_done
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse when all LOCAL_D done

    input  wire signed [LOCAL_D*ACCUM_W-1:0] accum_in_flat,   // from snm_gather_lane

    // ---- results ----
    output reg  [LOCAL_D-1:0]                 spike_out,       // this tick's fired bits
    output wire signed [LOCAL_D*DATA_W-1:0]   vmem_out_flat    // post-tick membrane (debug/verify)
);
    localparam integer LD_W = $clog2(LOCAL_D+1);

    // ---- Option A (2026-07-29): SNM_NEURON_STATE_BRAM moves the per-neuron
    // state arrays from distributed LUTRAM / flip-flops into BLOCK RAM, to break
    // the O(N) LUT/FF neuron-count ceiling (per-lane u_neuron measured ~4225 FF,
    // ~60% of all FF, at LOCAL_D=64 -- see ARCH_SCALING_ANALYSIS.md). The lane
    // visits its LOCAL_D neurons ONE PER CYCLE, so single-port synchronous-read
    // memory is sufficient -- the arrays were only in fabric for the bulk async
    // reset and pipeline-read convenience. This is OPT-IN and REQUIRES the
    // A0-raw-read pipeline (4/5-stage): those register the array read
    // (`x_r0 <= arr[idx]`), which is BRAM-legal; the 3-stage path reads-and-
    // computes combinationally, which forces distributed RAM, so the define is
    // rejected there. Un-defined => byte-identical to the original fabric build.
`ifdef SNM_NEURON_STATE_BRAM
  `ifndef SNM_INFER_NEURON_PIPE_5STAGE
      // synthesis translate_off
      initial $fatal(1, "SNM_NEURON_STATE_BRAM requires SNM_INFER_NEURON_PIPE_5STAGE (the A0 raw-read pipeline the clear FSM is wired into)");
      // synthesis translate_on
  `endif
    // CONFIG params stay in fabric (original efficient LUTRAM/FF): they are
    // WIDTH-shallow (LOCAL_D deep), so forcing each of the 6 arrays to its own
    // block-RAM tile costs ~6 tiles/lane -- 48 tiles at K=8, which BLEW basys3's
    // 50-tile budget and spilled into pathological distributed logic (measured:
    // LUT 20437 -> 45022). Field-at-a-time host writes (cmd_ctrl uses PF_THR/
    // PF_LEAK/... not PF_ALL) would also force read-modify-write to pack them.
    // So Option A moves ONLY the pure RUNTIME state (vmem + refrac_count) to
    // block RAM -- engine-written (stage C, full word), never host-written, so
    // no RMW and no clear-walk write race. That is the dominant FF consumer
    // (~2048 FF/lane) and needs just 2 tiles/lane.
    reg signed [DATA_W-1:0] threshold    [0:LOCAL_D-1];
    reg        [DATA_W-1:0] leak         [0:LOCAL_D-1];
    reg signed [DATA_W-1:0] reset_state  [0:LOCAL_D-1];
    reg        [REF_W-1:0]  refrac_period[0:LOCAL_D-1];
    reg                     input_enable [0:LOCAL_D-1];
    reg signed [DATA_W-1:0] input_value  [0:LOCAL_D-1];   // host-written; stays fabric
    reg                     input_valid_r;
    // RUNTIME state -> BRAM: cleared 0..LOCAL_D-1 on reset (BRAM has no bulk
    // reset) before any tick may start -- preserves the soft-reset "clear
    // runtime state" contract. Engine is the ONLY writer -> no host race.
    (* ram_style = "block" *) reg signed [DATA_W-1:0] vmem        [0:LOCAL_D-1];
    (* ram_style = "block" *) reg        [REF_W-1:0]  refrac_count[0:LOCAL_D-1];
    reg              clr_active;
    reg [LD_W-1:0]   clr_idx;
    // SINGLE write port for the BRAM state (2026-07-29 fix): BRAM inference
    // requires ONE write statement per process. The clear walk and the stage-C
    // writeback both funnel through these combinational temps -> one nonblocking
    // write at the end of the tick block. Without this, "RAM has multiple writes
    // via different ports in same process" [Synth 8-4767] forced vmem/refrac_count
    // back into FFs (LUT/FF blew up instead of dropping).
    reg              nst_we;
    reg [LD_W-1:0]   nst_waddr;
    reg signed [DATA_W-1:0] nst_vmem;
    reg        [REF_W-1:0]  nst_rc;
`else
    // ---- per-neuron config storage (original fabric storage) ----
    reg signed [DATA_W-1:0] threshold    [0:LOCAL_D-1];
    reg        [DATA_W-1:0] leak         [0:LOCAL_D-1];
    reg signed [DATA_W-1:0] reset_state  [0:LOCAL_D-1];
    reg        [REF_W-1:0]  refrac_period[0:LOCAL_D-1];
    reg                     input_enable [0:LOCAL_D-1];
    reg signed [DATA_W-1:0] input_value  [0:LOCAL_D-1];   // per-tick external stimulus
    reg                     input_valid_r;                // latched at start for the whole tick

    // ---- per-neuron runtime state (persists across ticks) ----
    reg signed [DATA_W-1:0] vmem        [0:LOCAL_D-1];
    reg        [REF_W-1:0]  refrac_count[0:LOCAL_D-1];
`endif

    genvar gv;
    generate
`ifdef SNM_NEURON_STATE_BRAM
        // vmem is BLOCK RAM now: a full combinational fan-out read of every entry
        // (this debug/verify port) would defeat BRAM inference and build a giant
        // LOCAL_D:1 read mux. vmem_out_flat is UNCONNECTED in snm_infer_engine_stdp
        // (see that file's own "deliberately left unconnected" comment), so tie it
        // off here -- no functional path reads it. A real membrane-readback port
        // would add a proper synchronous BRAM read address, not this flat view.
        assign vmem_out_flat = {(LOCAL_D*DATA_W){1'b0}};
`else
        for (gv = 0; gv < LOCAL_D; gv = gv + 1) begin : g_vmem_out
            assign vmem_out_flat[gv*DATA_W +: DATA_W] = vmem[gv];
        end
`endif
    endgenerate

    localparam [2:0] PF_THR=3'd0, PF_LEAK=3'd1, PF_RST=3'd2, PF_RP=3'd3,
                     PF_IEN=3'd4, PF_ALL=3'd7;
    // input_value[] IS reset here (2026-07-25 soft-reset support): it is
    // per-tick RUNTIME stimulus, not config, and its persistence across
    // separate host runs is exactly what caused the hardware inter-run
    // divergence (a neuron left with a stale teacher current kept firing).
    // Resetting it lets OP_ENGINE_RESET (which drops reset_n for a cycle via
    // engine_reset_n) clear it without a bitstream reload. The CONFIG params
    // (threshold/leak/reset_state/refrac_period) are deliberately NOT reset
    // here -- they are preserved across a soft-reset (the host reloads them if
    // it wants to; a soft-reset is "clear runtime state", not "wipe config").
    // This stays a SINGLE-driver block for every array it touches (the
    // multi-driver hazard the input_enable[] comment below documents is
    // avoided: input_value's only other writer is this same block's in_we
    // path). integer cfg_k for the reset loop.
    integer cfg_k;
    always @(posedge clk) begin
`ifdef SNM_NEURON_STATE_BRAM
        // Config params -> BRAM: simple indexed writes, no reset (config survives
        // soft-reset). input_value STAYS in fabric with its original atomic
        // 1-cycle bulk reset + in_we write (no clear-FSM involvement -> no race
        // with the host's post-reset rewrite; see the declaration comment).
        if (cfg_param_we) begin
            if (cfg_param_field == PF_THR  || cfg_param_field == PF_ALL)
                threshold[cfg_param_idx]     <= cfg_param_threshold;
            if (cfg_param_field == PF_LEAK || cfg_param_field == PF_ALL)
                leak[cfg_param_idx]          <= cfg_param_leak;
            if (cfg_param_field == PF_RST  || cfg_param_field == PF_ALL)
                reset_state[cfg_param_idx]   <= cfg_param_reset_state;
            if (cfg_param_field == PF_RP   || cfg_param_field == PF_ALL)
                refrac_period[cfg_param_idx] <= cfg_param_refrac_period;
        end
        if (!reset_n) begin
            for (cfg_k = 0; cfg_k < LOCAL_D; cfg_k = cfg_k + 1)
                input_value[cfg_k] <= {DATA_W{1'b0}};
        end else if (in_we) begin
            input_value[in_idx] <= in_value;
        end
`else
        if (!reset_n) begin
            for (cfg_k = 0; cfg_k < LOCAL_D; cfg_k = cfg_k + 1)
                input_value[cfg_k] <= {DATA_W{1'b0}};
        end else begin
            if (cfg_param_we) begin
                if (cfg_param_field == PF_THR  || cfg_param_field == PF_ALL)
                    threshold[cfg_param_idx]     <= cfg_param_threshold;
                if (cfg_param_field == PF_LEAK || cfg_param_field == PF_ALL)
                    leak[cfg_param_idx]          <= cfg_param_leak;
                if (cfg_param_field == PF_RST  || cfg_param_field == PF_ALL)
                    reset_state[cfg_param_idx]   <= cfg_param_reset_state;
                if (cfg_param_field == PF_RP   || cfg_param_field == PF_ALL)
                    refrac_period[cfg_param_idx] <= cfg_param_refrac_period;
            end
            if (in_we) input_value[in_idx] <= in_value;
        end
`endif
    end

    // input_enable[] gets its OWN write-only always block (2026-07-13 hardware
    // bring-up fix), rather than sharing an array with two different always
    // blocks (this one's cfg_param_we write, and the tick-logic block's
    // reset-time zeroing below). Two always blocks driving the same reg array
    // is a multi-driver hazard: iverilog silently tolerated it (all offline sims
    // passed), but real ZCU104 hardware bring-up (first-ever silicon run of this
    // engine, 2026-07-13) showed input_enable writes NEVER took effect -- a
    // neuron configured input_enable=1 and fed external current never fired,
    // while threshold/leak/reset_state/refrac_period (each single-driver) all
    // worked correctly. Root cause: Vivado's synthesis of the array resolved the
    // two-always-block conflict in favor of the reset block, silently dropping
    // the config-write path. Fix: single always block owns input_enable[] end to
    // end.
    //
    // 2026-07-27 soft-reset correctness fix: this block ORIGINALLY also reset
    // input_enable[] to 0 on !reset_n, which was harmless when written (only
    // fired at power-up, before OP_ENGINE_RESET existed) but became a real bug
    // once soft-reset was added (2026-07-25): OP_ENGINE_RESET pulses this SAME
    // reset_n (via engine_reset_n), so every soft_reset() call silently
    // disabled external input on every neuron until the host explicitly
    // reconfigured input_enable again -- discovered via a real Basys3 hardware
    // test where an inference tick right after soft_reset() never fired any
    // neuron despite correct weights/thresholds/injected current. input_enable
    // is CONFIG (the host sets it once via configure_neuron()), exactly like
    // threshold/leak/reset_state/refrac_period just above, which never had a
    // reset branch here for the same reason: config survives a soft-reset, the
    // host reloads it explicitly only if it wants to change it. Removing the
    // reset branch (keeping this a single-driver, write-only block) restores
    // that same "config-survives-soft-reset" contract for input_enable without
    // reintroducing the 2026-07-13 multi-driver hazard.
    always @(posedge clk) begin
        if (cfg_param_we &&
            (cfg_param_field == PF_IEN || cfg_param_field == PF_ALL)) begin
            input_enable[cfg_param_idx] <= cfg_param_input_enable;
        end
    end

    // ---- membrane math, identical semantics to spikeengine_core.v ----
    function automatic signed [DATA_W-1:0] leak_toward_reset;
        input signed [DATA_W-1:0] vmem_in;
        input signed [DATA_W-1:0] reset_in;
        input [DATA_W-1:0] leak_in;
        reg signed [DATA_W:0] candidate, vmem_ext, reset_ext, leak_ext;
        begin
            vmem_ext  = {vmem_in[DATA_W-1], vmem_in};
            reset_ext = {reset_in[DATA_W-1], reset_in};
            leak_ext  = {1'b0, leak_in};
            if (vmem_ext > reset_ext) begin
                candidate = vmem_ext - leak_ext;
                leak_toward_reset = (candidate < reset_ext) ? reset_in : candidate[DATA_W-1:0];
            end else if (vmem_ext < reset_ext) begin
                candidate = vmem_ext + leak_ext;
                leak_toward_reset = (candidate > reset_ext) ? reset_in : candidate[DATA_W-1:0];
            end else begin
                leak_toward_reset = reset_in;
            end
        end
    endfunction

    function automatic signed [DATA_W-1:0] saturating_fold;
        input signed [DATA_W-1:0]  base;
        input signed [ACCUM_W-1:0] accum;
        reg signed [ACCUM_W:0] sum, max_value, min_value;
        begin
            sum = {{(ACCUM_W+1-DATA_W){base[DATA_W-1]}}, base} + {accum[ACCUM_W-1], accum};
            max_value = {{(ACCUM_W+2-DATA_W){1'b0}}, {(DATA_W-1){1'b1}}};
            min_value = {{(ACCUM_W+2-DATA_W){1'b1}}, {(DATA_W-1){1'b0}}};
            if (sum > max_value) saturating_fold = {1'b0, {(DATA_W-1){1'b1}}};
            else if (sum < min_value) saturating_fold = {1'b1, {(DATA_W-1){1'b0}}};
            else saturating_fold = sum[DATA_W-1:0];
        end
    endfunction

    // Same-width saturating add (core's saturating_add) -- used for external input.
    function automatic signed [DATA_W-1:0] saturating_add;
        input signed [DATA_W-1:0] a;
        input signed [DATA_W-1:0] b;
        reg signed [DATA_W:0] sum, max_value, min_value;
        begin
            sum = {a[DATA_W-1], a} + {b[DATA_W-1], b};
            max_value = {2'b00, {(DATA_W-1){1'b1}}};
            min_value = {2'b11, {(DATA_W-1){1'b0}}};
            if (sum > max_value) saturating_add = {1'b0, {(DATA_W-1){1'b1}}};
            else if (sum < min_value) saturating_add = {1'b1, {(DATA_W-1){1'b0}}};
            else saturating_add = sum[DATA_W-1:0];
        end
    endfunction

    // stage-A base: leak toward reset, then (if this neuron accepts input AND the
    // tick carries valid input) saturating-add the external stimulus -- exactly the
    // core's `tick_base = (input_valid && input_enable) ? saturating_add(leaked,
    // input_value) : leaked`. Kept in stage A alongside leak (two DATA_W adds);
    // if a future timing report flags stage A as the limiter, split the input-add
    // into its own stage as the core does (ST_TICK_UPDATE_LEAK vs ACCUM_REQ).
    function automatic signed [DATA_W-1:0] leak_and_input;
        input signed [DATA_W-1:0] vmem_in, reset_in, in_val;
        input [DATA_W-1:0]        leak_in;
        input                     in_en, in_vld;
        reg signed [DATA_W-1:0]   leaked;
        begin
            leaked = leak_toward_reset(vmem_in, reset_in, leak_in);
            leak_and_input = (in_vld && in_en) ? saturating_add(leaked, in_val) : leaked;
        end
    endfunction

`ifdef SNM_INFER_NEURON_PIPE_5STAGE
    // ---- 5-stage pipeline registers (2026-07-14: stage A1 split into A1a/A1b) ----
    // The 4-stage split (A0 raw-read / A1 leak_and_input-compute) fixed Basys3's
    // ORIGINAL bottleneck (combined state-read+leak, 14 logic levels) but the
    // N/K capacity sweep (2026-07-14) found the REMAINING worst path is entirely
    // WITHIN stage A1's own leak_and_input() call: leak_toward_reset() (compare +
    // 17-bit add/sub + compare + mux, ~8-10 levels) feeds DIRECTLY into
    // saturating_add() (17-bit add + compare + mux, ~6-8 levels) in the SAME
    // cycle -- 14-17 logic levels combined, MEASURED IDENTICAL at both N=16 and
    // N=128 (fixed-width per-neuron arithmetic, doesn't scale with N or K at
    // all), sitting right at the ~10ns/100MHz wall regardless of scale. This
    // registers leak_toward_reset()'s result between the two functions (stage
    // A1a computes leak_toward_reset only; stage A1b computes the input
    // saturating_add on the now-REGISTERED A1a output), roughly halving the
    // combinational depth of the worst remaining single-cycle chain. Stages B
    // and C are UNCHANGED -- A1b's outputs are named identically to the old
    // stage A1's outputs (leaked_ab etc) so nothing downstream differs.
    // Selected per-board via SNM_INFER_NEURON_PIPE_5STAGE (gen_config.py, from
    // boards.yaml's infer_neuron_pipe_stages: 5 field) -- a per-board
    // timing-closure tunable, same pattern as SNM_INFER_NEURON_PIPE_4STAGE.
    reg                      v_r0;
    reg [LD_W-1:0]           d_r0;
    reg signed [DATA_W-1:0]  vmem_r0, rst_r0;
    reg [DATA_W-1:0]         leak_r0;
    reg signed [DATA_W-1:0]  inval_r0;
    reg                      inen_r0;
    reg signed [ACCUM_W-1:0] accum_r0;
    reg signed [DATA_W-1:0]  thr_r0;
    reg [REF_W-1:0]          rp_r0, rc_r0;
    // stage A0 -> A1a: raw fields carried forward
    reg                      v_r1;
    reg [LD_W-1:0]           d_r1;
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  leaked_r1;
    reg signed [DATA_W-1:0]  rst_r1;
    reg signed [DATA_W-1:0]  inval_r1;
    reg                      inen_r1;
    reg signed [ACCUM_W-1:0] accum_r1;
    reg signed [DATA_W-1:0]  thr_r1;
    reg [REF_W-1:0]          rp_r1, rc_r1;
    // stage A1b -> B: leaked+input vmem + selected accumulator + params carried forward
    reg                      v_ab;
    reg [LD_W-1:0]           d_ab;
    // (* use_dsp = "no" *): future-ASIC target -- leaked_ab/final_bc carry
    // this module's leak/saturating-add/saturating-fold results, the widest
    // arithmetic here (up to ACCUM_W+1 bits in saturating_fold). No real
    // multiply anywhere in this file (leak is an additive decay per
    // superneuromat's own semantics, not multiplicative), so this is
    // belt-and-suspenders against Vivado folding a plain add into a DSP48
    // ALU -- keeps the mapped gates a plain adder/carry-chain on any target.
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  leaked_ab;
    reg signed [ACCUM_W-1:0] accum_ab;
    reg signed [DATA_W-1:0]  thr_ab, rst_ab;
    reg [REF_W-1:0]          rp_ab, rc_ab;
    // stage B -> C: folded membrane + params carried forward
    reg                      v_bc;
    reg [LD_W-1:0]           d_bc;
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  final_bc;
    reg signed [DATA_W-1:0]  thr_bc, rst_bc;
    reg [REF_W-1:0]          rp_bc, rc_bc;

    reg [LD_W-1:0] issue_d;
    reg running;

    integer k;
    // Synchronous reset (2026-07-24, future-ASIC methodology fix -- see
    // snm_gather_lane_stdp.v's own always-block comment for the full
    // rationale: avoids a large async reset fan-out over vmem[]/threshold[]/
    // leak[]/etc, all LOCAL_D-wide arrays. Same reset values/logic, applied
    // synchronously instead of immediately; transparent to every testbench,
    // which already holds reset_n low across multiple clock edges).
    always @(posedge clk) begin
`ifdef SNM_NEURON_STATE_BRAM
        // default: no BRAM-state write this cycle (blocking temps; single
        // nonblocking write is at the end of the block -- see decl comment).
        nst_we    = 1'b0;
        nst_waddr = {LD_W{1'b0}};
        nst_vmem  = {DATA_W{1'b0}};
        nst_rc    = {REF_W{1'b0}};
`endif
        if (!reset_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            running <= 1'b0;
            issue_d <= {LD_W{1'b0}};
            v_r0 <= 1'b0;
            v_r1 <= 1'b0;
            v_ab <= 1'b0;
            v_bc <= 1'b0;
            d_r0 <= {LD_W{1'b0}};
            d_r1 <= {LD_W{1'b0}};
            d_ab <= {LD_W{1'b0}};
            d_bc <= {LD_W{1'b0}};
            vmem_r0 <= {DATA_W{1'b0}};
            rst_r0 <= {DATA_W{1'b0}};
            leak_r0 <= {DATA_W{1'b0}};
            inval_r0 <= {DATA_W{1'b0}};
            inen_r0 <= 1'b0;
            accum_r0 <= {ACCUM_W{1'b0}};
            thr_r0 <= {DATA_W{1'b0}};
            rp_r0 <= {REF_W{1'b0}};
            rc_r0 <= {REF_W{1'b0}};
            leaked_r1 <= {DATA_W{1'b0}};
            rst_r1 <= {DATA_W{1'b0}};
            inval_r1 <= {DATA_W{1'b0}};
            inen_r1 <= 1'b0;
            accum_r1 <= {ACCUM_W{1'b0}};
            thr_r1 <= {DATA_W{1'b0}};
            rp_r1 <= {REF_W{1'b0}};
            rc_r1 <= {REF_W{1'b0}};
            input_valid_r <= 1'b0;
            leaked_ab <= {DATA_W{1'b0}};
            accum_ab <= {ACCUM_W{1'b0}};
            thr_ab <= {DATA_W{1'b0}};
            rst_ab <= {DATA_W{1'b0}};
            rp_ab <= {REF_W{1'b0}};
            rc_ab <= {REF_W{1'b0}};
            final_bc <= {DATA_W{1'b0}};
            thr_bc <= {DATA_W{1'b0}};
            rst_bc <= {DATA_W{1'b0}};
            rp_bc <= {REF_W{1'b0}};
            rc_bc <= {REF_W{1'b0}};
            spike_out <= {LOCAL_D{1'b0}};
`ifdef SNM_NEURON_STATE_BRAM
            // vmem/refrac_count are BRAM: cannot bulk-reset. Arm the clear walk;
            // it zeroes them (and input_value in the cfg block) sequentially
            // before any tick can start.
            clr_active <= 1'b1;
            clr_idx    <= {LD_W{1'b0}};
`else
            for (k = 0; k < LOCAL_D; k = k + 1) begin
                vmem[k] <= {DATA_W{1'b0}};
                refrac_count[k] <= {REF_W{1'b0}};
            end
`endif
        end else begin
            done <= 1'b0;

`ifdef SNM_NEURON_STATE_BRAM
            // ---- post-reset clear walk: zero one runtime-state address/cycle
            // (vmem/refrac_count here; input_value in the cfg block reads the
            // same clr_idx). Blocks ticks until complete. ----
            if (clr_active) begin
                nst_we    = 1'b1;              // -> single write port (zero this addr)
                nst_waddr = clr_idx;
                nst_vmem  = {DATA_W{1'b0}};
                nst_rc    = {REF_W{1'b0}};
                if (clr_idx == LOCAL_D[LD_W-1:0] - 1'b1)
                    clr_active <= 1'b0;
                clr_idx <= clr_idx + 1'b1;
            end
`endif

            if (!running
`ifdef SNM_NEURON_STATE_BRAM
                && !clr_active
`endif
               ) begin
                busy <= 1'b0;
                v_r0 <= 1'b0;
                v_r1 <= 1'b0;
                v_ab <= 1'b0;
                v_bc <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    running <= 1'b1;
                    issue_d <= {LD_W{1'b0}};
                    input_valid_r <= input_valid;   // freeze input gate for the tick
                end
            end else if (running) begin
                // ---- stage C: decide + writeback (consumes B->C regs) ----
                if (v_bc) begin
`ifdef SNM_NEURON_STATE_BRAM
                    // route vmem/refrac_count writeback through the single write
                    // port (nst_*); spike_out/running/done unchanged.
                    nst_we    = 1'b1;
                    nst_waddr = d_bc;
                    if (rc_bc != {REF_W{1'b0}}) begin
                        nst_rc   = rc_bc - 1'b1;
                        nst_vmem = final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end else if (final_bc > thr_bc) begin
                        nst_rc   = rp_bc;
                        nst_vmem = rst_bc;
                        spike_out[d_bc] <= 1'b1;
                    end else begin
                        nst_rc   = {REF_W{1'b0}};
                        nst_vmem = final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end
`else
                    if (rc_bc != {REF_W{1'b0}}) begin
                        // refractory: suppress spike, still integrate (core semantics)
                        refrac_count[d_bc] <= rc_bc - 1'b1;
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end else if (final_bc > thr_bc) begin
                        refrac_count[d_bc] <= rp_bc;
                        vmem[d_bc] <= rst_bc;
                        spike_out[d_bc] <= 1'b1;
                    end else begin
                        refrac_count[d_bc] <= {REF_W{1'b0}};
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end
`endif
                    // last neuron written back -> tick complete
                    if (d_bc == LOCAL_D - 1) begin
                        running <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end

                // ---- stage B: saturating fold (A1b->B regs in, B->C regs out) ----
                v_bc <= v_ab;
                d_bc <= d_ab;
                final_bc <= saturating_fold(leaked_ab, accum_ab);
                thr_bc <= thr_ab;
                rst_bc <= rst_ab;
                rp_bc <= rp_ab;
                rc_bc <= rc_ab;

                // ---- stage A1b: input saturating-add (A1a->A1b regs in, A1b->B
                //      regs out). leaked_ab now carries the "base" = leaked vmem
                //      with external stimulus applied (core's tick_base). ----
                v_ab <= v_r1;
                d_ab <= d_r1;
                leaked_ab <= (inen_r1 && input_valid_r) ? saturating_add(leaked_r1, inval_r1) : leaked_r1;
                accum_ab <= accum_r1;
                thr_ab <= thr_r1;
                rst_ab <= rst_r1;
                rp_ab <= rp_r1;
                rc_ab <= rc_r1;

                // ---- stage A1a: leak compute ONLY (A0->A1a regs in, A1a->A1b
                //      regs out). ----
                v_r1 <= v_r0;
                d_r1 <= d_r0;
                leaked_r1 <= leak_toward_reset(vmem_r0, rst_r0, leak_r0);
                rst_r1 <= rst_r0;
                inval_r1 <= inval_r0;
                inen_r1 <= inen_r0;
                accum_r1 <= accum_r0;
                thr_r1 <= thr_r0;
                rp_r1 <= rp_r0;
                rc_r1 <= rc_r0;

                // ---- stage A0: RAW state/param read only, no computation
                //      (issues one neuron/cycle). ----
                if (issue_d < LOCAL_D[LD_W-1:0]) begin
                    v_r0 <= 1'b1;
                    d_r0 <= issue_d;
                    vmem_r0  <= vmem[issue_d];
                    rst_r0   <= reset_state[issue_d];
                    leak_r0  <= leak[issue_d];
                    inval_r0 <= input_value[issue_d];
                    inen_r0  <= input_enable[issue_d];
                    accum_r0 <= accum_in_flat[issue_d*ACCUM_W +: ACCUM_W];
                    thr_r0   <= threshold[issue_d];
                    rp_r0    <= refrac_period[issue_d];
                    rc_r0    <= refrac_count[issue_d];
                    issue_d <= issue_d + 1'b1;
                end else begin
                    v_r0 <= 1'b0;
                end
            end

`ifdef SNM_NEURON_STATE_BRAM
            // ---- the ONE write statement for the BRAM runtime state. Both the
            // clear walk and stage C funnel through nst_* above; this is the
            // single write port BRAM inference requires (fixes [Synth 8-4767]
            // "multiple writes via different ports"). Lives at the non-reset-else
            // level so it fires for clear (running=0) and stage C (running=1). ----
            if (nst_we) begin
                vmem[nst_waddr]         <= nst_vmem;
                refrac_count[nst_waddr] <= nst_rc;
            end
`endif
        end
    end
`elsif SNM_INFER_NEURON_PIPE_4STAGE
    // ---- 4-stage pipeline registers (2026-07-13/14: stage A split into A0/A1) ----
    // Basys3's Artix-7-1 fabric (the slowest of the three boards) measured
    // stage A (state-read + leak_toward_reset, combined in one cycle) as the
    // worst path: 14 logic levels (CARRY4x7 + LUTs), WNS -0.891ns -- the SAME
    // class of long-combinational-chain issue this module's ORIGINAL 3-stage
    // split already fixed once (see the module-header comment), one notch
    // deeper than this board's fabric can absorb in a single cycle where
    // ZCU104/SP701 can (at 3 stages). Splitting stage A into A0 (raw array
    // read only, no computation) and A1 (leak_and_input compute on the now-
    // REGISTERED A0 outputs) halves that cycle's combinational depth. Stages
    // B and C are UNCHANGED -- A1's outputs are named identically to the old
    // stage A's outputs so nothing downstream differs between the two modes.
    // Selected per-board via SNM_INFER_NEURON_PIPE_4STAGE (gen_config.py, from
    // boards.yaml's infer_neuron_pipe_stages field) -- NOT a functional choice,
    // a per-board timing-closure tunable, same pattern as SNM_SYN_MEM_ULTRA.
    reg                      v_r0;
    reg [LD_W-1:0]           d_r0;
    reg signed [DATA_W-1:0]  vmem_r0, rst_r0;
    reg [DATA_W-1:0]         leak_r0;
    reg signed [DATA_W-1:0]  inval_r0;
    reg                      inen_r0;
    reg signed [ACCUM_W-1:0] accum_r0;
    reg signed [DATA_W-1:0]  thr_r0;
    reg [REF_W-1:0]          rp_r0, rc_r0;
    // stage A1 -> B: leaked vmem + selected accumulator + params carried forward
    reg                      v_ab;
    reg [LD_W-1:0]           d_ab;
    // (* use_dsp = "no" *): future-ASIC target -- leaked_ab/final_bc carry
    // this module's leak/saturating-add/saturating-fold results, the widest
    // arithmetic here (up to ACCUM_W+1 bits in saturating_fold). No real
    // multiply anywhere in this file (leak is an additive decay per
    // superneuromat's own semantics, not multiplicative), so this is
    // belt-and-suspenders against Vivado folding a plain add into a DSP48
    // ALU -- keeps the mapped gates a plain adder/carry-chain on any target.
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  leaked_ab;
    reg signed [ACCUM_W-1:0] accum_ab;
    reg signed [DATA_W-1:0]  thr_ab, rst_ab;
    reg [REF_W-1:0]          rp_ab, rc_ab;
    // stage B -> C: folded membrane + params carried forward
    reg                      v_bc;
    reg [LD_W-1:0]           d_bc;
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  final_bc;
    reg signed [DATA_W-1:0]  thr_bc, rst_bc;
    reg [REF_W-1:0]          rp_bc, rc_bc;

    reg [LD_W-1:0] issue_d;
    reg running;

    integer k;
    // Synchronous reset (2026-07-24, future-ASIC methodology fix -- see
    // snm_gather_lane_stdp.v's own always-block comment for the full
    // rationale: avoids a large async reset fan-out over vmem[]/threshold[]/
    // leak[]/etc, all LOCAL_D-wide arrays. Same reset values/logic, applied
    // synchronously instead of immediately; transparent to every testbench,
    // which already holds reset_n low across multiple clock edges).
    always @(posedge clk) begin
`ifdef SNM_NEURON_STATE_BRAM
        // default: no BRAM-state write this cycle (blocking temps; single
        // nonblocking write is at the end of the block -- see decl comment).
        nst_we    = 1'b0;
        nst_waddr = {LD_W{1'b0}};
        nst_vmem  = {DATA_W{1'b0}};
        nst_rc    = {REF_W{1'b0}};
`endif
        if (!reset_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            running <= 1'b0;
            issue_d <= {LD_W{1'b0}};
            v_r0 <= 1'b0;
            v_ab <= 1'b0;
            v_bc <= 1'b0;
            d_r0 <= {LD_W{1'b0}};
            d_ab <= {LD_W{1'b0}};
            d_bc <= {LD_W{1'b0}};
            vmem_r0 <= {DATA_W{1'b0}};
            rst_r0 <= {DATA_W{1'b0}};
            leak_r0 <= {DATA_W{1'b0}};
            inval_r0 <= {DATA_W{1'b0}};
            inen_r0 <= 1'b0;
            accum_r0 <= {ACCUM_W{1'b0}};
            thr_r0 <= {DATA_W{1'b0}};
            rp_r0 <= {REF_W{1'b0}};
            rc_r0 <= {REF_W{1'b0}};
            input_valid_r <= 1'b0;
            leaked_ab <= {DATA_W{1'b0}};
            accum_ab <= {ACCUM_W{1'b0}};
            thr_ab <= {DATA_W{1'b0}};
            rst_ab <= {DATA_W{1'b0}};
            rp_ab <= {REF_W{1'b0}};
            rc_ab <= {REF_W{1'b0}};
            final_bc <= {DATA_W{1'b0}};
            thr_bc <= {DATA_W{1'b0}};
            rst_bc <= {DATA_W{1'b0}};
            rp_bc <= {REF_W{1'b0}};
            rc_bc <= {REF_W{1'b0}};
            spike_out <= {LOCAL_D{1'b0}};
            for (k = 0; k < LOCAL_D; k = k + 1) begin
                vmem[k] <= {DATA_W{1'b0}};
                refrac_count[k] <= {REF_W{1'b0}};
            end
        end else begin
            done <= 1'b0;

            if (!running) begin
                busy <= 1'b0;
                v_r0 <= 1'b0;
                v_ab <= 1'b0;
                v_bc <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    running <= 1'b1;
                    issue_d <= {LD_W{1'b0}};
                    input_valid_r <= input_valid;   // freeze input gate for the tick
                end
            end else begin
                // ---- stage C: decide + writeback (consumes B->C regs) ----
                if (v_bc) begin
                    if (rc_bc != {REF_W{1'b0}}) begin
                        // refractory: suppress spike, still integrate (core semantics)
                        refrac_count[d_bc] <= rc_bc - 1'b1;
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end else if (final_bc > thr_bc) begin
                        refrac_count[d_bc] <= rp_bc;
                        vmem[d_bc] <= rst_bc;
                        spike_out[d_bc] <= 1'b1;
                    end else begin
                        refrac_count[d_bc] <= {REF_W{1'b0}};
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end
                    // last neuron written back -> tick complete
                    if (d_bc == LOCAL_D - 1) begin
                        running <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end

                // ---- stage B: saturating fold (A1->B regs in, B->C regs out) ----
                v_bc <= v_ab;
                d_bc <= d_ab;
                final_bc <= saturating_fold(leaked_ab, accum_ab);
                thr_bc <= thr_ab;
                rst_bc <= rst_ab;
                rp_bc <= rp_ab;
                rc_bc <= rc_ab;

                // ---- stage A1: leak + external input compute (A0->A1 regs in,
                //      A1->B regs out). leaked_ab now carries the "base" = leaked
                //      vmem with external stimulus applied (core's tick_base). ----
                v_ab <= v_r0;
                d_ab <= d_r0;
                leaked_ab <= leak_and_input(vmem_r0, rst_r0, inval_r0, leak_r0,
                                            inen_r0, input_valid_r);
                accum_ab <= accum_r0;
                thr_ab <= thr_r0;
                rst_ab <= rst_r0;
                rp_ab <= rp_r0;
                rc_ab <= rc_r0;

                // ---- stage A0: RAW state/param read only, no computation
                //      (issues one neuron/cycle). ----
                if (issue_d < LOCAL_D[LD_W-1:0]) begin
                    v_r0 <= 1'b1;
                    d_r0 <= issue_d;
                    vmem_r0  <= vmem[issue_d];
                    rst_r0   <= reset_state[issue_d];
                    leak_r0  <= leak[issue_d];
                    inval_r0 <= input_value[issue_d];
                    inen_r0  <= input_enable[issue_d];
                    accum_r0 <= accum_in_flat[issue_d*ACCUM_W +: ACCUM_W];
                    thr_r0   <= threshold[issue_d];
                    rp_r0    <= refrac_period[issue_d];
                    rc_r0    <= refrac_count[issue_d];
                    issue_d <= issue_d + 1'b1;
                end else begin
                    v_r0 <= 1'b0;
                end
            end
        end
    end
`else
    // ---- 3-stage pipeline registers ----
    // stage A -> B: leaked vmem + selected accumulator + params carried forward
    reg                      v_ab;
    reg [LD_W-1:0]           d_ab;
    // (* use_dsp = "no" *): future-ASIC target -- leaked_ab/final_bc carry
    // this module's leak/saturating-add/saturating-fold results, the widest
    // arithmetic here (up to ACCUM_W+1 bits in saturating_fold). No real
    // multiply anywhere in this file (leak is an additive decay per
    // superneuromat's own semantics, not multiplicative), so this is
    // belt-and-suspenders against Vivado folding a plain add into a DSP48
    // ALU -- keeps the mapped gates a plain adder/carry-chain on any target.
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  leaked_ab;
    reg signed [ACCUM_W-1:0] accum_ab;
    reg signed [DATA_W-1:0]  thr_ab, rst_ab;
    reg [REF_W-1:0]          rp_ab, rc_ab;
    // stage B -> C: folded membrane + params carried forward
    reg                      v_bc;
    reg [LD_W-1:0]           d_bc;
    (* use_dsp = "no" *) reg signed [DATA_W-1:0]  final_bc;
    reg signed [DATA_W-1:0]  thr_bc, rst_bc;
    reg [REF_W-1:0]          rp_bc, rc_bc;

    reg [LD_W-1:0] issue_d;
    reg running;

    integer k;
    // Synchronous reset (2026-07-24, future-ASIC methodology fix -- see
    // snm_gather_lane_stdp.v's own always-block comment for the full
    // rationale: avoids a large async reset fan-out over vmem[]/threshold[]/
    // leak[]/etc, all LOCAL_D-wide arrays. Same reset values/logic, applied
    // synchronously instead of immediately; transparent to every testbench,
    // which already holds reset_n low across multiple clock edges).
    always @(posedge clk) begin
`ifdef SNM_NEURON_STATE_BRAM
        // default: no BRAM-state write this cycle (blocking temps; single
        // nonblocking write is at the end of the block -- see decl comment).
        nst_we    = 1'b0;
        nst_waddr = {LD_W{1'b0}};
        nst_vmem  = {DATA_W{1'b0}};
        nst_rc    = {REF_W{1'b0}};
`endif
        if (!reset_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            running <= 1'b0;
            issue_d <= {LD_W{1'b0}};
            v_ab <= 1'b0;
            v_bc <= 1'b0;
            d_ab <= {LD_W{1'b0}};
            d_bc <= {LD_W{1'b0}};
            input_valid_r <= 1'b0;
            leaked_ab <= {DATA_W{1'b0}};
            accum_ab <= {ACCUM_W{1'b0}};
            thr_ab <= {DATA_W{1'b0}};
            rst_ab <= {DATA_W{1'b0}};
            rp_ab <= {REF_W{1'b0}};
            rc_ab <= {REF_W{1'b0}};
            final_bc <= {DATA_W{1'b0}};
            thr_bc <= {DATA_W{1'b0}};
            rst_bc <= {DATA_W{1'b0}};
            rp_bc <= {REF_W{1'b0}};
            rc_bc <= {REF_W{1'b0}};
            spike_out <= {LOCAL_D{1'b0}};
            for (k = 0; k < LOCAL_D; k = k + 1) begin
                vmem[k] <= {DATA_W{1'b0}};
                refrac_count[k] <= {REF_W{1'b0}};
                // input_enable's reset moved OUT of this block (2026-07-13, see the
                // dedicated always block above) -- having it here AND in the
                // cfg_param_we write block made it a multi-driver reg array, which
                // real Vivado synthesis resolved by silently dropping the write path
                // (iverilog sim never caught this). Do not re-add it here.
            end
        end else begin
            done <= 1'b0;

            if (!running) begin
                busy <= 1'b0;
                v_ab <= 1'b0;
                v_bc <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    running <= 1'b1;
                    issue_d <= {LD_W{1'b0}};
                    input_valid_r <= input_valid;   // freeze input gate for the tick
                end
            end else begin
                // ---- stage C: decide + writeback (consumes B->C regs) ----
                if (v_bc) begin
                    if (rc_bc != {REF_W{1'b0}}) begin
                        // refractory: suppress spike, still integrate (core semantics)
                        refrac_count[d_bc] <= rc_bc - 1'b1;
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end else if (final_bc > thr_bc) begin
                        refrac_count[d_bc] <= rp_bc;
                        vmem[d_bc] <= rst_bc;
                        spike_out[d_bc] <= 1'b1;
                    end else begin
                        refrac_count[d_bc] <= {REF_W{1'b0}};
                        vmem[d_bc] <= final_bc;
                        spike_out[d_bc] <= 1'b0;
                    end
                    // last neuron written back -> tick complete
                    if (d_bc == LOCAL_D - 1) begin
                        running <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end
                end

                // ---- stage B: saturating fold (A->B regs in, B->C regs out) ----
                v_bc <= v_ab;
                d_bc <= d_ab;
                final_bc <= saturating_fold(leaked_ab, accum_ab);
                thr_bc <= thr_ab;
                rst_bc <= rst_ab;
                rp_bc <= rp_ab;
                rc_bc <= rc_ab;

                // ---- stage A: state/param read + leak + external input
                //      (issues one neuron/cycle). leaked_ab now carries the "base"
                //      = leaked vmem with external stimulus applied (core's tick_base). ----
                if (issue_d < LOCAL_D[LD_W-1:0]) begin
                    v_ab <= 1'b1;
                    d_ab <= issue_d;
                    leaked_ab <= leak_and_input(vmem[issue_d], reset_state[issue_d],
                                                input_value[issue_d], leak[issue_d],
                                                input_enable[issue_d], input_valid_r);
                    accum_ab <= accum_in_flat[issue_d*ACCUM_W +: ACCUM_W];
                    thr_ab <= threshold[issue_d];
                    rst_ab <= reset_state[issue_d];
                    rp_ab <= refrac_period[issue_d];
                    rc_ab <= refrac_count[issue_d];
                    issue_d <= issue_d + 1'b1;
                end else begin
                    v_ab <= 1'b0;
                end
            end
        end
    end
`endif
endmodule
