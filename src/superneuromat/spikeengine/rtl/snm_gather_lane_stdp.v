`timescale 1us/1ns

// STDP-CAPABLE variant of snm_gather_lane (2026-07-21, npu_stdp_dev experiment).
//
// This is a NEW SIBLING module, not a modification of snm_gather_lane.v -- that
// file stays byte-for-byte untouched (it is the validated, shipped inference-only
// module; an STDP-enabled build instantiates THIS module instead, an
// inference-only build keeps instantiating the original). Two modules sharing
// almost all of their logic is more duplication than an `ifdef inside one file,
// but it makes the "inference-mode builds are completely unaffected" guarantee
// trivially true by construction, rather than something that has to be verified
// by diffing generated Verilog -- worth the duplication for an experimental fork
// (see NOTES.md's fold-back plan for how this gets reconciled if it works).
//
// Everything through the ORIGINAL inference read-and-accumulate walk (config
// load, packed-row synapse storage, pipelined read, per-destination accumulate)
// is copied VERBATIM from snm_gather_lane.v -- see that file's comments for the
// reasoning behind PACK_FACTOR, the registered-read-for-BRAM-inference pattern,
// etc. New material is the STDP phase below the "==== STDP walk ====" divider.
//
// ---- STDP phase design (first pass: correctness over throughput/synthesis) ----
// Triggered by `stdp_start` AFTER the neuron-update stage has produced this
// tick's spike decisions for this lane's destinations (`spike_out_local`).
// Destination-triggered (matches this lane's CSC-by-destination memory layout
// directly, unlike the classic core's CSR-by-source walk): for each local
// destination d whose spike_out_local[d] is set, re-walk its synapse range
// [dst_ptr[d], dst_ptr[d+1]) a SECOND time (same address-generation logic as
// inference), and for each entry {src, weight}: look up whether `src` fired at
// each of STDP_WINDOW past ticks (from `src_hist_flat`, a flattened
// [STDP_WINDOW][1<<SRC_W] history the caller maintains and replicates to every
// lane -- the same "registered input, replicated" treatment already given to
// spike_frozen for inference), compute
//     delta = sum over j in [0,STDP_WINDOW) of (src_hist[j][src] ? apos[j] : aneg[j])
// (matching the exact semantics of the software reference in
// fpga_gui_app/snm_network.py's predict_weights() -- hist[j] = spikes j+1
// ticks ago; every window slot contributes EITHER apos[j] or aneg[j], not just
// the slots where src fired), saturate weight+delta to the WEIGHT_W range, and
// write the updated entry back into syn_row (read-modify-write on the packed
// row -- the other entries sharing that row must be preserved).
//
// STATUS (corrected 2026-07-31): the two limitations described below were the
// state of the FIRST draft. BOTH have since been resolved in this same file,
// and this module is synthesized in every board build and hardware-validated
// (Basys3/SP701/ZCU104, citation-graph datasets, exact software/hardware
// agreement). The paragraphs are kept because they explain WHY the current
// design looks the way it does; read them as history, not as current caveats.
//   1. THROUGHPUT -- RESOLVED. See "STDP walk (2026-07-24 II=1 PIPELINE
//      REWORK)" below: one result per cycle, not 3 cycles per synapse.
//   2. SYNTHESIS/BRAM INFERENCE -- RESOLVED. See "SHARED single BRAM read
//      port (2026-07-22 de-duplication rework)" below: the STDP walk and the
//      inference walk share ONE registered-read port, which is what Vivado
//      infers as a single true-dual-port BRAM.
// NOTE: tb_snm_gather_lane_stdp.v, referenced below, is a development
// testbench and is NOT bundled in this package; no RTL testbench ships here.
//
// ORIGINAL DRAFT NOTES (historical):
//   1. THROUGHPUT: one synapse's read-compute-writeback takes 3 cycles here,
//      not pipelined to 1/clock the way inference reads are. Get correctness
//      proven first (bit-exact vs a Python reference, see
//      tb_snm_gather_lane_stdp.v), matching this project's own established
//      practice elsewhere (the gather engine itself was proven correct before
//      being pipelined for cycle count).
//   2. SYNTHESIS/BRAM INFERENCE: the STDP walk reads syn_row via plain
//      (effectively combinational, within-cycle) array indexing, NOT the
//      registered-read template snm_gather_lane.v's own comments document as
//      the one Vivado actually infers as a single BRAM read port. Mixing that
//      with the inference walk's already-registered read on the SAME array is
//      a real synthesis risk (documented, not yet resolved at the time) --
//      which is why the draft was labelled simulation-only. RESOLVED since;
//      see the STATUS note above. The module IS synthesized and hardware-
//      validated today.
module snm_gather_lane_stdp #(
    parameter integer LOCAL_D     = 32,
    parameter integer SYN_CAP     = 4096,
    parameter integer SRC_W       = 16,
    parameter integer WEIGHT_W    = 8,
    parameter integer ACCUM_W     = 32,
    parameter integer STDP_WINDOW = 5,
    parameter integer DPTR_W      = $clog2(SYN_CAP + 1)
)(
    input  wire                     clk,
    input  wire                     reset_n,

    // ---- configuration load (identical to snm_gather_lane) ----
    input  wire                     cfg_dptr_we,
    input  wire [$clog2(LOCAL_D+1)-1:0] cfg_dptr_idx,
    input  wire [DPTR_W-1:0]        cfg_dptr_wdata,
    input  wire                     cfg_syn_we,
    input  wire [DPTR_W-1:0]        cfg_syn_idx,
    input  wire [SRC_W-1:0]         cfg_syn_src,
    input  wire signed [WEIGHT_W-1:0] cfg_syn_weight,

    // ---- synapse weight readback (2026-07-25 addition) ----
    // Host pulses cfg_syn_re with cfg_syn_idx already set (same addressing as
    // a write); the entry at that index is returned via cfg_syn_rd_src/
    // cfg_syn_rd_weight, qualified by a cfg_syn_rd_valid pulse 2 cycles later
    // (1 cycle for syn_row's own registered read latency, 1 more to unpack
    // the row into an entry -- see the read logic next to rd_row_reg below).
    // Deliberately reuses the SAME shared BRAM read port as the inference/
    // STDP walks (just a third mux leg on rd_addr_mux) instead of adding a
    // new read port -- config reads only ever happen while the engine is
    // idle (host protocol never issues OP_WRITE_CONFIG/read mid-tick), so
    // there is no time overlap with either walk to arbitrate, and the
    // BRAM-duplication bug documented at the SHARED BRAM read port comment
    // below stays avoided.
    input  wire                     cfg_syn_re,
    output reg                      cfg_syn_rd_valid,
    output reg  [SRC_W-1:0]         cfg_syn_rd_src,
    output reg  signed [WEIGHT_W-1:0] cfg_syn_rd_weight,

    // ---- STDP parameter config load (new) ----
    input  wire                       cfg_stdp_we,     // pulses once per (apos,aneg) pair
    input  wire [$clog2(STDP_WINDOW):0] cfg_stdp_idx,  // 0..STDP_WINDOW-1
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_apos,
    input  wire signed [WEIGHT_W-1:0] cfg_stdp_aneg,
    input  wire                       stdp_global_enable,
    // Effective window depth for THIS tick = min(STDP_WINDOW, tick_index),
    // supplied by the multilane wrapper's tick counter. This is
    // superneuromat's own `t = min(stdp_time_steps, len(spike_train)-1)`:
    // 0 on the very first tick (=> NO stdp update at all), then growing by one
    // per tick until the window is fully populated. Only slots j < this value
    // participate in the sum -- there is NO Aneg-padding of slots that have
    // not happened yet (the old behaviour, which did not match superneuromat).
    input  wire [7:0]                 stdp_win_eff,

    // ---- per-tick INFERENCE run (identical to snm_gather_lane) ----
    input  wire                     run_start,
    output reg                      run_busy,
    output reg                      run_done,
    input  wire [(1<<SRC_W)-1:0]    spike_frozen,
    output wire signed [LOCAL_D*ACCUM_W-1:0] accum_out_flat,
    output reg [31:0]               cycles_this_run,

    // ---- STDP phase (new) ----
    input  wire                     stdp_start,       // pulse: begin STDP walk this tick
    output reg                      stdp_busy,
    output reg                      stdp_done,
    input  wire [LOCAL_D-1:0]       spike_out_local,  // this lane's destinations that fired

    // ---- history refresh (2026-07-24: source-indexed BRAM, replaces the old
    // flattened [STDP_WINDOW][1<<SRC_W] combinationally-muxed src_hist_flat).
    // spike_out_global = ALL N_MAX neurons' fired-this-tick bits, assembled and
    // broadcast by the multilane wrapper (same "registered input, replicated to
    // every lane" treatment spike_frozen already gets). hist_refresh_start
    // pulses once per tick, after this tick's spike_out is valid and BEFORE the
    // next tick's inference/STDP phases begin (never overlaps either walk, so
    // sharing storage is safe) -- see the hist_mem_a/hist_mem_b refresh FSM
    // below for why this takes ~(1<<SRC_W) cycles instead of 1.
    input  wire [(1<<SRC_W)-1:0]    spike_out_global,
    input  wire                     hist_refresh_start,
    output reg                      hist_refresh_done,
    output reg [31:0]               stdp_cycles_this_run
);
    function integer flog2;
        input integer value;
        integer i;
        begin
            flog2 = 0;
            for (i = value; i > 1; i = i >> 1) flog2 = flog2 + 1;
        end
    endfunction

    // ---- storage (identical layout/packing to snm_gather_lane) ----
    reg [DPTR_W-1:0] dst_ptr [0:LOCAL_D];

    localparam integer ENTRY_W       = WEIGHT_W + SRC_W;
`ifdef SNM_SYN_MEM_ULTRA
    localparam integer MEM_NATIVE_W  = 72;
`else
    localparam integer MEM_NATIVE_W  = ENTRY_W;
`endif
    localparam integer PACK_FACTOR_RAW = MEM_NATIVE_W / ENTRY_W;
    localparam integer PACK_SHIFT  = flog2(PACK_FACTOR_RAW);
    localparam integer PACK_FACTOR = 1 << PACK_SHIFT;
    localparam integer ROW_W       = PACK_FACTOR * ENTRY_W;
    localparam integer NUM_ROWS    = (SYN_CAP + PACK_FACTOR - 1) / PACK_FACTOR;
    localparam integer ROW_ADDR_W  = ($clog2(NUM_ROWS) < 1) ? 1 : $clog2(NUM_ROWS);
    localparam integer PACK_SHIFT_W = (PACK_SHIFT < 1) ? 1 : PACK_SHIFT;

    // PACK_FACTOR legality guard. The history-port mapping (g_hist_ports_wide/
    // narrow below) and the 2-copies-x-2-ports = 4-read-path history datapath
    // are validated ONLY for PACK_FACTOR == 1 (plain BRAM) and == 4 (URAM).
    // PACK=2 happens to be safe by construction (entry_src's sub index
    // truncates, and PACK=2 only consumes mem_a's two ports) but is UNTESTED,
    // and any other value is unsupported. Fail loudly at elaboration rather
    // than silently mis-map. `initial $fatal` fires at time 0 in simulation
    // (Icarus/xsim) and is honoured as an elaboration assertion by Vivado.
    // synthesis translate_off
    initial begin
        if ((PACK_FACTOR != 1) && (PACK_FACTOR != 4)) begin
            $fatal(1, "snm_gather_lane_stdp: unsupported PACK_FACTOR=%0d (only 1 and 4 are validated)", PACK_FACTOR);
        end
    end
    // synthesis translate_on

`ifdef SNM_SYN_MEM_ULTRA
    (* ram_style = "ultra" *) reg [ROW_W-1:0] syn_row [0:NUM_ROWS-1];
`else
    (* ram_style = "block" *) reg [ROW_W-1:0] syn_row [0:NUM_ROWS-1];
`endif

    // ---- STDP parameter tables (small, ordinary registers -- STDP_WINDOW <= ~8) ----
    reg signed [WEIGHT_W-1:0] apos_tbl [0:STDP_WINDOW-1];
    reg signed [WEIGHT_W-1:0] aneg_tbl [0:STDP_WINDOW-1];

    // ==== source-indexed history BRAM (2026-07-24 congestion-fix redesign) ====
    // REPLACES the flat [STDP_WINDOW][1<<SRC_W] combinationally-muxed history
    // vector (src_hist_flat / src_hist_flat_rep): that design read a WIDE,
    // shared bit-vector via a variable (data-dependent) index -- a genuine
    // (1<<SRC_W):1 multiplexer, instantiated PACK_FACTOR x STDP_WINDOW times
    // per lane. At N_MAX=1024/PACK_FACTOR=4/STDP_WINDOW=5/NUM_LANES=16 that is
    // 320 independent 1024:1 muxes, measured via a real Vivado post-synth
    // report at 55% of BOTH the device's F7 AND F8 mux-tree resources -- a
    // concentrated LOCAL resource that caused real, reproducible routing
    // failures ("not legally routed", up to 9233 node overlaps at K=16; a
    // global-congestion pre-check refusal at K=20) even at only ~70% overall
    // LUT utilization. Explicitly replicating the SOURCE register (the
    // src_hist_flat_rep fix, kept working but superseded here) only split the
    // FANOUT of that mux; it did not shrink the mux WIDTH itself.
    //
    // The fix: TRANSPOSE storage from "per window-slot, all N_MAX neurons" to
    // "per neuron (address), all STDP_WINDOW history bits" -- i.e. an ordinary
    // small memory, (1<<SRC_W) deep x STDP_WINDOW wide, addressed by the
    // source neuron id. A history lookup becomes ONE normal registered memory
    // read (fixed cost, independent of N_MAX) instead of a wide combinational
    // mux. Two IDENTICAL physical copies (mem_a/mem_b) provide 2 read ports
    // each = 4 total, matching PACK_FACTOR's 4 simultaneous per-cycle lookups
    // (T_UPDATE_A1 issues addresses; T_UPDATE_A1B consumes the 1-cycle-later
    // registered result -- see the pipeline below).
    //
    // Refresh: once per tick, via hist_refresh_start (pulsed by the multilane
    // wrapper strictly AFTER this tick's spike_out is valid and BEFORE the
    // NEXT tick's inference/STDP phases begin -- never overlapping either
    // walk, so read/write timing is unambiguous, same non-overlap property
    // the old design already relied on). Sequentially walks every address:
    // reads the OLD STDP_WINDOW-bit word, shifts in spike_out_global[addr] as
    // the new bit 0, writes the result back to BOTH copies. ~(1<<SRC_W) cycles
    // (~1024 at N_MAX=1024) -- a ~1% addition to the existing ~98,829-cycle
    // dense-worst-case tick budget, paid once for eliminating the mux forest.
    // Each memory below is accessed from exactly TWO physical ports, matching
    // a true-dual-port BRAM exactly (same discipline as syn_row's own shared
    // read port further down: one combinational address mux per port, one
    // unconditional registered access per port, inside the single main
    // clocked block -- NEVER a separate always block touching the same array,
    // which is precisely what caused Vivado to duplicate syn_row earlier in
    // this project). Port A of each memory is READ-ONLY, address muxed
    // between the refresh scan and the STDP walk's first assigned PACK entry.
    // Port B is READ-OR-WRITE: a write during the refresh scan's Stage 2,
    // otherwise a read for the STDP walk's second assigned PACK entry. The
    // actual access statements live in the main clocked block below, next to
    // rd_row_reg's own read; only address/control generation lives here.
    // KNOWN REDUNDANCY, evaluated and deliberately NOT eliminated (2026-07-24):
    // every lane instantiates its OWN hist_mem_a/hist_mem_b, and because every
    // lane's refresh scan walks the identical address range from the SAME
    // broadcast spike_out_global, all NUM_LANES copies hold byte-identical
    // content at all times -- real duplicated BRAM/URAM (NUM_LANES x, e.g.
    // 16x on ZCU104). This was NOT collapsed into one shared, cross-lane
    // memory because the STDP pipeline above demands PACK_FACTOR simultaneous
    // history reads PER LANE PER CYCLE (II=1) -- at ZCU104 scale that is
    // 16 lanes x 4 reads = 64 concurrent read ports, which no realistic
    // shared SRAM/BRAM macro provides. Serving that demand from one shared
    // copy would mean arbitrating/serializing lane access, directly
    // undermining the II=1 pipelining this file just implemented (throughput
    // would drop by roughly NUM_LANES x in the worst case). Per-lane
    // replication is the standard way multi-ported access to small,
    // frequently-read data is provided when a true multi-port SRAM isn't
    // available (the same tradeoff register files/caches make) -- this is an
    // area-for-throughput tradeoff already made deliberately, not an
    // oversight, and should stay this way unless a future redesign is
    // willing to give back STDP throughput for it.
    //
    // The ONE piece of this redundancy that IS free to remove: on
    // PACK_FACTOR==1 boards (no URAM packing), only mem_a's port A is ever
    // READ (see g_hist_ports_narrow below, which ties mem_b's addresses to a
    // constant 0 and never feeds its read result to anything) -- mem_b is
    // still WRITTEN every refresh (the main clocked block below writes both
    // unconditionally) but that write result is providably never consumed by
    // any downstream logic on this path. A memory that is written but never
    // read is exactly the pattern Vivado's synthesis dead-logic elimination
    // is expected to remove entirely on its own; left as a single array
    // (rather than forked into a generate-conditional declaration) to avoid
    // duplicating this file's large, already-verified main clocked block
    // into two separate generate-wrapped copies for a synthesis-only-visible
    // saving. Confirm via a real Vivado utilization report during the build
    // phase (a non-zero hist_mem_b tile count on a PACK_FACTOR==1 board would
    // mean this optimization needs the explicit generate-conditional split
    // instead of relying on inference).
    // (* ram_style = "block" *): steer these to true dual-port BRAM (RAMB18),
    // not LUT/distributed RAM. At (1<<SRC_W) deep x STDP_WINDOW wide (e.g.
    // 1024 x 5) Vivado could infer either; distributed RAM would (a) burn LUTs
    // this design would rather keep for logic, and (b) not give a real second
    // registered read/write port, which the two-physical-ports-per-copy
    // discipline here depends on. Block RAM is the intended mapping. CONFIRM
    // against a real post-synth utilization report during the build phase
    // (expect ~2 RAMB18 per lane -> ~32 across 16 lanes on ZCU104); if the
    // tool still picks LUTRAM, escalate to an explicit XPM/primitive.
    (* ram_style = "block" *) reg [STDP_WINDOW-1:0] hist_mem_a [0:(1<<SRC_W)-1];
    (* ram_style = "block" *) reg [STDP_WINDOW-1:0] hist_mem_b [0:(1<<SRC_W)-1];

    reg [SRC_W-1:0] hist_refresh_addr;       // next address to READ (advances every active cycle)
    reg [SRC_W-1:0] hist_refresh_addr_q;     // address whose read result is valid THIS cycle (one behind hist_refresh_addr)
    reg             hist_refresh_active;     // refresh scan in progress, still issuing reads
    reg             hist_refresh_write_valid;// this cycle's port-B access is the refresh scan's writeback (else a walk read)
    // Port A read results, shared between the refresh scan and the STDP
    // walk's PACK entries 0/2 (never simultaneously active -- see the
    // combinational address muxes below). Port B read results serve PACK
    // entries 1/3 only (port B's OTHER role, refresh writeback, has no read
    // result to register).
    reg [STDP_WINDOW-1:0] hist_mem_a_porta_rdata, hist_mem_b_porta_rdata;
    reg [STDP_WINDOW-1:0] hist_mem_a_portb_rdata, hist_mem_b_portb_rdata;
    // Fixed 4-element view of the same four registers above, indexed to match
    // PACK entry number (0=mem_a portA, 1=mem_a portB, 2=mem_b portA,
    // 3=mem_b portB). Always sized [0:3] regardless of PACK_FACTOR -- unlike
    // hist_bits_p1_arr (sized [0:PACK_FACTOR-1]), so a PACK_FACTOR-bounded
    // copy loop into hist_bits_p1_arr (see T_UPDATE_A1B) never indexes this
    // array out of range either way.
    wire [STDP_WINDOW-1:0] hist_port_rdata [0:3];
    assign hist_port_rdata[0] = hist_mem_a_porta_rdata;
    assign hist_port_rdata[1] = hist_mem_a_portb_rdata;
    assign hist_port_rdata[2] = hist_mem_b_porta_rdata;
    assign hist_port_rdata[3] = hist_mem_b_portb_rdata;

    // Synchronous reset (see the main clocked block below for the full
    // future-ASIC rationale -- applies uniformly across this module).
    always @(posedge clk) begin
        if (!reset_n) begin
            hist_refresh_addr        <= {SRC_W{1'b0}};
            hist_refresh_addr_q      <= {SRC_W{1'b0}};
            hist_refresh_active      <= 1'b0;
            hist_refresh_write_valid <= 1'b0;
            hist_refresh_done <= 1'b0;
        end else begin
            hist_refresh_done <= 1'b0;

            if (hist_refresh_start && !hist_refresh_active && !hist_refresh_write_valid) begin
                hist_refresh_active <= 1'b1;
                hist_refresh_addr   <= {SRC_W{1'b0}};
            end

            hist_refresh_addr_q      <= hist_refresh_addr;
            hist_refresh_write_valid <= hist_refresh_active;

            if (hist_refresh_active) begin
                if (hist_refresh_addr == {SRC_W{1'b1}})
                    hist_refresh_active <= 1'b0;   // this cycle issued the LAST address
                else
                    hist_refresh_addr <= hist_refresh_addr + 1'b1;
            end

            if (hist_refresh_write_valid && hist_refresh_addr_q == {SRC_W{1'b1}})
                hist_refresh_done <= 1'b1;  // the write for the LAST address happens this cycle (see main block)
        end
    end

    // ---- history memory port address/data muxes (combinational) ----
    // Port A: always a read. During the refresh scan, walks every address in
    // order; otherwise, serves the STDP walk's PACK entries 0 (mem_a) and 2
    // (mem_b), issued combinationally by T_UPDATE_A1 from the just-read
    // synapse row (cur_src_arr, declared with the rest of the PACK-parallel
    // datapath further below).
    // Port B: a write during the refresh scan's Stage 2 (one cycle behind
    // port A's read, same address), otherwise a read for PACK entries 1
    // (mem_a) and 3 (mem_b).
    //
    // PACK_FACTOR is a COMPILE-TIME constant here (1 on non-URAM boards, 4 on
    // URAM boards -- see ENTRY_W/MEM_NATIVE_W above) and cur_src_arr is sized
    // [0:PACK_FACTOR-1], so cur_src_arr[1]/[2]/[3] are literally out-of-range
    // declarations when PACK_FACTOR==1 -- a static elaboration error, not a
    // runtime concern, so an ordinary `if` cannot guard it (both branches
    // would still be elaborated). `generate if` is required: at PACK_FACTOR==1
    // there is only ONE synapse per row, so only mem_a's port A is ever
    // consumed (T_UPDATE_A1B's PACK_FACTOR-bounded loop below never touches
    // index 1+); the other three address wires are still declared (mem_b
    // still physically exists -- trivial extra BRAM on boards with ample
    // spare capacity) but tied to a safe, always-in-range address (0) since
    // nothing ever reads their result.
    wire [SRC_W-1:0] hist_mem_a_porta_addr;
    wire [SRC_W-1:0] hist_mem_b_porta_addr;
    wire [SRC_W-1:0] hist_mem_a_portb_addr;
    wire [SRC_W-1:0] hist_mem_b_portb_addr;
    // NOTE: address muxes moved below entry_src()'s definition (see below) --
    // they must be driven by entry_src(rd_row_reg, ...) directly, NOT by
    // cur_src_arr[ss] (which is written with a BLOCKING assignment inside
    // T_UPDATE_A1). Reading a blocking-assigned reg from a continuous assign
    // that a nonblocking statement in the SAME always block/timestep also
    // depends on (hist_mem_a_porta_rdata <= hist_mem_a[hist_mem_a_porta_addr])
    // is a genuine event-ordering race: the nonblocking RHS can sample the
    // address wire before it re-settles from cur_src_arr's fresh blocking
    // write, issuing a read against the PREVIOUS row's stale address for a
    // few cycles until it "catches up". entry_src(rd_row_reg, sub) is a pure
    // function of rd_row_reg (itself only ever nonblocking-updated, stable
    // all cycle), so driving the address muxes from it directly sidesteps
    // the race entirely instead of relying on simulator-specific ordering.
    // History shift-in for the refresh write: newest spike enters bit 0, the
    // existing STDP_WINDOW-1 older bits shift up one. The `[STDP_WINDOW-2:0]`
    // slice is a NEGATIVE part-select when STDP_WINDOW==1 (`[-1:0]`, malformed
    // -- Vivado/Icarus reject it at elaboration regardless of reachability),
    // so a generate guard picks the degenerate single-bit form (the whole
    // word IS just the newest spike, no old bits retained) when STDP_WINDOW==1.
    // Fixed default is 5; this only matters if the window is ever parameterized
    // down to 1, but keeping the RTL generic-safe costs nothing.
    wire [STDP_WINDOW-1:0] hist_mem_a_portb_wdata;
    wire [STDP_WINDOW-1:0] hist_mem_b_portb_wdata;
    generate
        if (STDP_WINDOW > 1) begin : g_hist_shift_multi
            assign hist_mem_a_portb_wdata = {hist_mem_a_porta_rdata[STDP_WINDOW-2:0], spike_out_global[hist_refresh_addr_q]};
            assign hist_mem_b_portb_wdata = {hist_mem_b_porta_rdata[STDP_WINDOW-2:0], spike_out_global[hist_refresh_addr_q]};
        end else begin : g_hist_shift_single
            assign hist_mem_a_portb_wdata = spike_out_global[hist_refresh_addr_q];
            assign hist_mem_b_portb_wdata = spike_out_global[hist_refresh_addr_q];
        end
    endgenerate

    // ---- helper functions (must precede use for broad tool compatibility,
    // incl. Icarus Verilog in Verilog-2001 mode). 2026-07-21 synthesis-safety
    // rework: these now take an ALREADY-FETCHED row VALUE (not an address) --
    // pure bit-slicing of a value the caller already read via the proper
    // registered-read pattern, never touching syn_row directly themselves.
    // The original versions read syn_row[row_addr] internally, which is
    // exactly the "combinational read of a registered address" pattern
    // snm_gather_lane.v's own comments document as NOT what Vivado infers as
    // a synchronous BRAM port.
    function [SRC_W-1:0] entry_src;
        input [ROW_W-1:0] row;
        input [PACK_SHIFT_W-1:0] sub;
        reg [ENTRY_W-1:0] e;
        begin
            e = row[sub*ENTRY_W +: ENTRY_W];
            entry_src = e[SRC_W-1:0];
        end
    endfunction

    function signed [WEIGHT_W-1:0] entry_weight;
        input [ROW_W-1:0] row;
        input [PACK_SHIFT_W-1:0] sub;
        reg [ENTRY_W-1:0] e;
        begin
            e = row[sub*ENTRY_W +: ENTRY_W];
            entry_weight = e[ENTRY_W-1:SRC_W];
        end
    endfunction

    generate
        if (PACK_FACTOR > 1) begin : g_hist_ports_wide
            assign hist_mem_a_porta_addr = hist_refresh_active ? hist_refresh_addr : entry_src(rd_row_reg, 2'd0);
            assign hist_mem_b_porta_addr = hist_refresh_active ? hist_refresh_addr : entry_src(rd_row_reg, 2'd2);
            assign hist_mem_a_portb_addr = hist_refresh_write_valid ? hist_refresh_addr_q : entry_src(rd_row_reg, 2'd1);
            assign hist_mem_b_portb_addr = hist_refresh_write_valid ? hist_refresh_addr_q : entry_src(rd_row_reg, 2'd3);
        end else begin : g_hist_ports_narrow
            assign hist_mem_a_porta_addr = hist_refresh_active ? hist_refresh_addr : entry_src(rd_row_reg, {PACK_SHIFT_W{1'b0}});
            assign hist_mem_b_porta_addr = hist_refresh_active ? hist_refresh_addr : {SRC_W{1'b0}};
            assign hist_mem_a_portb_addr = hist_refresh_write_valid ? hist_refresh_addr_q : {SRC_W{1'b0}};
            assign hist_mem_b_portb_addr = hist_refresh_write_valid ? hist_refresh_addr_q : {SRC_W{1'b0}};
        end
    endgenerate

    // ---- config read path (synapse readback, 2026-07-25) ----
    // wr_sub (declared next) doubles as the read sub-index: same cfg_syn_idx
    // packing convention a write uses, and reads/writes of a given index are
    // never in flight at the same time (one host command at a time).
    reg cfg_syn_re_d1;

    // ---- config write path (dst_ptr / synapse load / STDP tables) ----
    reg [ROW_W-1:0] wr_accum;
    wire [PACK_SHIFT_W-1:0] wr_sub;
    generate
        if (PACK_SHIFT == 0) begin : g_wr_sub_flat
            assign wr_sub = {PACK_SHIFT_W{1'b0}};
        end else begin : g_wr_sub_split
            assign wr_sub = cfg_syn_idx[PACK_SHIFT-1:0];
        end
    endgenerate
    reg [ROW_W-1:0] wr_accum_next;
    always @(*) begin
        wr_accum_next = wr_accum;
        wr_accum_next[wr_sub*ENTRY_W +: ENTRY_W] = {cfg_syn_weight, cfg_syn_src};
    end

    // ---- STDP write-back (read-modify-write on the packed row) ----
    // Set by the STDP FSM below; shares the physical write port with config
    // load (config load only ever happens before a run starts, STDP write-back
    // only during the STDP phase -- the two never overlap in time).
    reg                     stdp_wr_en;
    reg [ROW_ADDR_W-1:0]    stdp_wr_row_addr;
    reg [ROW_W-1:0]         stdp_wr_row_data;

    // Single muxed write address/data/enable feeding ONE array-write statement
    // below. A first draft wrote syn_row at two different addresses
    // (cfg_syn_idx-derived vs stdp_wr_row_addr) from two `if` arms in the same
    // always block; real Vivado synthesis rejected that ("[Synth 8-3391]
    // Unable to infer a block/distributed RAM for 'syn_row_reg' ... RAM has
    // multiple writes via different ports in same process") even though the
    // two writes are functionally mutually exclusive in time -- the tool wants
    // a single write-address expression per process to infer one write port.
    // Muxing here, in board_variants/npu_stdp_dev only, confirmed via a real
    // synth-only Vivado run (see NOTES.md progress update 3).
    wire                  syn_row_wr_en   = cfg_syn_we | stdp_wr_en;
    wire [ROW_ADDR_W-1:0] syn_row_wr_addr = cfg_syn_we
        ? cfg_syn_idx[ROW_ADDR_W+PACK_SHIFT-1:PACK_SHIFT]
        : stdp_wr_row_addr;
    wire [ROW_W-1:0]      syn_row_wr_data = cfg_syn_we ? wr_accum_next : stdp_wr_row_data;

    always @(posedge clk) begin
        if (cfg_dptr_we) dst_ptr[cfg_dptr_idx] <= cfg_dptr_wdata;
        if (cfg_syn_we) wr_accum <= wr_accum_next;
        if (syn_row_wr_en) syn_row[syn_row_wr_addr] <= syn_row_wr_data;
        if (cfg_stdp_we) begin
            apos_tbl[cfg_stdp_idx] <= cfg_stdp_apos;
            aneg_tbl[cfg_stdp_idx] <= cfg_stdp_aneg;
        end
    end

    // ---- private per-destination accumulators (identical to snm_gather_lane) ----
    // (* use_dsp = "no" *): future-ASIC target (RTL must stay portable to a
    // standard-cell flow, not just FPGA) -- this is the widest accumulate in
    // the design (ACCUM_W-bit add-and-register, the classic pattern Vivado's
    // synthesis will happily fold into a DSP48 ALU/pre-adder on its own).
    // Nothing in this design actually NEEDS a hard multiplier/DSP primitive
    // (grep confirms every `*` operator here is a variable-times-COMPILE-TIME-
    // CONSTANT bit-offset computation for array indexing, e.g. `sub*ENTRY_W`,
    // which synthesizes to address decode logic, not a real multiplier), so
    // this attribute is belt-and-suspenders: it costs nothing on Xilinx parts
    // (LUT/carry-chain adders are already plenty fast at these widths) and
    // keeps the RTL's arithmetic style consistent with what a standard-cell
    // ASIC synthesis flow will map anyway (plain adder cells, no vendor-
    // specific hard macro to account for).
    (* use_dsp = "no" *) reg signed [ACCUM_W-1:0] accum [0:LOCAL_D-1];
    genvar gi;
    generate
        for (gi = 0; gi < LOCAL_D; gi = gi + 1) begin : g_accum_out
            assign accum_out_flat[gi*ACCUM_W +: ACCUM_W] = accum[gi];
        end
    endgenerate

    // ==== INFERENCE walk (identical FSM/pipeline to snm_gather_lane) ====
    localparam [1:0] S_IDLE      = 2'd0;
    localparam [1:0] S_NEXT_DST  = 2'd1;
    localparam [1:0] S_WALK      = 2'd2;
    localparam [1:0] S_DONE      = 2'd3;

    reg [1:0] state;
    reg [$clog2(LOCAL_D+1)-1:0] cur_d;
    reg [DPTR_W-1:0] syn_i, syn_end;

    reg              rd_valid_d1;
    reg [ROW_W-1:0]  rd_row_reg;
    reg [PACK_SHIFT_W-1:0] rd_sub_d1;
    reg [$clog2(LOCAL_D+1)-1:0] rd_d_d1;
    // ---- inference 4-per-row (2026-07-23): process ALL PACK entries of a
    // fetched row per cycle instead of one, same idea as the STDP rework.
    // These latch the row's base cursor and the destination's [lo,hi) range
    // so the accumulate stage (one cycle later) can sum only the in-range
    // entries whose source fired, into accum[rd_d_d1]. ~PACK x fewer inference
    // cycles at the dense worst case; degenerates to the old 1-entry path at
    // PACK_FACTOR==1.
    reg [DPTR_W-1:0] rd_base_d1, rd_lo_d1, rd_hi_d1;
    reg signed [ACCUM_W-1:0] infer_sum;   // combinational (accumulate stage)
    integer isub;

`ifdef SNM_INFER_GATHER_PIPE_DEEP
    // ---- extra accumulate pipeline stage (2026-07-24, ported from
    // snm_gather_lane.v's own SNM_INFER_GATHER_PIPE_DEEP -- see that file's
    // comment for the original single-entry-per-row version and rationale). ----
    // Real Vivado post-route data (this file's own Basys3 STDP N=256/K=8
    // build, WNS=+0.001ns): worst path source=syn_row_reg/CLKBWRCLK, dest=
    // accum_reg[.][.]/D, 13 logic levels (CARRY4x6, MUXF7x2, MUXF8x1) -- the
    // SAME `spike_frozen[rd_row_reg[...]]` wide dynamic mux this define
    // already exists to fix in the plain inference engine, now the binding
    // constraint here too (the history redesign fixed the STDP-side mux
    // forest; this is the inference-side one the analysis flagged as "the
    // next likely hotspot, don't fix until reports confirm it" -- they now
    // do). The PACK-parallel rework (4-per-row) compounds it: PACK_FACTOR
    // independent wide muxes THEN a PACK_FACTOR-way conditional sum, all in
    // the one cycle rd_row_reg becomes valid. Splits exactly like the
    // original: register each entry's {fired, weight, in-range} the cycle
    // the muxes resolve, sum-and-accumulate the FOLLOWING cycle -- ends the
    // wide-mux chain immediately at a flop instead of carrying it into the
    // adder. Adds one fixed cycle of pipeline-fill/drain latency (not a
    // per-row throughput cost -- reads still issue every cycle). Board-
    // selected the same way as the original (gen_config.py / boards.yaml's
    // infer_gather_pipe_deep field); Basys3's STDP build scripts already
    // pass this define (it was already wired through for the plain engine),
    // so this fix applies automatically on rebuild with no script changes.
    reg                       rd_valid_d2;
    reg [$clog2(LOCAL_D+1)-1:0] rd_d_d2;
    reg                       fired_d2    [0:PACK_FACTOR-1];
    reg signed [WEIGHT_W-1:0] weight_d2   [0:PACK_FACTOR-1];
    reg                       inrange_d2  [0:PACK_FACTOR-1];
`endif

    wire [PACK_SHIFT_W-1:0] rd_sub_next;
    generate
        if (PACK_SHIFT == 0) begin : g_rd_sub_flat
            assign rd_sub_next = {PACK_SHIFT_W{1'b0}};
        end else begin : g_rd_sub_split
            assign rd_sub_next = syn_i[PACK_SHIFT-1:0];
        end
    endgenerate

    wire [ENTRY_W-1:0] rd_entry = rd_row_reg[rd_sub_d1*ENTRY_W +: ENTRY_W];
    wire [SRC_W-1:0] rd_src    = rd_entry[SRC_W-1:0];
    wire signed [WEIGHT_W-1:0] rd_weight = rd_entry[ENTRY_W-1:SRC_W];

    // ==== STDP walk (2026-07-24 II=1 PIPELINE REWORK) ====
    // Replaces the previous single-row-in-flight, 6-state sequential walk
    // (T_ISSUE..T_UPDATE_B, ~6 cycles/row-group, only one row anywhere in the
    // datapath at a time) with a true 5-stage pipeline that accepts a NEW row
    // every cycle (initiation interval 1) whenever there is no address
    // hazard. Every quantity that used to be a scalar "current row" register
    // held across several states (row address, the destination's [lo,hi)
    // range, dst_fired, the fetched row content, extracted src/weight) is
    // now a genuine per-position tag/data record that shifts forward one
    // position per cycle: with multiple rows in flight simultaneously, a
    // plain "write once, read several cycles later" register would get
    // clobbered by a LATER row's write before an EARLIER row's consumer
    // ever reads it -- the exact class of bug this project's own history
    // BRAM redesign just found and fixed for a related reason (see the
    // hist_mem_a/b comments above), so this rework applies the same
    // discipline deliberately rather than by accident.
    //
    // Positions (relative to the cycle a row's read address is issued):
    //   1 = read issued this cycle (rd_addr_mux picks this row's address;
    //       fuses the OLD T_ISSUE+T_WAIT_READ into a single cycle since
    //       there is no longer any reason to spend an extra cycle just
    //       latching an address before using it)
    //   2 = row data valid (rd_row_reg) -- extract src/weight, issue history
    //       BRAM reads (entry_src(rd_row_reg,...), unchanged from before)
    //   3 = history BRAM reads valid (hist_port_rdata) -- latch into hist bits
    //   4 = pure adder tree over the latched history bits -> delta (unchanged
    //       from the original T_UPDATE_A2's logic)
    //   5 = saturating add + row-pack + ISSUE write-back (stdp_wr_en<=1); the
    //       actual syn_row write commits ONE MORE cycle later, in the
    //       separate config-write always block (~line 384) -- an existing,
    //       already synthesis-validated structure (see that block's own
    //       comment for why the write can't just live in this block). The
    //       hazard check below accounts for that extra cycle directly via
    //       stdp_wr_en/stdp_wr_row_addr (no separate "position 6" register
    //       needed -- those ARE position 6, already present for other
    //       reasons).
    //
    // HAZARD: two DIFFERENT destinations can share one physical row (a row
    // holds PACK_FACTOR entries; a destination's synapse range need not be
    // row-aligned), so a later-issued row can target the SAME syn_row
    // address as an earlier row still in flight. Reading that address before
    // the earlier row's write commits would silently lose the earlier
    // update (read-modify-write on a stale copy). Since this only matters
    // for identical addresses, the fix is a simple in-flight address
    // comparison: before issuing a new row, compare its address against
    // every row currently resident in positions 2..5 AND against the
    // pending write-commit (stdp_wr_en/stdp_wr_row_addr); on a match, stall
    // (bubble) -- the conflicting row is guaranteed to keep shifting forward
    // and eventually retire (nothing else in this design ever stalls a row
    // once it is in flight), so the hazard always clears within a few
    // cycles and can never deadlock.
    localparam [1:0] G_IDLE     = 2'd0;
    localparam [1:0] G_NEXT_DST = 2'd1;
    localparam [1:0] G_ROW      = 2'd2;
    localparam [1:0] G_DONE     = 2'd3;

    reg [1:0] gstate;
    reg [$clog2(LOCAL_D+1)-1:0] g_cur_d;
    reg [DPTR_W-1:0] g_syn_i, g_syn_end;
    reg              g_dst_fired;

    // STDP delta accumulator width. The per-synapse delta sums up to
    // STDP_WINDOW terms, each a WEIGHT_W-bit signed table entry (magnitude up
    // to 2^(WEIGHT_W-1)), then that delta is added to a WEIGHT_W-bit weight.
    // Worst-case magnitude ~ (STDP_WINDOW+1) * 2^(WEIGHT_W-1), so the width
    // must grow with log2(STDP_WINDOW). The old hardcoded `WEIGHT_W + 4` is
    // correct ONLY up to STDP_WINDOW ~= 7 (WINDOW=5 needs +4: clog2(6)=3 plus
    // one carry bit for the weight add -> 4); it would silently overflow the
    // saturation compare for larger windows. Derive it instead so the module
    // stays parameter-safe (matters for the future-ASIC config, where
    // STDP_WINDOW may not stay 5). For STDP_WINDOW=5 this evaluates to
    // WEIGHT_W+4 exactly -- identical to before, zero behavioural change.
    localparam integer SUM_GROWTH = (STDP_WINDOW <= 1) ? 1 : $clog2(STDP_WINDOW + 1);
    localparam integer SUMW_W = WEIGHT_W + SUM_GROWTH + 1;

    // ---- pipeline TAG: present at every position, forwarded every cycle ----
    reg                  p2_valid, p3_valid, p4_valid, p5_valid;
    reg [ROW_ADDR_W-1:0] p2_row_addr, p3_row_addr, p4_row_addr, p5_row_addr;
    reg [DPTR_W-1:0]     p2_row_base, p3_row_base, p4_row_base, p5_row_base;
    reg [DPTR_W-1:0]     p2_syn_lo,   p3_syn_lo,   p4_syn_lo,   p5_syn_lo;
    reg [DPTR_W-1:0]     p2_syn_hi,   p3_syn_hi,   p4_syn_hi,   p5_syn_hi;
    reg                  p2_dst_fired,p3_dst_fired,p4_dst_fired,p5_dst_fired;

    // ---- pipeline DATA: created once (from rd_row_reg, at position 2) and
    // delayed forward -- these need real per-position storage (not a single
    // shared register) because they are CONSUMED two positions later than
    // they are created (position 5), unlike hist_bits_arr/delta_acc_arr
    // below which are consumed the very next position after creation and so
    // need no extra depth of their own (only one row ever occupies a given
    // position in a given cycle -- true pipeline, not a shared resource). ----
    reg [ROW_W-1:0]           p3_row_data, p4_row_data, p5_row_data;
    reg [SRC_W-1:0]           p3_cur_src    [0:PACK_FACTOR-1];
    reg [SRC_W-1:0]           p4_cur_src    [0:PACK_FACTOR-1];
    reg [SRC_W-1:0]           p5_cur_src    [0:PACK_FACTOR-1];
    reg signed [WEIGHT_W-1:0] p3_cur_weight [0:PACK_FACTOR-1];
    reg signed [WEIGHT_W-1:0] p4_cur_weight [0:PACK_FACTOR-1];
    reg signed [WEIGHT_W-1:0] p5_cur_weight [0:PACK_FACTOR-1];

    // ---- per-position combinational/single-buffered scratch (same role as
    // the original cur_src_arr/hist_bits_p1_arr/delta_acc_p1_arr -- these
    // run EVERY cycle unconditionally now instead of inside a case-branch,
    // exactly mirroring how the old FSM's per-state actions were themselves
    // unconditional-once-you're-in-that-state; "being in a position" is now
    // simply "the corresponding p*_valid is true", checked only where it
    // actually matters for correctness (write-back issuance, gated on
    // p5_valid below) -- values computed while invalid are harmless, never
    // consumed. ----
    reg [SRC_W-1:0]           cur_src_arr    [0:PACK_FACTOR-1];
    reg signed [WEIGHT_W-1:0] cur_weight_arr [0:PACK_FACTOR-1];
    reg [STDP_WINDOW-1:0]     hist_bits_arr  [0:PACK_FACTOR-1];  // latched @3, consumed @4
    reg signed [SUMW_W-1:0]   delta_acc_arr  [0:PACK_FACTOR-1];  // comb @4
    // (* use_dsp = "no" *): the STDP-window adder tree (delta) and the
    // saturating add against the weight (sum_w) below -- same future-ASIC
    // rationale as accum[] above, applied to the STDP datapath's own
    // widest adds.
    (* use_dsp = "no" *) reg signed [SUMW_W-1:0]   delta_p1_arr   [0:PACK_FACTOR-1];  // latched @4, consumed @5
    integer jj, ss;
    reg signed [WEIGHT_W-1:0] new_weight;
    (* use_dsp = "no" *) reg signed [SUMW_W-1:0]   sum_w;
    reg signed [SUMW_W-1:0]   wmax_value, wmin_value;
    reg [ROW_W-1:0]           wr_row_tmp;

    // ---- combinational issue/hazard logic (generator side) ----
    // Row-address arithmetic mirrors the inference walk's own rd_sub_next
    // pattern exactly (same PACK_SHIFT==0 malformed-part-select trap this
    // file's other generate/if wires already document and avoid -- here
    // sidestepped entirely since the row-address shift only ever operates
    // on g_syn_i, needing no per-entry "sub" index at all now that write-
    // back always processes the whole PACK_FACTOR-wide row in one shot).
    wire [ROW_ADDR_W-1:0] gen_candidate_addr = g_syn_i[ROW_ADDR_W+PACK_SHIFT-1:PACK_SHIFT];
    wire [DPTR_W-1:0]     gen_candidate_base = (g_syn_i >> PACK_SHIFT) << PACK_SHIFT;
    wire gen_hazard = (p2_valid && p2_row_addr == gen_candidate_addr) ||
                      (p3_valid && p3_row_addr == gen_candidate_addr) ||
                      (p4_valid && p4_row_addr == gen_candidate_addr) ||
                      (p5_valid && p5_row_addr == gen_candidate_addr) ||
                      (stdp_wr_en && stdp_wr_row_addr == gen_candidate_addr);
    wire gen_issue_valid = (gstate == G_ROW) && (g_syn_i < g_syn_end) && !gen_hazard;


    // ==== SHARED single BRAM read port (2026-07-22 de-duplication rework) ====
    // Previously this module had TWO separate `reg <= syn_row[addr];` read
    // statements -- one for the inference walk (S_WALK), one for the STDP walk
    // (T_WAIT_READ). Vivado maps each distinct read statement to its OWN
    // physical BRAM read port; with the write port that is 2R + 1W = 3 ports,
    // but a 7-series/UltraScale+ BRAM is TRUE DUAL-PORT (2 ports total, each
    // read-OR-write per cycle). Unable to fit 3 ports onto 2, Vivado
    // DUPLICATED the entire synapse memory (syn_row_reg_1 + syn_row_reg_2, ~2x
    // the BRAM tiles) so each read got its own copy -- the dominant reason
    // STDP capacity fell so far below the inference-only max.
    //
    // KEY OBSERVATION making the fix free: the inference walk and the STDP walk
    // are ALREADY strictly time-disjoint -- snm_infer_multilane_stdp.v's tick
    // barrier runs B_INFER (all lanes finish inference) fully BEFORE B_STDP (all
    // lanes do STDP), so these two reads NEVER occur in the same cycle. They
    // can therefore SHARE one physical read port via a simple address mux. That
    // leaves exactly 1 read + 1 write = the two native ports of ONE true-dual-
    // port BRAM, no duplication -- restoring ~1x BRAM (inference-only capacity).
    //
    // The single read statement lives in the clocked block below as
    // `rd_row_reg <= syn_row[rd_addr_mux];` (unconditional, every cycle -- the
    // cleanest BRAM-read template; the result is only CONSUMED when valid).
    // Inference consumes rd_row_reg directly (its accumulate stage is one cycle
    // after S_WALK presents the address, matching the BRAM read latency, exactly
    // as before). STDP consumes rd_row_reg in T_UPDATE_A1 and latches it into
    // t_row_reg (a plain reg-to-reg copy -- NOT an array read, so it adds no
    // BRAM port) for use by the later T_UPDATE_B row-pack.
    reg [ROW_ADDR_W-1:0] rd_addr_mux;
    always @(*) begin
        if (gen_issue_valid)
            rd_addr_mux = gen_candidate_addr;                             // STDP-phase read address
        else if (cfg_syn_re)
            rd_addr_mux = cfg_syn_idx[ROW_ADDR_W+PACK_SHIFT-1:PACK_SHIFT]; // host config-read address
        else
            rd_addr_mux = syn_i[ROW_ADDR_W+PACK_SHIFT-1:PACK_SHIFT];       // inference-phase read address
    end

    integer k;
    // 2026-07-24 synchronous-reset rework (future-ASIC methodology fix): this
    // module's reset touches several LOCAL_D/PACK_FACTOR-wide arrays (accum[],
    // the STDP pipeline's p2..p5 tag/data records). An asynchronous reset
    // fans a single global reset_n net out to every bit of every one of
    // these registers combinationally, which is exactly the kind of large,
    // ad-hoc async reset tree standard-cell ASIC flows want to avoid (reset
    // recovery/removal timing on every leaf, plus a wide OR-tree of reset
    // buffers that don't scale with array width the way ordinary datapath
    // logic does). Switched to a synchronous reset (`posedge clk` only,
    // `if (!reset_n)` as the first branch) -- same reset VALUES and same
    // reset LOGIC as before, just applied ON the clock edge instead of
    // immediately; verified transparent to every testbench in this project
    // (each already holds reset_n low across several clock edges before
    // releasing it, so no test ever depended on true asynchronous
    // reset-removal timing). The large history/synapse memories
    // (hist_mem_a/b, syn_row) were already never bulk-reset at all (see
    // their own declaration comments) -- this closes the same gap for the
    // smaller-but-still-nontrivial arrays that WERE being reset.
    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            run_busy <= 1'b0;
            run_done <= 1'b0;
            cur_d <= {($clog2(LOCAL_D+1)){1'b0}};
            syn_i <= {DPTR_W{1'b0}};
            syn_end <= {DPTR_W{1'b0}};
            rd_valid_d1 <= 1'b0;
            rd_row_reg <= {ROW_W{1'b0}};
            rd_sub_d1 <= {PACK_SHIFT_W{1'b0}};
            rd_d_d1 <= {($clog2(LOCAL_D+1)){1'b0}};
            rd_base_d1 <= {DPTR_W{1'b0}};
            rd_lo_d1 <= {DPTR_W{1'b0}};
            rd_hi_d1 <= {DPTR_W{1'b0}};
`ifdef SNM_INFER_GATHER_PIPE_DEEP
            rd_valid_d2 <= 1'b0;
            rd_d_d2 <= {($clog2(LOCAL_D+1)){1'b0}};
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                fired_d2[ss]   <= 1'b0;
                weight_d2[ss]  <= {WEIGHT_W{1'b0}};
                inrange_d2[ss] <= 1'b0;
            end
`endif
            cycles_this_run <= 32'd0;
            for (k = 0; k < LOCAL_D; k = k + 1) accum[k] <= {ACCUM_W{1'b0}};

            gstate <= G_IDLE;
            stdp_busy <= 1'b0;
            stdp_done <= 1'b0;
            g_cur_d <= {($clog2(LOCAL_D+1)){1'b0}};
            g_syn_i <= {DPTR_W{1'b0}};
            g_syn_end <= {DPTR_W{1'b0}};
            g_dst_fired <= 1'b0;
            p2_valid <= 1'b0; p3_valid <= 1'b0; p4_valid <= 1'b0; p5_valid <= 1'b0;
            p2_row_addr <= {ROW_ADDR_W{1'b0}}; p3_row_addr <= {ROW_ADDR_W{1'b0}};
            p4_row_addr <= {ROW_ADDR_W{1'b0}}; p5_row_addr <= {ROW_ADDR_W{1'b0}};
            p2_row_base <= {DPTR_W{1'b0}}; p3_row_base <= {DPTR_W{1'b0}};
            p4_row_base <= {DPTR_W{1'b0}}; p5_row_base <= {DPTR_W{1'b0}};
            p2_syn_lo <= {DPTR_W{1'b0}}; p3_syn_lo <= {DPTR_W{1'b0}};
            p4_syn_lo <= {DPTR_W{1'b0}}; p5_syn_lo <= {DPTR_W{1'b0}};
            p2_syn_hi <= {DPTR_W{1'b0}}; p3_syn_hi <= {DPTR_W{1'b0}};
            p4_syn_hi <= {DPTR_W{1'b0}}; p5_syn_hi <= {DPTR_W{1'b0}};
            p2_dst_fired <= 1'b0; p3_dst_fired <= 1'b0; p4_dst_fired <= 1'b0; p5_dst_fired <= 1'b0;
            p3_row_data <= {ROW_W{1'b0}}; p4_row_data <= {ROW_W{1'b0}}; p5_row_data <= {ROW_W{1'b0}};
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                p3_cur_src[ss]    <= {SRC_W{1'b0}};
                p4_cur_src[ss]    <= {SRC_W{1'b0}};
                p5_cur_src[ss]    <= {SRC_W{1'b0}};
                p3_cur_weight[ss] <= {WEIGHT_W{1'b0}};
                p4_cur_weight[ss] <= {WEIGHT_W{1'b0}};
                p5_cur_weight[ss] <= {WEIGHT_W{1'b0}};
                hist_bits_arr[ss] <= {STDP_WINDOW{1'b0}};
                delta_p1_arr[ss]  <= {SUMW_W{1'b0}};
            end
            hist_mem_a_porta_rdata <= {STDP_WINDOW{1'b0}};
            hist_mem_b_porta_rdata <= {STDP_WINDOW{1'b0}};
            hist_mem_a_portb_rdata <= {STDP_WINDOW{1'b0}};
            hist_mem_b_portb_rdata <= {STDP_WINDOW{1'b0}};
            stdp_wr_en <= 1'b0;
            stdp_wr_row_addr <= {ROW_ADDR_W{1'b0}};
            stdp_wr_row_data <= {ROW_W{1'b0}};
            stdp_cycles_this_run <= 32'd0;
            cfg_syn_re_d1 <= 1'b0;
            cfg_syn_rd_valid <= 1'b0;
            cfg_syn_rd_src <= {SRC_W{1'b0}};
            cfg_syn_rd_weight <= {WEIGHT_W{1'b0}};
        end else begin
            run_done <= 1'b0;
            rd_valid_d1 <= 1'b0;
            stdp_done <= 1'b0;
            stdp_wr_en <= 1'b0;

            // ---- THE single shared BRAM read port (see the de-duplication
            // comment above). Unconditional every cycle; address muxed between
            // the inference and STDP walks (never active in the same cycle).
            rd_row_reg <= syn_row[rd_addr_mux];

            // ---- config synapse readback (2026-07-25): rd_row_reg becomes
            // valid one cycle after cfg_syn_re presented its address (above);
            // cfg_syn_re_d1 tracks that same one-cycle delay so this unpacks
            // the row the SAME cycle rd_row_reg carries it, then registers the
            // unpacked entry (2 cycles total latency from cfg_syn_re to
            // cfg_syn_rd_valid). cfg_syn_idx is held stable by the host for
            // the whole request (one command in flight at a time), so reusing
            // wr_sub (combinationally derived from the still-stable cfg_syn_idx)
            // here is safe.
            cfg_syn_re_d1    <= cfg_syn_re;
            cfg_syn_rd_valid <= 1'b0;
            if (cfg_syn_re_d1) begin
                cfg_syn_rd_src    <= entry_src(rd_row_reg, wr_sub);
                cfg_syn_rd_weight <= entry_weight(rd_row_reg, wr_sub);
                cfg_syn_rd_valid  <= 1'b1;
            end

            // ---- history memory ports (see the hist_mem_a/hist_mem_b
            // declaration's comment above): exactly two physical ports per
            // memory, matching a true-dual-port BRAM. Port A: unconditional
            // read every cycle (refresh scan or STDP-walk PACK entry
            // 0/2, address-muxed combinationally). Port B: a write during the
            // refresh scan's Stage 2, else a read for PACK entry 1/3.
            hist_mem_a_porta_rdata <= hist_mem_a[hist_mem_a_porta_addr];
            hist_mem_b_porta_rdata <= hist_mem_b[hist_mem_b_porta_addr];
            // Port B is READ-OR-WRITE (one true-dual-port port can do exactly
            // one of the two per cycle). During the refresh scan's Stage 2 it
            // WRITES (the shifted history word); at ALL other times -- i.e.
            // during the STDP walk -- it must READ, feeding PACK entries 1
            // (mem_a) and 3 (mem_b). The STDP-walk read is the common case,
            // so it belongs in the `else`. (2026-07-24 fix: this else branch
            // was missing -- port B only ever read during refresh, so the
            // STDP walk's entries 1/3 consumed a STALE hist_mem_*_portb_rdata,
            // producing X deltas -> X weights on PACK_FACTOR>1 boards. This
            // never showed up in test because every complex reference TB runs
            // at PACK_FACTOR==1 (none define SNM_SYN_MEM_ULTRA), where entries
            // 1/3 and mem_b are unused entirely; found via an external PACK=4
            // review. The refresh branch does NOT need a port-B read -- its
            // read result is never consumed by the STDP datapath.)
            if (hist_refresh_write_valid) begin
                hist_mem_a[hist_mem_a_portb_addr] <= hist_mem_a_portb_wdata;
                hist_mem_b[hist_mem_b_portb_addr] <= hist_mem_b_portb_wdata;
            end else begin
                hist_mem_a_portb_rdata <= hist_mem_a[hist_mem_a_portb_addr];
                hist_mem_b_portb_rdata <= hist_mem_b[hist_mem_b_portb_addr];
            end

            // ---- inference accumulate: sum ALL in-range entries of the
            // fetched row whose source fired. Entries outside the
            // destination's [rd_lo_d1, rd_hi_d1) range (row shared with an
            // adjacent destination) are skipped -- they accumulate when their
            // own destination is walked.
`ifdef SNM_INFER_GATHER_PIPE_DEEP
            // Split into two cycles (see the rd_valid_d2 declaration's own
            // comment for the real post-route data motivating this): register
            // each PACK entry's {fired, weight, in-range} the cycle
            // rd_row_reg/spike_frozen resolve, then sum-and-accumulate the
            // FOLLOWING cycle off the now-registered bits. Reads still issue
            // every cycle regardless (S_WALK below is unchanged) -- this only
            // adds one fixed cycle of pipeline-fill/drain latency.
            rd_valid_d2 <= 1'b0;
            if (rd_valid_d1) begin
                rd_valid_d2 <= 1'b1;
                rd_d_d2     <= rd_d_d1;
                for (isub = 0; isub < PACK_FACTOR; isub = isub + 1) begin
                    fired_d2[isub]   <= spike_frozen[rd_row_reg[isub*ENTRY_W +: SRC_W]];
                    weight_d2[isub]  <= $signed(rd_row_reg[isub*ENTRY_W + SRC_W +: WEIGHT_W]);
                    inrange_d2[isub] <= ((rd_base_d1 + isub[DPTR_W-1:0]) >= rd_lo_d1) &&
                                        ((rd_base_d1 + isub[DPTR_W-1:0]) <  rd_hi_d1);
                end
            end
            if (rd_valid_d2) begin
                infer_sum = {ACCUM_W{1'b0}};
                for (isub = 0; isub < PACK_FACTOR; isub = isub + 1)
                    if (inrange_d2[isub] && fired_d2[isub])
                        infer_sum = infer_sum + weight_d2[isub];
                accum[rd_d_d2] <= accum[rd_d_d2] + infer_sum;
            end
`else
            if (rd_valid_d1) begin
                infer_sum = {ACCUM_W{1'b0}};
                for (isub = 0; isub < PACK_FACTOR; isub = isub + 1) begin
                    if (((rd_base_d1 + isub[DPTR_W-1:0]) >= rd_lo_d1) &&
                        ((rd_base_d1 + isub[DPTR_W-1:0]) <  rd_hi_d1) &&
                        spike_frozen[rd_row_reg[isub*ENTRY_W +: SRC_W]])
                        infer_sum = infer_sum +
                            $signed(rd_row_reg[isub*ENTRY_W + SRC_W +: WEIGHT_W]);
                end
                accum[rd_d_d1] <= accum[rd_d_d1] + infer_sum;
            end
`endif

            case (state)
                S_IDLE: begin
                    run_busy <= 1'b0;
                    if (run_start) begin
                        run_busy <= 1'b1;
                        cur_d <= {($clog2(LOCAL_D+1)){1'b0}};
                        cycles_this_run <= 32'd0;
                        for (k = 0; k < LOCAL_D; k = k + 1) accum[k] <= {ACCUM_W{1'b0}};
                        state <= S_NEXT_DST;
                    end
                end
                S_NEXT_DST: begin
                    cycles_this_run <= cycles_this_run + 1'b1;
                    if (cur_d >= LOCAL_D) begin
                        state <= S_DONE;
                    end else begin
                        syn_i   <= dst_ptr[cur_d];
                        syn_end <= dst_ptr[cur_d + 1'b1];
                        state   <= S_WALK;
                    end
                end
                S_WALK: begin
                    cycles_this_run <= cycles_this_run + 1'b1;
                    if (syn_i < syn_end) begin
                        // The shared read at the top of this block fetches the
                        // row for the CURRENT syn_i (rd_addr_mux = syn_i>>PACK).
                        // Latch the row base + this destination's [syn_i,syn_end)
                        // range so next cycle's accumulate sums all in-range
                        // entries of that row at once, then jump to the NEXT row.
                        rd_sub_d1   <= rd_sub_next;
                        rd_d_d1     <= cur_d;
                        rd_base_d1  <= (syn_i >> PACK_SHIFT) << PACK_SHIFT;
                        rd_lo_d1    <= syn_i;
                        rd_hi_d1    <= syn_end;
                        rd_valid_d1 <= 1'b1;
                        syn_i <= ((syn_i >> PACK_SHIFT) << PACK_SHIFT) + PACK_FACTOR;
                    end else begin
                        cur_d <= cur_d + 1'b1;
                        state <= S_NEXT_DST;
                    end
                end
                S_DONE: begin
                    // MUST also wait for the extra accumulate stage (when
                    // present) to drain -- otherwise the tick could be
                    // declared done, and accum[] read out/reset by the next
                    // run, one cycle before the LAST synapse's accumulate
                    // actually lands, silently dropping it (the exact bug
                    // class this project has hit before with pipeline-depth
                    // changes -- see snm_gather_lane.v's own S_DONE comment).
`ifdef SNM_INFER_GATHER_PIPE_DEEP
                    if (!rd_valid_d1 && !rd_valid_d2) begin
`else
                    if (!rd_valid_d1) begin
`endif
                        run_busy <= 1'b0;
                        run_done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        cycles_this_run <= cycles_this_run + 1'b1;
                    end
                end
                default: state <= S_IDLE;
            endcase

            // ==== STDP pipeline: shift the tag + delayed data forward one
            // position every cycle, unconditionally (matches the original
            // FSM's own per-state actions being unconditional-once-in-that-
            // state; validity now lives in p*_valid instead of in which
            // case-branch is executing). ====

            // ---- position 2: row data now valid (rd_row_reg). Extract
            // src/weight (same entry_src/entry_weight calls as before) and
            // forward the tag from position 1 (the generator's combinational
            // issue this cycle) into position 2. History-BRAM read issuance
            // is unchanged -- see the hist_mem_a/b generate block above,
            // which already derives its addresses from entry_src(rd_row_reg,
            // ...) unconditionally every cycle. ----
            p2_valid     <= gen_issue_valid;
            p2_row_addr  <= gen_candidate_addr;
            p2_row_base  <= gen_candidate_base;
            p2_syn_lo    <= g_syn_i;
            p2_syn_hi    <= g_syn_end;
            p2_dst_fired <= g_dst_fired;

            p3_row_data <= rd_row_reg;
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                cur_src_arr[ss]    = entry_src(rd_row_reg, ss[PACK_SHIFT_W-1:0]);
                cur_weight_arr[ss] = entry_weight(rd_row_reg, ss[PACK_SHIFT_W-1:0]);
                p3_cur_src[ss]    <= cur_src_arr[ss];
                p3_cur_weight[ss] <= cur_weight_arr[ss];
            end
            p3_valid <= p2_valid; p3_row_addr <= p2_row_addr; p3_row_base <= p2_row_base;
            p3_syn_lo <= p2_syn_lo; p3_syn_hi <= p2_syn_hi; p3_dst_fired <= p2_dst_fired;

            // ---- position 3: history BRAM reads now valid (hist_port_rdata,
            // 1 cycle after position 2 issued the address) -- latch them
            // (same as the old T_UPDATE_A1B), and forward row_data/cur_src/
            // cur_weight + tag from position 3 into position 4. ----
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1)
                hist_bits_arr[ss] <= hist_port_rdata[ss];
            p4_row_data <= p3_row_data;
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                p4_cur_src[ss]    <= p3_cur_src[ss];
                p4_cur_weight[ss] <= p3_cur_weight[ss];
            end
            p4_valid <= p3_valid; p4_row_addr <= p3_row_addr; p4_row_base <= p3_row_base;
            p4_syn_lo <= p3_syn_lo; p4_syn_hi <= p3_syn_hi; p4_dst_fired <= p3_dst_fired;

            // ---- position 4: pure adder tree over the latched history bits
            // (identical math to the old T_UPDATE_A2 -- see that state's own
            // comment for the superneuromat rule this implements), then
            // forward row_data/cur_src/cur_weight + tag into position 5. ----
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                delta_acc_arr[ss] = {SUMW_W{1'b0}};
                for (jj = 0; jj < STDP_WINDOW; jj = jj + 1)
                    if (jj < stdp_win_eff)
                        delta_acc_arr[ss] = delta_acc_arr[ss] +
                            ((p4_dst_fired && hist_bits_arr[ss][jj])
                                ? apos_tbl[jj] : aneg_tbl[jj]);
                delta_p1_arr[ss] <= delta_acc_arr[ss];
            end
            p5_row_data <= p4_row_data;
            for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                p5_cur_src[ss]    <= p4_cur_src[ss];
                p5_cur_weight[ss] <= p4_cur_weight[ss];
            end
            p5_valid <= p4_valid; p5_row_addr <= p4_row_addr; p5_row_base <= p4_row_base;
            p5_syn_lo <= p4_syn_lo; p5_syn_hi <= p4_syn_hi; p5_dst_fired <= p4_dst_fired;

            // ---- position 5: saturating add + row-pack + issue write-back
            // (identical math to the old T_UPDATE_B -- see that state's own
            // comments for the signedness/width bugs this already fixed).
            // GATED on p5_valid: this is the one place validity actually
            // matters, so an empty pipeline slot never issues a spurious
            // write. ----
            if (p5_valid) begin
                wmax_value = {{(SUMW_W-WEIGHT_W+1){1'b0}}, {(WEIGHT_W-1){1'b1}}};
                wmin_value = {{(SUMW_W-WEIGHT_W+1){1'b1}}, {(WEIGHT_W-1){1'b0}}};
                // wr_row_tmp starts as the ORIGINAL fetched row; only entries
                // inside this row's own [p5_syn_lo, p5_syn_hi) range are
                // updated (see the tag's own comment for why only the FIRST
                // row of a destination can have a non-trivial lo bound and
                // only the LAST can have a non-trivial hi bound -- middle
                // rows update all PACK_FACTOR entries unconditionally).
                // Entries outside the range keep their original value; they
                // get their own update when their own destination is walked.
                wr_row_tmp = p5_row_data;
                for (ss = 0; ss < PACK_FACTOR; ss = ss + 1) begin
                    if (((p5_row_base + ss[DPTR_W-1:0]) >= p5_syn_lo) &&
                        ((p5_row_base + ss[DPTR_W-1:0]) <  p5_syn_hi)) begin
                        sum_w = {{(SUMW_W-WEIGHT_W){p5_cur_weight[ss][WEIGHT_W-1]}},
                                 p5_cur_weight[ss]} + delta_p1_arr[ss];
                        if (sum_w > wmax_value)
                            new_weight = {1'b0, {(WEIGHT_W-1){1'b1}}};
                        else if (sum_w < wmin_value)
                            new_weight = {1'b1, {(WEIGHT_W-1){1'b0}}};
                        else
                            new_weight = sum_w[WEIGHT_W-1:0];
                        wr_row_tmp[ss*ENTRY_W +: ENTRY_W] = {new_weight, p5_cur_src[ss]};
                    end
                end
`ifdef SNM_STDP_DEBUG
                $display("[STDP-WB @%0t] row=%0d base=%0d [%0d,%0d) win=%0d fired=%0d w0=%0d d0=%0d hb0=%b wr=%b",
                    $time, p5_row_addr, p5_row_base, p5_syn_lo, p5_syn_hi, stdp_win_eff,
                    p5_dst_fired, p5_cur_weight[0], delta_p1_arr[0], hist_bits_arr[0], wr_row_tmp);
`endif
                stdp_wr_en       <= 1'b1;
                stdp_wr_row_addr <= p5_row_addr;
                stdp_wr_row_data <= wr_row_tmp;
            end

            if (stdp_busy) stdp_cycles_this_run <= stdp_cycles_this_run + 1'b1;

            // ==== generator: decides which row (if any) to inject at
            // position 1 each cycle -- see gen_issue_valid/gen_candidate_*
            // above for the combinational issue+hazard logic; this just
            // walks destinations and advances g_syn_i once a row actually
            // gets issued. ====
            case (gstate)
                G_IDLE: begin
                    stdp_busy <= 1'b0;
                    if (stdp_start) begin
                        stdp_busy <= 1'b1;
                        g_cur_d <= {($clog2(LOCAL_D+1)){1'b0}};
                        stdp_cycles_this_run <= 32'd0;
                        gstate <= G_NEXT_DST;
                    end
                end
                G_NEXT_DST: begin
                    if (g_cur_d >= LOCAL_D) begin
                        gstate <= G_DONE;
                    end else if (!stdp_global_enable || (stdp_win_eff == 0)) begin
                        // STDP globally off, OR this is the very first tick
                        // (superneuromat's t==0 case) -- no synapse is
                        // updated at all. Skip the whole lane's walk.
                        g_cur_d <= g_cur_d + 1'b1;
                    end else begin
                        // NOTE: we deliberately do NOT skip destinations that
                        // did not fire. superneuromat applies sum(Aneg[j<t])
                        // to EVERY stdp-enabled synapse every tick regardless
                        // of whether the destination fired; only the Apos
                        // part is gated on firing. Skipping non-firing
                        // destinations was the single biggest source of
                        // divergence from superneuromat (it let hardware
                        // weights stay high while software steadily
                        // depressed them, so hardware fired far more). Cost:
                        // the STDP walk is O(all synapses) per tick rather
                        // than O(firing destinations x fan-in).
                        g_dst_fired <= spike_out_local[g_cur_d];
                        g_syn_i     <= dst_ptr[g_cur_d];
                        g_syn_end   <= dst_ptr[g_cur_d + 1'b1];
                        gstate      <= G_ROW;
                    end
                end
                G_ROW: begin
                    if (g_syn_i >= g_syn_end) begin
                        g_cur_d <= g_cur_d + 1'b1;
                        gstate  <= G_NEXT_DST;
                    end else if (gen_issue_valid) begin
                        // Advance to the START of the next row. On a hazard
                        // stall (gen_issue_valid false but g_syn_i still <
                        // g_syn_end), g_syn_i is simply left unchanged and
                        // retried next cycle -- the conflicting in-flight
                        // row is guaranteed to keep shifting forward, so
                        // this always clears.
                        g_syn_i <= gen_candidate_base + PACK_FACTOR;
                    end
                end
                G_DONE: begin
                    // Wait for the pipeline to fully drain (every position
                    // empty AND no write still pending commit) before
                    // declaring the tick's STDP walk done -- stdp_done firing
                    // early would let the multilane barrier start the next
                    // tick's history refresh while a write-back from THIS
                    // tick is still in flight.
                    if (!p2_valid && !p3_valid && !p4_valid && !p5_valid && !stdp_wr_en) begin
                        stdp_busy <= 1'b0;
                        stdp_done <= 1'b1;
                        gstate    <= G_IDLE;
                    end
                end
                default: gstate <= G_IDLE;
            endcase
        end
    end

    // ==== simulation-only invariant checks (2026-07-24, strengthened) ====
    // Simulation-only (SNM_STDP_ASSERT): plain if/$error rather than the SVA
    // `assert` keyword for portability across Icarus (-g2012) and Vivado xsim;
    // synthesis never sees this block. Enable with +define+SNM_STDP_ASSERT.
    //
    // These check INDEPENDENT quantities, not restatements of a signal's own
    // definition. (An earlier "gen_issue_valid && gen_hazard" check was a
    // tautology -- gen_issue_valid is *defined* as ...&& !gen_hazard, so it
    // could never fire and proved nothing. Replaced with the real invariants
    // that a genuine pipeline/address bug would actually violate.)
    //
    //   1. DRAIN: stdp_done never fires while any stage still holds a valid
    //      row or a write is pending commit -- keeps the next tick's history
    //      refresh from racing an in-flight write-back.
    //   2. REFRESH/WALK EXCLUSION: history refresh scan and STDP walk are
    //      never simultaneously active -- the basis for sharing the two
    //      physical history-RAM ports between the two roles.
    //   3. ROW-ADDRESS BOUNDS (independent): every in-flight stage's row
    //      address and the pending write address are within NUM_ROWS. A bad
    //      row-address computation (or the ragged dst_ptr hazard: an
    //      uninitialised terminal pointer producing an out-of-range range)
    //      trips this directly, at the stage where it first becomes visible.
    //   4. RANGE SANITY (independent, ragged-dst_ptr guard): whenever the
    //      generator issues a row, the destination's [g_syn_i, g_syn_end)
    //      range must be well-formed -- lo <= hi and hi <= SYN_CAP. An
    //      uninitialised/garbage terminal dst_ptr (the ragged-lane bug:
    //      LOCAL_D mismatch leaving dst_ptr[LOCAL_D] unwritten) yields
    //      hi < lo or hi > SYN_CAP and is caught here BEFORE it can stall the
    //      walk.
    //   5. PROGRESS: while the generator is in G_ROW with work remaining, it
    //      must EITHER issue a row this cycle OR be genuinely blocked by a
    //      live hazard -- it can never sit idle with a free pipeline and
    //      un-walked synapses (would indicate a lost-issue / deadlock bug,
    //      the failure mode independent of the hazard definition).
`ifdef SNM_STDP_ASSERT
    always @(posedge clk) begin
        if (reset_n) begin
            // 1. drain
            if (stdp_done && (p2_valid || p3_valid || p4_valid || p5_valid || stdp_wr_en))
                $error("[SNM_STDP_ASSERT @%0t] stdp_done with pipeline not drained (p2=%b p3=%b p4=%b p5=%b wr=%b)",
                    $time, p2_valid, p3_valid, p4_valid, p5_valid, stdp_wr_en);
            // 2. refresh/walk exclusion
            if (hist_refresh_active && stdp_busy)
                $error("[SNM_STDP_ASSERT @%0t] history refresh overlaps STDP walk", $time);
            // 3. row-address bounds (each stage + pending write)
            if (p2_valid && (p2_row_addr >= NUM_ROWS))
                $error("[SNM_STDP_ASSERT @%0t] p2 row_addr %0d >= NUM_ROWS %0d", $time, p2_row_addr, NUM_ROWS);
            if (p3_valid && (p3_row_addr >= NUM_ROWS))
                $error("[SNM_STDP_ASSERT @%0t] p3 row_addr %0d >= NUM_ROWS %0d", $time, p3_row_addr, NUM_ROWS);
            if (p4_valid && (p4_row_addr >= NUM_ROWS))
                $error("[SNM_STDP_ASSERT @%0t] p4 row_addr %0d >= NUM_ROWS %0d", $time, p4_row_addr, NUM_ROWS);
            if (p5_valid && (p5_row_addr >= NUM_ROWS))
                $error("[SNM_STDP_ASSERT @%0t] p5 row_addr %0d >= NUM_ROWS %0d", $time, p5_row_addr, NUM_ROWS);
            if (stdp_wr_en && (stdp_wr_row_addr >= NUM_ROWS))
                $error("[SNM_STDP_ASSERT @%0t] write row_addr %0d >= NUM_ROWS %0d", $time, stdp_wr_row_addr, NUM_ROWS);
            // 4. range sanity (ragged dst_ptr guard)
            if ((gstate == G_ROW) && (g_syn_end > SYN_CAP[DPTR_W-1:0]))
                $error("[SNM_STDP_ASSERT @%0t] dst_ptr terminal g_syn_end=%0d > SYN_CAP %0d (uninitialised/ragged dst_ptr?)",
                    $time, g_syn_end, SYN_CAP);
            // 5. progress (no lost issue / deadlock while work remains)
            if ((gstate == G_ROW) && (g_syn_i < g_syn_end) && !gen_issue_valid && !gen_hazard)
                $error("[SNM_STDP_ASSERT @%0t] generator stalled with work remaining but no hazard (lost-issue?)", $time);
        end
    end
`endif

endmodule
