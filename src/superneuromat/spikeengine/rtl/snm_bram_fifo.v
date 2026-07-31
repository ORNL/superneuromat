`timescale 1us/1ns

// SuperNeuroMAT3 generic BRAM-backed FIFO (protocol-v3 batch, 2026-07-11).
//
// Drop-in port-compatible with snm_reg_fifo (same push/din/full/pop/dout/empty/
// count contract, same first-word-fall-through read style), but stores entries in
// an inferred block RAM instead of flip-flops, so DEEP queues are cheap:
//   - UART command byte FIFO   (W=8,  DEPTH=8192  -> 2 RAMB36)   was 512 B of FFs
//   - input event FIFO         (W=EV_W, DEPTH=N_MAX)             was 64 deep in FFs
//
// FWFT with a BRAM requires hiding the RAM's 1-cycle synchronous read latency:
// a 2-entry register output stage (out0 = oldest = dout, out1 = next) is kept
// topped up by a prefetch read.  `dout` therefore continuously shows the oldest
// entry exactly like snm_reg_fifo; `pop` advances.  The only visible difference
// is fill latency: an entry pushed into an EMPTY fifo appears on `dout` ~3 clks
// later (BRAM write -> prefetch read -> output stage).  Both users tolerate this:
// the UART bridge pops at most 1 byte/clk while bytes ARRIVE 250 clks apart
// (4 Mbaud at 100 MHz), and the event-replay FSM pops at most every 3rd clk.
//
// Flow control:
//   - push when full  is ignored (no overflow, no data loss)
//   - pop  when empty is ignored (no underflow)
//   - simultaneous push+pop supported.  Unlike snm_reg_fifo, a push at FULL is
//     dropped even with a coincident pop (the freed slot is in the output stage,
//     not the RAM).  No user pushes at full: the event path checks `full` first,
//     the vector-expand path checks `count`, and the UART host paces below the
//     FIFO size via the response stream.
//
// Reset clears pointers/counters/output stage only -- BRAM contents are not
// cleared (irrelevant: entries are only read back after being written).

module snm_bram_fifo #(
    parameter integer W     = 8,
    parameter integer DEPTH = 1024
)(
    clk,
    reset_n,
    push,
    din,
    full,
    pop,
    dout,
    empty,
    count
);
    localparam integer PTR_W = (DEPTH < 2) ? 1 : $clog2(DEPTH);
    localparam integer CNT_W = $clog2(DEPTH + 1);
    localparam [31:0]      DEPTH_M1 = DEPTH - 1;
    localparam [PTR_W-1:0] LAST_PTR = DEPTH_M1[PTR_W-1:0];

    input  wire             clk;
    input  wire             reset_n;
    input  wire             push;
    input  wire [W-1:0]     din;
    output wire             full;
    input  wire             pop;
    output wire [W-1:0]     dout;
    output wire             empty;
    output wire [CNT_W-1:0] count;

    (* ram_style = "block" *) reg [W-1:0] mem [0:DEPTH-1];

    reg [PTR_W-1:0] wr_ptr;
    reg [PTR_W-1:0] rd_ptr;
    reg [CNT_W-1:0] cnt;        // TOTAL occupancy: RAM + read-in-flight + output stage
    reg [CNT_W-1:0] mem_cnt;    // entries still in the RAM (not yet prefetched)

    // FWFT output stage: out0 is the visible head, out1 the entry behind it.
    reg [W-1:0] out0;
    reg [W-1:0] out1;
    reg [1:0]   out_cnt;
    reg         rd_pending;     // a RAM read was issued last clk; rdata is valid NOW
    reg [W-1:0] rdata;

    assign dout  = out0;
    assign empty = (out_cnt == 2'd0);
    assign full  = (cnt == DEPTH[CNT_W-1:0]);
    assign count = cnt;

    wire do_pop  = pop && (out_cnt != 2'd0);
    // Push at full is dropped (see header) -- eliminates the same-address BRAM
    // write/read collision corner outright (rd_ptr == wr_ptr only at mem empty/full).
    wire do_push = push && !full;

    // Prefetch: issue a RAM read when entries remain in RAM and the output stage
    // (after this clk's pop, plus any read already in flight) has room for it.
    // Invariant kept: out_cnt + rd_pending <= 2 at every clk.
    wire [1:0] out_after_pop = out_cnt - {1'b0, do_pop};
    wire       rd_issue      = (mem_cnt != {CNT_W{1'b0}}) &&
                               (({1'b0, out_after_pop} + {1'b0, 1'b0, rd_pending}) < 3'd2);

    // RAM ports live in their OWN reset-free block: a memory touched inside an
    // async-reset process cannot map to BRAM (BRAM contents are not resettable),
    // and Vivado would fall back to registers -- fatally, at these depths.
    always @(posedge clk) begin
        if (do_push)
            mem[wr_ptr] <= din;
        if (rd_issue)
            rdata <= mem[rd_ptr];   // synchronous read; lands next clk (rd_pending)
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            wr_ptr     <= {PTR_W{1'b0}};
            rd_ptr     <= {PTR_W{1'b0}};
            cnt        <= {CNT_W{1'b0}};
            mem_cnt    <= {CNT_W{1'b0}};
            out0       <= {W{1'b0}};
            out1       <= {W{1'b0}};
            out_cnt    <= 2'd0;
            rd_pending <= 1'b0;
        end else begin
            // pointer updates (the RAM accesses themselves are in the block above)
            if (do_push) begin
                wr_ptr <= (wr_ptr == LAST_PTR) ? {PTR_W{1'b0}} : (wr_ptr + 1'b1);
            end
            if (rd_issue) begin
                rd_ptr <= (rd_ptr == LAST_PTR) ? {PTR_W{1'b0}} : (rd_ptr + 1'b1);
            end
            rd_pending <= rd_issue;

            // RAM occupancy: gains a push, loses a prefetch
            mem_cnt <= mem_cnt + {{(CNT_W-1){1'b0}}, do_push}
                               - {{(CNT_W-1){1'b0}}, rd_issue};

            // Output stage: place last clk's prefetched data / advance on pop.
            case ({rd_pending, do_pop})
                2'b01: begin                       // pop only
                    out0    <= out1;
                    out_cnt <= out_cnt - 2'd1;
                end
                2'b10: begin                       // place only (out_cnt is 0 or 1 here)
                    if (out_cnt == 2'd0) out0 <= rdata;
                    else                 out1 <= rdata;
                    out_cnt <= out_cnt + 2'd1;
                end
                2'b11: begin                       // pop + place (out_cnt is 1 or 2)
                    if (out_cnt == 2'd1) begin
                        out0 <= rdata;             // popped the only entry; refill head
                    end else begin
                        out0 <= out1;              // shift, refill tail
                        out1 <= rdata;
                    end
                    // out_cnt unchanged
                end
                default: ;                         // 2'b00: nothing
            endcase

            // Total occupancy
            cnt <= cnt + {{(CNT_W-1){1'b0}}, do_push}
                       - {{(CNT_W-1){1'b0}}, do_pop};
        end
    end

endmodule
