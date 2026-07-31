`timescale 1ns/1ps

// Simple 8N1 UART receiver.
//
// One start bit (0), 8 data bits LSB-first, one stop bit (1). The line idles
// high. Oversamples the incoming bit by sampling at the mid-point of each bit
// period. Output strobe rx_valid pulses high for one clk when rx_data is valid.
//
// CLKS_PER_BIT = CLK_HZ / BAUD must be >= 8 for reliable mid-bit sampling.

module uart_rx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 1_000_000
)(
    input  wire       clk,
    input  wire       rst,        // active-high synchronous reset
    input  wire       rx,         // serial input (already synchronized externally)
    output reg        rx_valid,   // 1-clk strobe when rx_data is ready
    output reg  [7:0] rx_data
);
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer HALF_BIT      = CLKS_PER_BIT / 2;

    localparam [1:0] S_IDLE  = 2'd0;
    localparam [1:0] S_START = 2'd1;
    localparam [1:0] S_DATA  = 2'd2;
    localparam [1:0] S_STOP  = 2'd3;

    // Double-flop synchronizer for the asynchronous rx line. ASYNC_REG keeps the
    // two flops in the same slice for best metastability MTBF.
    (* ASYNC_REG = "TRUE" *) reg rx_meta, rx_sync;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    always @(posedge clk) begin
        if (rst) begin
            rx_meta  <= 1'b1;
            rx_sync  <= 1'b1;
        end else begin
            rx_meta  <= rx;
            rx_sync  <= rx_meta;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            clk_cnt  <= 16'd0;
            bit_idx  <= 3'd0;
            shifter  <= 8'd0;
            rx_valid <= 1'b0;
            rx_data  <= 8'd0;
        end else begin
            rx_valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (rx_sync == 1'b0) // start bit edge
                        state <= S_START;
                end

                S_START: begin
                    // Sample at mid start bit to confirm it is still low.
                    if (clk_cnt == HALF_BIT[15:0]) begin
                        if (rx_sync == 1'b0) begin
                            clk_cnt <= 16'd0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE; // false start
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT[15:0] - 1) begin
                        clk_cnt        <= 16'd0;
                        shifter        <= {rx_sync, shifter[7:1]}; // LSB first
                        if (bit_idx == 3'd7)
                            state <= S_STOP;
                        else
                            bit_idx <= bit_idx + 1'b1;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT[15:0] - 1) begin
                        rx_data  <= shifter;
                        rx_valid <= 1'b1;
                        clk_cnt  <= 16'd0;
                        state    <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
