`timescale 1ns/1ps

// Simple 8N1 UART transmitter.
//
// Assert tx_start for one clk with tx_data valid. tx_busy stays high until the
// stop bit completes. The line idles high. One start bit, 8 data bits LSB-first,
// one stop bit.

module uart_tx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 1_000_000
)(
    input  wire       clk,
    input  wire       rst,        // active-high synchronous reset
    input  wire       tx_start,   // 1-clk strobe to begin a byte
    input  wire [7:0] tx_data,
    output reg        tx,         // serial output
    output reg        tx_busy
);
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    localparam [1:0] S_IDLE  = 2'd0;
    localparam [1:0] S_START = 2'd1;
    localparam [1:0] S_DATA  = 2'd2;
    localparam [1:0] S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            shifter <= 8'd0;
            tx      <= 1'b1;
            tx_busy <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (tx_start) begin
                        shifter <= tx_data;
                        tx_busy <= 1'b1;
                        tx      <= 1'b0;      // start bit
                        state   <= S_START;
                    end
                end

                S_START: begin
                    if (clk_cnt == CLKS_PER_BIT[15:0] - 1) begin
                        clk_cnt <= 16'd0;
                        tx      <= shifter[0];
                        state   <= S_DATA;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT[15:0] - 1) begin
                        clk_cnt <= 16'd0;
                        if (bit_idx == 3'd7) begin
                            tx    <= 1'b1;    // stop bit
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1'b1;
                            shifter <= {1'b0, shifter[7:1]};
                            tx      <= shifter[1];
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT[15:0] - 1) begin
                        clk_cnt <= 16'd0;
                        tx_busy <= 1'b0;
                        state   <= S_IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
