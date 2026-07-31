`timescale 1ns/1ps

// UART <-> SPI bridge for the SuperNeuroMAT3 FPGA build.
//
// One host transaction = exactly 8 UART bytes in, 8 UART bytes out:
//
//   Host -> FPGA : 8 bytes, MSB byte first, forming a 64-bit SPI command word
//                  matching snm_spi_slave's frame:
//                    byte0 = [63:56] opcode
//                    byte1 = [55:48] selector
//                    byte2 = [47:40] addr_hi
//                    byte3 = [39:32] addr_lo
//                    byte4 = [31:24] data[31:24]
//                    ...
//                    byte7 = [7:0]   data[7:0]
//   FPGA -> Host : the 8 bytes captured on MISO during the same SPI frame,
//                  MSB byte first (status in byte0, response data in byte4..7).
//
// The bridge is a thin shifter: it drives spi_cs_n / spi_sclk / spi_mosi to the
// UNMODIFIED snm_spi_slave and captures spi_miso. The host-side command/poll/
// response protocol of snm_spi_slave is preserved end to end and handled by the
// Python driver (send command frame, then send a NOP 0x00.. frame to read the
// prior response).
//
// SPI is Mode 0 (CPOL=0, CPHA=0), MSB first. spi_sclk is generated at
// clk / (2*SCLK_DIV). snm_spi_slave's bare minimum is clk >= 4x spi_sclk, but
// its 3-stage input synchronizer needs more margin before MISO settles, so the
// default SCLK_DIV=8 (spi_sclk = clk/16) leaves comfortable headroom. SPI speed
// is irrelevant here anyway: the UART link is the throughput bottleneck.
// (Verified in tb_spikeengine_fpga: SCLK_DIV=2 mis-samples MISO by one bit;
//  SCLK_DIV>=4 is correct, 8 is used for margin.)

module uart_to_spi_master #(
    parameter integer CLK_HZ   = 100_000_000,
    parameter integer BAUD     = 1_000_000,
    parameter integer SCLK_DIV = 8,           // spi_sclk = clk/(2*SCLK_DIV)
    // Command byte FIFO depth. Buffers UART RX bytes so none are dropped while the
    // bridge is busy (SPI + response TX), letting the host STREAM commands back-to-back
    // instead of a full round-trip (and its USB latency) per command.
    // v3 (2026-07-11): BRAM-backed and deepened 512 -> 8192 bytes (64 -> 1024 frames,
    // 2 RAMB36). The host's sliding flow-control window is sized from this; it must be
    // >= the link's bandwidth-delay product to saturate the wire (at 4 Mbaud with the
    // Windows FTDI VCP's 16 ms latency timer, BDP ~= 800 frames -- the old 64-frame
    // window measured only 2,245 frames/s of the 50,000 frames/s wire limit).
    parameter integer CMD_FIFO_DEPTH = 8192
)(
    input  wire       clk,
    input  wire       rst,        // active-high synchronous reset

    // UART pins
    input  wire       uart_rx,
    output wire       uart_tx,

    // SPI master pins -> snm_spi_slave
    output reg        spi_cs_n,
    output reg        spi_sclk,
    output reg        spi_mosi,
    input  wire       spi_miso,

    // observability
    output wire       active       // high while a transaction is in flight
);
    // ---------------- UART instances ----------------
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx),
        .rx_valid(rx_valid), .rx_data(rx_data)
    );

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst), .tx_start(tx_start), .tx_data(tx_data),
        .tx(uart_tx), .tx_busy(tx_busy)
    );

    // ---------------- command byte FIFO ----------------
    // Every received UART byte is pushed here immediately; the bridge FSM pops from
    // it in S_RX. This decouples reception from the bridge's busy periods so bytes
    // that arrive mid-transaction are buffered rather than dropped -> the host can
    // stream commands continuously (FWFT: cmd_dout shows the oldest byte; cmd_pop
    // advances). v3: BRAM-backed (snm_bram_fifo, same FWFT contract) so the deep
    // window costs 2 RAMB36 instead of 64k flip-flops; its ~3-clk fill latency is
    // invisible next to the 250-clk UART byte time. Active-low async reset.
    wire       cmd_empty;
    wire [7:0] cmd_dout;
    wire       cmd_pop;
    snm_bram_fifo #(.W(8), .DEPTH(CMD_FIFO_DEPTH)) u_cmd_fifo (
        .clk(clk), .reset_n(~rst),
        .push(rx_valid), .din(rx_data), .full(/*ignored: host paces via responses*/),
        .pop(cmd_pop), .dout(cmd_dout), .empty(cmd_empty), .count(/*unused*/)
    );

    // ---------------- bridge state ----------------
    localparam [2:0] S_RX     = 3'd0; // collecting 8 command bytes
    localparam [2:0] S_CS_LO  = 3'd1; // assert CS, preload first MISO bit
    localparam [2:0] S_SHIFT  = 3'd2; // 64-bit SPI shift
    localparam [2:0] S_CS_HI  = 3'd3; // deassert CS
    localparam [2:0] S_TX     = 3'd4; // streaming 8 response bytes

    reg [2:0]  state;
    reg [63:0] tx_word;   // command word being shifted out on MOSI
    reg [63:0] rx_word;   // MISO captured
    reg [3:0]  byte_cnt;  // 0..8 byte counter for UART in/out
    reg [6:0]  bit_cnt;   // 0..63 SPI bit index
    reg [15:0] div_cnt;   // sclk half-period divider
    reg        sclk_phase;// 0 = first (low) half, 1 = second (high) half
    reg [23:0] idle_cnt;  // ticks since the last byte while mid-frame (resync)

    // Frame resync: if a partial frame stalls (host crash/disconnect mid-frame),
    // drop it after this many idle clocks so byte alignment self-heals instead of
    // staying lost until reconfigure. ~16 byte-times >> any in-frame inter-byte gap.
    localparam integer RESYNC_CYCLES = (CLK_HZ / BAUD) * 10 * 16;

    assign active = (state != S_RX) || (byte_cnt != 4'd0);

    // Consume one buffered command byte per clock while assembling a frame (S_RX).
    assign cmd_pop = (state == S_RX) && !cmd_empty;

    wire div_tick = (div_cnt == SCLK_DIV[15:0] - 1);

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_RX;
            tx_word    <= 64'd0;
            rx_word    <= 64'd0;
            byte_cnt   <= 4'd0;
            bit_cnt    <= 7'd0;
            div_cnt    <= 16'd0;
            sclk_phase <= 1'b0;
            spi_cs_n   <= 1'b1;
            spi_sclk   <= 1'b0;
            spi_mosi   <= 1'b0;
            tx_start   <= 1'b0;
            tx_data    <= 8'd0;
            idle_cnt   <= 24'd0;
        end else begin
            tx_start <= 1'b0;

            case (state)
                // -------- receive 8 command bytes (MSB first) --------
                S_RX: begin
                    // Pop one buffered command byte per clock (FWFT: cmd_dout is the
                    // oldest byte, cmd_pop advances). Bytes that arrived while the bridge
                    // was busy are waiting in the FIFO, so nothing is dropped.
                    if (!cmd_empty) begin
                        idle_cnt <= 24'd0;
                        tx_word  <= {tx_word[55:0], cmd_dout};
                        byte_cnt <= byte_cnt + 1'b1;
                        if (byte_cnt == 4'd7) begin
                            byte_cnt <= 4'd0;
                            state    <= S_CS_LO;
                        end
                    end else if (byte_cnt != 4'd0) begin
                        // mid-frame and the FIFO ran dry: time out and realign so a
                        // truncated frame (host crash mid-frame) self-heals.
                        if (idle_cnt >= RESYNC_CYCLES[23:0]) begin
                            byte_cnt <= 4'd0;
                            idle_cnt <= 24'd0;
                        end else begin
                            idle_cnt <= idle_cnt + 1'b1;
                        end
                    end
                end

                // -------- assert CS, present MSB on MOSI --------
                S_CS_LO: begin
                    spi_cs_n   <= 1'b0;
                    spi_sclk   <= 1'b0;
                    spi_mosi   <= tx_word[63];
                    bit_cnt    <= 7'd0;
                    div_cnt    <= 16'd0;
                    sclk_phase <= 1'b0;
                    // Give the slave's CS synchronizer a few clks before the
                    // first rising edge by holding here one divider period.
                    if (div_tick) begin
                        div_cnt <= 16'd0;
                        state   <= S_SHIFT;
                    end else begin
                        div_cnt <= div_cnt + 1'b1;
                    end
                end

                // -------- 64-bit SPI shift, Mode 0 --------
                S_SHIFT: begin
                    if (div_tick) begin
                        div_cnt <= 16'd0;
                        if (sclk_phase == 1'b0) begin
                            // rising edge: slave samples MOSI; we sample MISO.
                            spi_sclk   <= 1'b1;
                            rx_word    <= {rx_word[62:0], spi_miso};
                            sclk_phase <= 1'b1;
                        end else begin
                            // falling edge: advance to next bit / present it.
                            spi_sclk   <= 1'b0;
                            sclk_phase <= 1'b0;
                            if (bit_cnt == 7'd63) begin
                                state <= S_CS_HI;
                            end else begin
                                bit_cnt  <= bit_cnt + 1'b1;
                                spi_mosi <= tx_word[62 - bit_cnt];
                            end
                        end
                    end else begin
                        div_cnt <= div_cnt + 1'b1;
                    end
                end

                // -------- deassert CS --------
                S_CS_HI: begin
                    spi_sclk <= 1'b0;
                    if (div_tick) begin
                        div_cnt  <= 16'd0;
                        spi_cs_n <= 1'b1;
                        byte_cnt <= 4'd0;
                        state    <= S_TX;
                    end else begin
                        div_cnt <= div_cnt + 1'b1;
                    end
                end

                // -------- stream 8 response bytes (MSB first) --------
                S_TX: begin
                    if (!tx_busy && !tx_start) begin
                        tx_data  <= rx_word[63 -: 8];
                        rx_word  <= {rx_word[55:0], 8'd0};
                        tx_start <= 1'b1;
                        byte_cnt <= byte_cnt + 1'b1;
                        if (byte_cnt == 4'd7) begin
                            byte_cnt <= 4'd0;
                            state    <= S_RX;
                        end
                    end
                end

                default: state <= S_RX;
            endcase
        end
    end

endmodule
