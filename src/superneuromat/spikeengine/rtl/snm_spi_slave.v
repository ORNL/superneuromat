`timescale 1us/1ns

// SuperNeuroMAT3 SPI slave wrapper (G6).
//
// SPI mode 0, CPOL=0, CPHA=0.  The system clock must run at least 4x faster
// than spi_sclk so the synchronized edge detector sees every SPI edge.
//
// Packet format from the host is 64 bits, MSB first:
//   [63:56] opcode
//   [55:48] selector
//   [47:32] address
//   [31:0]  data
//
// Response packets are also shifted MSB first:
//   [63:56] response status
//   [55:32] reserved zero
//   [31:0]  response data
//
// Since a command can take many system clocks, the response to packet N is read
// by a later SPI transaction.  A zero MOSI packet is treated as a NOP/poll and
// is not forwarded to the command controller.  response_pending asserts when a
// command response is ready and clears when the host starts the next SPI frame.

module snm_spi_slave (
    clk,
    reset_n,
    spi_sclk,
    spi_cs_n,
    spi_mosi,
    spi_miso,
    cmd_valid,
    cmd_ready,
    cmd_opcode,
    cmd_sel,
    cmd_addr,
    cmd_wdata,
    rsp_valid,
    rsp_ready,
    rsp_status,
    rsp_rdata,
    busy,
    response_pending,
    packet_error_clear,
    packet_error
);
    localparam [7:0] ST_OK   = 8'h00;
    localparam [7:0] ST_BUSY = 8'h01;

    localparam [1:0] C_IDLE     = 2'd0;
    localparam [1:0] C_SEND_CMD = 2'd1;
    localparam [1:0] C_WAIT_RSP = 2'd2;

    input  wire        clk;
    input  wire        reset_n;

    input  wire        spi_sclk;
    input  wire        spi_cs_n;
    input  wire        spi_mosi;
    output reg         spi_miso;

    output reg         cmd_valid;
    input  wire        cmd_ready;
    output reg  [7:0]  cmd_opcode;
    output reg  [7:0]  cmd_sel;
    output reg  [15:0] cmd_addr;
    output reg  [31:0] cmd_wdata;

    input  wire        rsp_valid;
    output wire        rsp_ready;
    input  wire [7:0]  rsp_status;
    input  wire [31:0] rsp_rdata;

    output wire        busy;
    output reg         response_pending;
    input  wire        packet_error_clear;
    output reg         packet_error;

    reg [1:0] cmd_state;

    reg [2:0] sclk_sync;
    reg [2:0] cs_sync;
    reg [1:0] mosi_sync;

    reg [63:0] rx_shift;
    reg [63:0] tx_shift;
    reg [63:0] response_packet;
    reg [6:0]  bit_count;
    reg        spi_active;
    reg        packet_done;
    reg [63:0] packet_latched;

    wire cs_active  = ~cs_sync[2];
    wire cs_fall    =  cs_sync[2] & ~cs_sync[1];
    wire cs_rise    = ~cs_sync[2] &  cs_sync[1];
    wire sclk_rise  = cs_active & ~sclk_sync[2] &  sclk_sync[1];
    wire sclk_fall  = cs_active &  sclk_sync[2] & ~sclk_sync[1];
    wire [63:0] rx_next = {rx_shift[62:0], mosi_sync[1]};

    assign rsp_ready = (cmd_state == C_WAIT_RSP);
    assign busy = (cmd_state != C_IDLE);

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            sclk_sync <= 3'b000;
            cs_sync <= 3'b111;
            mosi_sync <= 2'b00;
        end else begin
            sclk_sync <= {sclk_sync[1:0], spi_sclk};
            cs_sync <= {cs_sync[1:0], spi_cs_n};
            mosi_sync <= {mosi_sync[0], spi_mosi};
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            spi_miso <= 1'b0;
            rx_shift <= 64'd0;
            tx_shift <= 64'd0;
            bit_count <= 7'd0;
            spi_active <= 1'b0;
            packet_done <= 1'b0;
            packet_latched <= 64'd0;
            packet_error <= 1'b0;
        end else begin
            packet_done <= 1'b0;

            if (packet_error_clear)
                packet_error <= 1'b0;

            if (cs_fall) begin
                spi_active <= 1'b1;
                bit_count <= 7'd0;
                rx_shift <= 64'd0;
                // response_packet is owned (reset + written) solely by the
                // command-FSM block below; this block only reads it.
                tx_shift <= response_packet;
                spi_miso <= response_packet[63];
            end

            if (sclk_rise) begin
                rx_shift <= rx_next;
                if (bit_count == 7'd63) begin
                    packet_done <= 1'b1;
                    packet_latched <= rx_next;
                    bit_count <= 7'd64;
                end else if (bit_count < 7'd64) begin
                    bit_count <= bit_count + 1'b1;
                end
            end

            if (sclk_fall && (bit_count < 7'd64)) begin
                tx_shift <= {tx_shift[62:0], 1'b0};
                spi_miso <= tx_shift[62];
            end

            if (cs_rise) begin
                if (spi_active && (bit_count != 7'd0) && (bit_count != 7'd64))
                    packet_error <= 1'b1;
                spi_active <= 1'b0;
                bit_count <= 7'd0;
            end
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cmd_state <= C_IDLE;
            cmd_valid <= 1'b0;
            cmd_opcode <= 8'd0;
            cmd_sel <= 8'd0;
            cmd_addr <= 16'd0;
            cmd_wdata <= 32'd0;
            response_pending <= 1'b0;
            response_packet <= 64'd0;
        end else begin
            cmd_valid <= 1'b0;

            if (cs_fall && response_pending)
                response_pending <= 1'b0;

            if (packet_done) begin
                if (packet_latched == 64'd0) begin
                    // NOP/poll transaction; leave the last response readable.
                end else if (cmd_state != C_IDLE) begin
                    response_packet <= {ST_BUSY, 24'd0, 32'd0};
                    response_pending <= 1'b1;
                end else begin
                    cmd_opcode <= packet_latched[63:56];
                    cmd_sel <= packet_latched[55:48];
                    cmd_addr <= packet_latched[47:32];
                    cmd_wdata <= packet_latched[31:0];
                    response_pending <= 1'b0;
                    cmd_state <= C_SEND_CMD;
                end
            end

            case (cmd_state)
                C_IDLE: begin
                    cmd_valid <= 1'b0;
                end

                C_SEND_CMD: begin
                    cmd_valid <= 1'b1;
                    if (cmd_ready)
                        cmd_state <= C_WAIT_RSP;
                end

                C_WAIT_RSP: begin
                    if (rsp_valid) begin
                        response_packet <= {rsp_status, 24'd0, rsp_rdata};
                        response_pending <= 1'b1;
                        cmd_state <= C_IDLE;
                    end
                end

                default: begin
                    cmd_state <= C_IDLE;
                end
            endcase
        end
    end

endmodule
