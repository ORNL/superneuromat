`timescale 1us/1ns

// Board top for the STDP-capable parallel-lane engine (2026-07-21, npu_stdp_dev
// experiment). New sibling of snm_infer_top.v (that file stays byte-for-byte
// untouched) -- IDENTICAL pin boundary and SPI front-end, instantiates
// snm_infer_engine_stdp instead of snm_infer_engine. Existing UART/SPI bridge,
// XDC, and host runtime reach this design with no pin-level changes; only the
// IL_* config map (now including IL_STDP_TABLE/IL_STDP_ENABLE) differs, and
// that lives entirely inside snm_infer_cmd_ctrl_stdp.
module snm_infer_top_stdp #(
    parameter integer N_MAX     = 1024,
    parameter integer NUM_LANES = 16,
    parameter integer DATA_W    = 16,
    parameter integer WEIGHT_W  = 8,
    parameter integer ACCUM_W   = 32,
    parameter integer REF_W     = 8,
    parameter integer STDP_WINDOW = 5,
    parameter integer SPIKE_MON_BASE = 0,   // LED[0] = this neuron id (see cmd_ctrl)
    // forwarded from the board top to the engine (see the board top's comment):
    // must be threaded here or the engine falls back to dense LOCAL_D*N_MAX.
    parameter integer SYN_CAP_PER_LANE = ((N_MAX + NUM_LANES - 1) / NUM_LANES) * N_MAX
)(
    input  wire clk,
    input  wire reset_n,
    input  wire spi_sclk,
    input  wire spi_cs_n,
    input  wire spi_mosi,
    output wire spi_miso,
    output wire irq,
    output wire busy,
    output wire [15:0] spike_mon
);
    wire        cmd_valid, cmd_ready;
    wire [7:0]  cmd_opcode, cmd_sel;
    wire [15:0] cmd_addr;
    wire [31:0] cmd_wdata;
    wire        rsp_valid, rsp_ready;
    wire [7:0]  rsp_status;
    wire [31:0] rsp_rdata;

    wire        spi_busy;
    wire        response_pending;
    wire        packet_error;

    snm_spi_slave u_spi_slave (
        .clk(clk), .reset_n(reset_n),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n), .spi_mosi(spi_mosi), .spi_miso(spi_miso),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_opcode(cmd_opcode),
        .cmd_sel(cmd_sel), .cmd_addr(cmd_addr), .cmd_wdata(cmd_wdata),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status),
        .rsp_rdata(rsp_rdata),
        .busy(spi_busy), .response_pending(response_pending),
        .packet_error_clear(1'b0), .packet_error(packet_error)
    );

    wire engine_busy, engine_irq;

    snm_infer_engine_stdp #(
        .N_MAX(N_MAX), .NUM_LANES(NUM_LANES), .DATA_W(DATA_W),
        .WEIGHT_W(WEIGHT_W), .ACCUM_W(ACCUM_W), .REF_W(REF_W), .STDP_WINDOW(STDP_WINDOW),
        .SPIKE_MON_BASE(SPIKE_MON_BASE),
        .SYN_CAP_PER_LANE(SYN_CAP_PER_LANE)
    ) u_engine (
        .clk(clk), .reset_n(reset_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_opcode(cmd_opcode),
        .cmd_sel(cmd_sel), .cmd_addr(cmd_addr), .cmd_wdata(cmd_wdata),
        .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status),
        .rsp_rdata(rsp_rdata),
        .busy(engine_busy), .irq(engine_irq), .spike_mon(spike_mon)
    );

    assign busy = spi_busy || engine_busy;
    assign irq  = engine_irq || response_pending || packet_error;
endmodule
