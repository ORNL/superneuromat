`timescale 1ns/1ps

// STDP-capable parallel-lane engine -- generic PL-only FPGA board top
// (2026-07-21, npu_stdp_dev experiment). New sibling of snm_infer_fpga_top.v
// (that file stays byte-for-byte untouched): the ONLY change is instantiating
// snm_infer_top_stdp instead of snm_infer_top -- same UART<->SPI bridge, same
// reset synchronizer, same 100 MHz board-oscillator-direct clocking, same
// port list (clk/rst_btn/uart_rx/uart_tx/led), so it drops behind any XDC
// already written for the inference-only top with no pin changes.
module snm_infer_fpga_top_stdp #(
    parameter integer CLK_HZ_IN = 100_000_000,
    parameter integer BAUD      = 4_000_000,
    parameter integer SCLK_DIV  = 8,
    parameter integer N_MAX     = 1024,
    parameter integer NUM_LANES = 16,
    parameter integer STDP_WINDOW = 5,
    // 2026-07-27: WEIGHT_W/DATA_W now exposed as board-top generics (were only
    // settable deeper, at snm_infer_top_stdp). Defaults keep the original
    // 8-bit weight / 16-bit membrane build byte-for-byte identical; the wider
    // fixed-point digits build (WEIGHT_W=16, DATA_W=24) selects them via
    // -generic. WEIGHT_W widens the synapse-weight datapath so sub-LSB STDP
    // taps (0.002 etc.) survive per-update quantization; DATA_W must grow with
    // it because the training recipe's dynamic range (tiny taps alongside
    // threshold 99 / teacher 100) needs membrane headroom at FRAC_BITS=11
    // (threshold 99 -> ~203k raw, ~18-bit) -- see the wider-fixed-point writeup.
    parameter integer WEIGHT_W  = 8,
    parameter integer DATA_W    = 16,
    parameter integer SPIKE_MON_BASE = 0,   // LED[0]=this neuron id; set to first OUTPUT neuron to show classes
    // SYN_CAP_PER_LANE forwarded from the synth top down to the engine
    // (2026-07-29). MUST be threaded here and in snm_infer_top_stdp -- a
    // -generic on this board top is otherwise silently dropped, and the engine
    // falls back to dense LOCAL_D*N_MAX sizing (real N=512/cap=512 basys3 build
    // hit 200 BRAM tiles because of exactly that gap). Default expression is
    // IDENTICAL to the engine default, so un-overridden builds are unchanged.
    parameter integer SYN_CAP_PER_LANE = ((N_MAX + NUM_LANES - 1) / NUM_LANES) * N_MAX
)(
    input  wire       clk,
    input  wire       rst_btn,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [15:0] led
);
    localparam integer SYS_CLK_HZ = CLK_HZ_IN;

    wire sys_clk;
`ifdef SYNTHESIS
    BUFG u_sys_bufg (.I(clk), .O(sys_clk));
`else
    assign sys_clk = clk;
`endif

    (* ASYNC_REG = "TRUE" *) reg [1:0] rst_sync;
    always @(posedge sys_clk or posedge rst_btn) begin
        if (rst_btn) rst_sync <= 2'b11;
        else         rst_sync <= {rst_sync[0], 1'b0};
    end
    wire rst     = rst_sync[1];
    wire reset_n = ~rst;

    wire spi_sclk, spi_cs_n, spi_mosi, spi_miso;
    wire bridge_active;

    uart_to_spi_master #(
        .CLK_HZ(SYS_CLK_HZ), .BAUD(BAUD), .SCLK_DIV(SCLK_DIV)
    ) u_bridge (
        .clk(sys_clk), .rst(rst),
        .uart_rx(uart_rx), .uart_tx(uart_tx),
        .spi_cs_n(spi_cs_n), .spi_sclk(spi_sclk), .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .active(bridge_active)
    );

    wire chip_irq, chip_busy;
    wire [15:0] chip_spike_mon;

    snm_infer_top_stdp #(
        .N_MAX(N_MAX), .NUM_LANES(NUM_LANES), .STDP_WINDOW(STDP_WINDOW),
        .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W),
        .SPIKE_MON_BASE(SPIKE_MON_BASE),
        .SYN_CAP_PER_LANE(SYN_CAP_PER_LANE)
    ) u_chip (
        .clk(sys_clk),
        .reset_n(reset_n),
        .spi_sclk(spi_sclk),
        .spi_cs_n(spi_cs_n),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .irq(chip_irq),
        .busy(chip_busy),
        .spike_mon(chip_spike_mon)
    );

    wire _unused_status = chip_irq | chip_busy | bridge_active;
    assign led = chip_spike_mon;
endmodule
