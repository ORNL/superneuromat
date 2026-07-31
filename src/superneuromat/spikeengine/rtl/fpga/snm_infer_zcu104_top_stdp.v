`timescale 1ns/1ps

// STDP-capable parallel-lane engine -- ZCU104 board wrapper (2026-07-22).
//
// New sibling of snm_infer_zcu104_top.v (that file stays byte-for-byte
// untouched): identical port list, pin names, clock wrapper (IBUFDS 125 MHz
// differential -> MMCME4_BASE -> 100 MHz), and reset handling -- the ONLY
// change is instantiating snm_infer_fpga_top_stdp (the STDP-capable board
// top) instead of snm_infer_fpga_top. Because the port list is byte-for-byte
// identical to the module the existing constraints/zcu104.xdc already
// targets (CLK_125_P/N, GPIO_PB_SW0, uart_rx, uart_tx, led[3:0]), that XDC
// applies to THIS top with NO edits.
//
// Config: N_MAX=256, NUM_LANES=8, STDP_WINDOW=5 -- matched to the
// Basys3 cap256x8 config validated this session (real hardware, 8/8 match)
// so results are directly comparable. ZCU104's UltraScale+ fabric has
// substantially more BRAM/LUT headroom than Basys3's xc7a35t, so this is
// expected to close with the largest margin of the three boards.
module snm_infer_zcu104_top_stdp #(
    parameter integer SCLK_DIV    = 8,
    parameter integer N_MAX       = 256,
    parameter integer NUM_LANES   = 8,
    parameter integer STDP_WINDOW = 5,
    // 2026-07-29: same fix as the sp701 top -- these were not declared/forwarded,
    // so build-script -generic WEIGHT_W/DATA_W/SPIKE_MON_BASE/SYN_CAP_PER_LANE on
    // this synth top were silently dropped (engine fell back to W8/D16/mon0/dense).
    // Declared + forwarded now; defaults match the generic fpga top.
    parameter integer WEIGHT_W    = 8,
    parameter integer DATA_W      = 16,
    parameter integer SPIKE_MON_BASE = 0,
    parameter integer SYN_CAP_PER_LANE = ((N_MAX + NUM_LANES - 1) / NUM_LANES) * N_MAX
) (
    input  wire       CLK_125_P,
    input  wire       CLK_125_N,
    input  wire       GPIO_PB_SW0,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [3:0] led
);
    wire clk125;
    IBUFDS #(
        .DQS_BIAS("FALSE")
    ) u_ibufds (.O(clk125), .I(CLK_125_P), .IB(CLK_125_N));

    wire clkfb, clk100_unbuf, mmcm_locked;
    MMCME4_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKIN1_PERIOD(8.000),
        .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(8.000),
        .CLKOUT0_DIVIDE_F(10.000),
        .STARTUP_WAIT("FALSE")
    ) u_mmcm (
        .CLKIN1(clk125),
        .CLKFBIN(clkfb),
        .CLKFBOUT(clkfb),
        .CLKOUT0(clk100_unbuf),
        .LOCKED(mmcm_locked),
        .PWRDWN(1'b0),
        .RST(1'b0)
    );

    wire rst_btn = GPIO_PB_SW0 | (~mmcm_locked);

    wire [15:0] led16;
    snm_infer_fpga_top_stdp #(
        .SCLK_DIV(SCLK_DIV), .N_MAX(N_MAX), .NUM_LANES(NUM_LANES), .STDP_WINDOW(STDP_WINDOW),
        .WEIGHT_W(WEIGHT_W), .DATA_W(DATA_W), .SPIKE_MON_BASE(SPIKE_MON_BASE),
        .SYN_CAP_PER_LANE(SYN_CAP_PER_LANE)
    ) u_top (
        .clk     (clk100_unbuf),
        .rst_btn (rst_btn),
        .uart_rx (uart_rx),
        .uart_tx (uart_tx),
        .led     (led16)
    );

    assign led = led16[3:0];
endmodule
