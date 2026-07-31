`timescale 1ns/1ps

// STDP-capable parallel-lane engine -- SP701 board wrapper (2026-07-22).
//
// New sibling of snm_infer_sp701_top.v (that file stays byte-for-byte
// untouched): identical port list, pin names, clock wrapper (IBUFDS 200 MHz
// differential -> MMCME2_BASE -> 100 MHz), and reset handling -- the ONLY
// change is instantiating snm_infer_fpga_top_stdp (the STDP-capable board
// top) instead of snm_infer_fpga_top. Same reasoning as the inference-only
// SP701 top's own comment: because the port list is byte-for-byte identical,
// the existing constraints/sp701.xdc applies to THIS top with NO edits.
//
// Config: N_MAX=256, NUM_LANES=8, STDP_WINDOW=5 -- matched to the
// Basys3 cap256x8 config validated this session (real hardware, 8/8 match,
// WNS +0.041..+0.335ns across rebuilds) so the two boards' STDP results are
// directly comparable. SP701's real budget (120 RAMB36 tiles, no URAM) has
// much more headroom than Basys3's 50 RAMB36 tiles, so this is expected to
// close with larger margin -- to be confirmed by a real build.
module snm_infer_sp701_top_stdp #(
    parameter integer SCLK_DIV    = 8,
    parameter integer N_MAX       = 256,
    parameter integer NUM_LANES   = 8,
    parameter integer STDP_WINDOW = 5,
    // 2026-07-29: these were NOT declared/forwarded before, so build-script
    // -generic WEIGHT_W/DATA_W/SPIKE_MON_BASE/SYN_CAP_PER_LANE on THIS synth top
    // were silently dropped and the engine fell back to W8/D16/mon0/dense (the
    // "wide16" sp701 bitstream was in fact an 8-bit build -- confirmed in the
    // build log: "WEIGHT_W bound to: 8"). Declared + forwarded now; defaults
    // match the generic fpga top so an un-overridden build is unchanged.
    parameter integer WEIGHT_W    = 8,
    parameter integer DATA_W      = 16,
    parameter integer SPIKE_MON_BASE = 0,
    parameter integer SYN_CAP_PER_LANE = ((N_MAX + NUM_LANES - 1) / NUM_LANES) * N_MAX
) (
    input  wire       SYSCLK_P,   // 200 MHz LVDS system clock (+)
    input  wire       SYSCLK_N,   // 200 MHz LVDS system clock (-)
    input  wire       CPU_RESET,  // active-HIGH push-button (official SP701 board file)
    input  wire       uart_rx,    // host -> FPGA
    output wire       uart_tx,    // FPGA -> host
    output wire [7:0] led         // GPIO_LED0..7 (subset of the engine's 16 spike LEDs)
);

    // ---- 1) differential 200 MHz -> single-ended ----
    wire clk200;
    IBUFDS #(
        .DIFF_TERM("FALSE"), .IBUF_LOW_PWR("FALSE"), .IOSTANDARD("LVDS_25")
    ) u_ibufds (.O(clk200), .I(SYSCLK_P), .IB(SYSCLK_N));

    // ---- 2) MMCM 200 MHz -> 100 MHz (VCO = 200*5 = 1000 MHz; 1000/10 = 100 MHz) ----
    wire clkfb, clk100_unbuf, mmcm_locked;
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKIN1_PERIOD(5.000),        // 200 MHz input
        .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(5.000),      // VCO = 1000 MHz
        .CLKOUT0_DIVIDE_F(10.000),    // 100 MHz core clock
        .STARTUP_WAIT("FALSE")
    ) u_mmcm (
        .CLKIN1(clk200),
        .CLKFBIN(clkfb),
        .CLKFBOUT(clkfb),             // internal feedback (no BUFG needed for internal clk)
        .CLKOUT0(clk100_unbuf),       // UNBUFFERED -> generic top's internal BUFG buffers it
        .LOCKED(mmcm_locked),
        .PWRDWN(1'b0),
        .RST(1'b0)
    );

    // ---- 3) reset: SP701 RESET button is active-high; also hold until MMCM locks ----
    wire rst_btn = CPU_RESET | (~mmcm_locked);

    // ---- 4) instantiate the generic PL-only STDP-capable top (its BUFG buffers clk100_unbuf) ----
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

    assign led = led16[7:0];   // SP701 has 8 user LEDs; upper spike LEDs unused

endmodule
