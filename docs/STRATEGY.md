# MTFTR Strategy Deep Dive

## Overview

MTFTR (Multi-Timeframe Trend Rider) is the core strategy family used in Aegis.

It exists in two execution surfaces:

- MQL5 Expert Advisors for MT5 runtime execution
- Python strategy mirror for analysis and strategy-engine workflows

Primary source files:

- `src/strategies/MTFTR_EA_v2.mq5`
- `src/strategies/MTFTR_EA_v4.0.mq5`
- `src/strategies/mtftr.py`

## Strategy Thesis

MTFTR is a trend-following framework that combines:

- higher-timeframe trend bias
- mid-timeframe confirmation
- lower-timeframe trigger precision
- strict risk and staged exits

The strategy is designed around controlled downside and asymmetric upside via partials and trailing.

## Timeframe Cascade

### EA v2

- 4H: trend state
- 1H: confirmation and trailing context
- 15M: entry trigger

### EA v4

- D1: master bias filter
- 4H: trend context
- 1H: confirmation plus ADX trend-strength filter
- 15M: entry precision

### Python mirror

- H4 + H1 + M1 currently used in `MTFTRStrategy.get_required_timeframes()`
- Logic remains multi-timeframe trend + confirmation + entry style, but timeframe naming differs from EA defaults

## Entry Methods

### Method A: EMA Bounce

- Pullback into EMA zone on entry timeframe
- Reversal candle or continuation confirmation
- RSI guardrails applied

### Method B: Structure Break

- Swing structure break in trend direction
- Additional confirmation from higher-timeframe alignment

### Method C (v4): Fibonacci Pullback

- Retrace to configured fib zone (50% / 61.8%)
- Filtered by RSI and trend alignment

## Indicator Stack

MTFTR uses a consistent indicator family:

- EMA 200 / 50 / 21 (timeframe-specific context)
- Hull MA 55 / 34 / 21 variants
- RSI(14)
- ATR(14)
- swing lookup windows
- ADX(14) in v4 for trend quality gating

## Exit State Machine

### Python mirror default (config-driven)

- TP1 at configurable `tp1_rr`
- TP2 at configurable `tp2_rr`
- partial closures controlled by percentages
- trailing remainder

### EA v2 state flow

- INITIAL
- BE_SET
- PARTIAL1
- PARTIAL2

### EA v4 state flow

- INITIAL
- TP1_HIT
- TP2_HIT
- TP3_TRAIL

## v2 vs v4 Practical Differences

| Area | v2 | v4 |
|---|---|---|
| Risk per trade default | 1.0% | 2.0% |
| Open trades max | 2 | 3 |
| Daily trades max | 5 | 5 |
| Consecutive loss pause | 2 | 4 |
| Daily DD default | 3% | 4% |
| Weekly DD default | 6% | 8% |
| Bias stack | 4H + 1H + 15M | D1 + 4H + 1H + 15M |
| ADX filter | No | Yes |
| TP model | 1:4-1:6 multi-stage variant with BE and partials | 3-tier target model (1.5R, 3R, 6R runner) |
| Session filter | Optional, default off | Configured window 7-16 |
| Monthly withdrawal simulation | No | Yes |

## Python Mirror Notes

`src/strategies/mtftr.py` defines:

- `MTFTRConfig` for indicator, filter, and risk parameters
- `MTFTRStrategy.analyze()` orchestration
- Trend analysis, confirmation checks, and entry signal construction
- `TradingSignal` generation with `market_context` and `strategy_data`

Key configurable defaults in mirror:

- `ema_200=200`, `ema_50=50`, `ema_21=21`
- `hull_55=55`, `hull_34=34`
- `rsi_period=14`, `atr_period=14`, `swing_lookback=5`
- `tp1_rr=1.0`, `tp2_rr=2.0`
- `tp1_close_percent=0.50`, `tp2_close_percent=0.30`, `trail_percent=0.20`
- RSI filter bounds and ATR stop bounds

## EA Input Catalog: v2

Source: `src/strategies/MTFTR_EA_v2.mq5`.

### Risk Management

- `InpRiskPercent`
- `InpMaxOpenTrades`
- `InpMaxDailyTrades`
- `InpMaxConsecLosses`
- `InpMaxDailyDD`
- `InpMaxWeeklyDD`
- `InpMaxSpreadATR`

### 4H Trend Settings

- `InpH4_EMA_Period`
- `InpH4_Hull_Period`
- `InpH4_ATR_Period`
- `InpH4_SlopeBars`
- `InpH4_SlopeMin`
- `InpH4_NoTradeATR`

### 1H Confirmation

- `InpH1_EMA_Period`
- `InpH1_Hull_Period`

### 15M Entry Settings

- `InpM15_EMA_Period`
- `InpM15_RSI_Period`
- `InpM15_ATR_Period`
- `InpRSI_Long_Min`
- `InpRSI_Long_Max`
- `InpRSI_Short_Min`
- `InpRSI_Short_Max`
- `InpEMA_Proximity`
- `InpSwingLookback`

### Stop Loss

- `InpSL_ATR_Buffer`
- `InpSL_Max_ATR`
- `InpSL_Min_ATR`

### Take Profit and Exits

- `InpBE_RR`
- `InpTP1_RR`
- `InpTP2_RR`
- `InpTP3_RR`
- `InpTP1_ClosePct`
- `InpTP2_ClosePct`
- `InpMaxTradeHours`

### Session Filter

- `InpUseSessionFilter`
- `InpGMTOffset`
- `InpSessionStart`
- `InpSessionEnd`
- `InpFridayFilter`

### General

- `InpMagicNumber`
- `InpEnableLogging`

### Visual Display

- `InpShowInfoPanel`
- `InpShowHullBands`
- `InpShowEMAs`
- `InpBullColor`
- `InpBearColor`
- `InpNeutralColor`
- `InpEMA200Color`
- `InpEMA50Color`
- `InpEMA21Color`

## EA Input Catalog: v4

Source: `src/strategies/MTFTR_EA_v4.0.mq5`.

### Risk Management

- `InpRiskPercent`
- `InpMaxOpenTrades`
- `InpMaxDailyTrades`
- `InpMaxConsecLosses`
- `InpMaxDailyDD`
- `InpMaxWeeklyDD`
- `InpMaxSpreadATR`

### Monthly Withdrawal

- `InpEnableWithdrawal`
- `InpWithdrawPct`
- `InpLogWithdrawals`

### Daily Master Bias

- `InpD1_EMA_Period`
- `InpD1_Hull_Period`

### 4H Trend Settings

- `InpH4_EMA_Period`
- `InpH4_Hull_Period`
- `InpH4_ATR_Period`
- `InpH4_SlopeBars`
- `InpH4_SlopeMin`
- `InpH4_NoTradeATR`
- `InpH4_HullMinBars`

### 1H Confirmation

- `InpH1_EMA_Period`
- `InpH1_Hull_Period`

### ADX Trend Strength Filter

- `InpADXEnabled`
- `InpADX_Period`
- `InpADX_Min`
- `InpADX_TF`

### 15M Entry Settings

- `InpM15_EMA_Period`
- `InpM15_RSI_Period`
- `InpM15_ATR_Period`
- `InpRSI_Long_Min`
- `InpRSI_Long_Max`
- `InpRSI_Short_Min`
- `InpRSI_Short_Max`
- `InpSwingLookback`

### Entry Precision

- `InpProximityATR`
- `InpSwingLookbackSL`

### Fibonacci Entry

- `InpFibEnabled`
- `InpFibLevel1`
- `InpFibLevel2`
- `InpFibTolerance`
- `InpFibSwingBarsH1`
- `InpFibRSI_Long_Min`
- `InpFibRSI_Long_Max`
- `InpFibRSI_Short_Min`
- `InpFibRSI_Short_Max`

### Stop Loss

- `InpSL_ATR_Buffer`
- `InpSL_Max_ATR`
- `InpSL_Min_ATR`

### 3-Tier Take Profit

- `InpTP1_RR`
- `InpTP2_RR`
- `InpTP3_RR`
- `InpTP1_ClosePct`
- `InpTP2_ClosePct`
- `InpTP1_SL_MoveToRR`
- `InpTP2_SL_MoveToRR`
- `InpMaxTradeHours`
- `InpSmartTimeExit`

### Session Filter

- `InpGMTOffset`
- `InpSessionStart`
- `InpSessionEnd`
- `InpFridayFilter`

### General

- `InpMagicNumber`

### Logging

- `InpEnableCSV`
- `InpCSVFileName`
- `InpEnableFastAPI`
- `InpFastAPIURL`
- `InpFastAPITimeout`

### Visual

- `InpShowInfoPanel`
- `InpBullColor`
- `InpBearColor`
- `InpNeutralColor`

## MQL5 to Python Parameter Mapping

| Concept | EA Inputs | Python Config |
|---|---|---|
| Trend EMA | `InpH4_EMA_Period` | `ema_200` |
| Confirmation EMA | `InpH1_EMA_Period` | `ema_50` |
| Entry EMA | `InpM15_EMA_Period` | `ema_21` |
| Hull trend | `InpH4_Hull_Period` | `hull_55` |
| Hull confirm | `InpH1_Hull_Period` | `hull_34` |
| RSI period | `InpM15_RSI_Period` | `rsi_period` |
| ATR period | `InpM15_ATR_Period` or `InpH4_ATR_Period` | `atr_period` |
| Swing lookback | `InpSwingLookback` | `swing_lookback` |
| TP1 ratio | `InpTP1_RR` | `tp1_rr` |
| TP2 ratio | `InpTP2_RR` | `tp2_rr` |
| TP1 close percent | `InpTP1_ClosePct` | `tp1_close_percent` |
| TP2 close percent | `InpTP2_ClosePct` | `tp2_close_percent` |
| SL ATR bounds | `InpSL_Min_ATR`, `InpSL_Max_ATR` | `min_sl_atr`, `max_sl_atr` |

## Signal Confidence Model

Current implementation stores confidence as a single scalar in generated signals. Future roadmap can move to explicit factorized scoring, for example:

- trend alignment
- pattern quality
- RSI regime
- ATR regime
- session quality
- hull agreement
- proximity quality

## Source Citations

- [src/strategies/mtftr.py](../src/strategies/mtftr.py#L51)
- [src/strategies/mtftr.py](../src/strategies/mtftr.py#L97)
- [src/strategies/MTFTR_EA_v2.mq5](../src/strategies/MTFTR_EA_v2.mq5#L55)
- [src/strategies/MTFTR_EA_v4.0.mq5](../src/strategies/MTFTR_EA_v4.0.mq5#L54)
- [src/strategies/MTFTR_EA_v4.0.mq5](../src/strategies/MTFTR_EA_v4.0.mq5#L121)

## Related Materials

- `docs/FX_Algo_Trading_Prompt_Bank.docx`
- `docs/Sniper_Plan_5R_v2.docx`
- `src/strategies/MTFTR_EA_v2.mq5`
- `src/strategies/MTFTR_EA_v4.0.mq5`
- `src/strategies/mtftr.py`
- `docs/RISK_MANAGEMENT.md`
