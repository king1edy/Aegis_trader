//+------------------------------------------------------------------+
//| MTFTR_EA.mq5                                                      |
//| Multi-Timeframe Trend Rider Expert Advisor                        |
//| Strategy: Daily Bias -> 4H Trend -> 1H Confirmation -> 15M Entry |
//| Instrument: XAUUSD (works on any symbol)                          |
//| Version: 2.2  (Full Dual Logging: CSV + FastAPI)                 |
//+------------------------------------------------------------------+
//| LOGIC FLOW v2.0+:                                                 |
//|  0. DAILY: EMA50 + Hull MA(21) = Master Trend Bias               |
//|  1. 4H: EMA200 + Hull MA(55) = Trend Bias (must align with Daily)|
//|     -> Hull must show minimum consecutive bars (strength gate)    |
//|  2. 1H: EMA50 + Hull MA(34) must align = Confirmation            |
//|  3. 15M: Method A - EMA21 bounce + reversal candle               |
//|          Method B - Structure break                               |
//|          Method C - Fibonacci 50%/61.8% pullback                 |
//|  4. Enter with 1% risk, SL at swing + 4H ATR buffer              |
//|  5. Scale out: 50% at 1:1, 30% at 2:1, trail 20% on 1H EMA50    |
//+------------------------------------------------------------------+
//| LOGGING ARCHITECTURE v2.2:                                        |
//|  DUAL LAYER — CSV is always written first (primary, no deps).    |
//|  FastAPI is attempted after if enabled (silently skipped when     |
//|  the server is offline — 2s timeout, never blocks trade exec).   |
//|                                                                   |
//|  Events logged: OPEN, TP1_HIT, TP2_HIT, SL_HIT, TIME_EXIT,      |
//|                 TRAIL_EXIT (Hull flip or EMA50 breach)            |
//|                                                                   |
//|  Each event carries: ticket, event, timestamp, direction, method, |
//|  entry, sl, tp1, tp2, lots, risk_pct, sl_dist, exit_price, pnl,  |
//|  outcome, rr_achieved, session, d1_bias, h4_bias, balance, eq    |
//|                                                                   |
//|  CSV: MQL5/Files/Common/MTFTR_TradeLog_XAUUSD.csv (same path     |
//|       as before, now with enriched columns + close events)        |
//|  FastAPI: POST http://localhost:8000/trade (JSON body)            |
//|           GET  http://localhost:8000/health  (connection test)    |
//|  Companion server: mtftr_server.py (run alongside MT5)           |
//+------------------------------------------------------------------+
//| FIXES HISTORY:                                                    |
//|  v2.1 FIX 1: SL uses 4H ATR (was M15 ATR -> immediate SL hits)  |
//|  v2.1 FIX 2: Netting mode protection (blocks merged positions)   |
//|  v2.2 FIX 3: Full event logging (was only OPEN, now all events)  |
//|  v2.2 FIX 4: TrackedPosition stores method/session/bias for use  |
//|              in close events (was lost after entry)               |
//+------------------------------------------------------------------+
#property copyright "MTFTR Strategy v2.2"
#property version   "2.20"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//| Enumerations                                                      |
//+------------------------------------------------------------------+
enum ENUM_TREND_BIAS
{
   BIAS_LONG    =  1,
   BIAS_SHORT   = -1,
   BIAS_NEUTRAL =  0
};

enum ENUM_ENTRY_METHOD
{
   ENTRY_NONE        = 0,
   ENTRY_EMA_BOUNCE  = 1,   // Method A: EMA 21 bounce with reversal candle
   ENTRY_STRUCTURE   = 2,   // Method B: Break of structure
   ENTRY_FIBONACCI   = 3    // Method C: Fibonacci 50%/61.8% pullback [NEW v2.0]
};

enum ENUM_POS_STATE
{
   STATE_INITIAL  = 0,     // Waiting for TP1
   STATE_TP1_HIT  = 1,     // 50% closed, SL at BE, waiting for TP2
   STATE_TP2_HIT  = 2      // 30% closed, trailing final 20%
};

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "====== Risk Management ======"
input double   InpRiskPercent      = 1.0;       // Risk per trade (%)
input int      InpMaxOpenTrades    = 2;         // Max simultaneous trades
input int      InpMaxDailyTrades   = 3;         // Max trades per day
input int      InpMaxConsecLosses  = 2;         // Consecutive losses -> pause
input double   InpMaxDailyDD       = 3.0;       // Max daily drawdown (%)
input double   InpMaxWeeklyDD      = 6.0;       // Max weekly drawdown (%)

input group "====== [NEW v2.0] Daily Master Bias ======"
input int      InpD1_EMA_Period    = 50;        // Daily EMA period
input int      InpD1_Hull_Period   = 21;        // Daily Hull MA period
input bool     InpD1_StrictAlign   = false;     // Strict: 4H MUST match Daily (false = Daily just cannot oppose)

input group "====== 4H Trend Settings ======"
input int      InpH4_EMA_Period    = 200;       // EMA period
input int      InpH4_Hull_Period   = 55;        // Hull MA period
input int      InpH4_ATR_Period    = 14;        // ATR period
input int      InpH4_SlopeBars     = 5;         // EMA slope lookback (bars)
input double   InpH4_SlopeMin      = 0.5;       // Min EMA slope (points)
input double   InpH4_NoTradeATR    = 1.5;       // No-trade zone (ATR multiplier)
input int      InpH4_HullMinBars   = 2;         // [NEW v2.0] Min consecutive Hull bars in trend direction

input group "====== 1H Confirmation ======"
input int      InpH1_EMA_Period    = 50;        // EMA period
input int      InpH1_Hull_Period   = 34;        // Hull MA period

input group "====== 15M Entry Settings ======"
input int      InpM15_EMA_Period   = 21;        // EMA period
input int      InpM15_RSI_Period   = 14;        // RSI period
input int      InpM15_ATR_Period   = 14;        // ATR period
input double   InpRSI_Long_Min     = 40.0;      // RSI min for long entry
input double   InpRSI_Long_Max     = 55.0;      // RSI max for long entry
input double   InpRSI_Short_Min    = 45.0;      // RSI min for short entry
input double   InpRSI_Short_Max    = 60.0;      // RSI max for short entry
input int      InpSwingLookback    = 3;         // Swing detection lookback

input group "====== [NEW v2.0] Fibonacci Entry (Method C) ======"
input bool     InpFibEnabled       = true;      // Enable Fibonacci entry method
input double   InpFibLevel1        = 0.500;     // First Fib level (50%)
input double   InpFibLevel2        = 0.618;     // Second Fib level (61.8%)
input double   InpFibTolerance     = 0.005;     // Fib zone tolerance (0.5% of swing range)
input int      InpFibSwingBarsH1   = 50;        // Lookback bars on 1H for Fib swing

input group "====== Stop Loss (uses 4H ATR for meaningful distance) ======"
input double   InpSL_ATR_Buffer    = 0.5;       // 4H ATR buffer below/above swing (0.5 ~ $5-8 on XAUUSD)
input double   InpSL_Max_ATR       = 4.0;       // Max SL width in 4H ATR multiples
input double   InpSL_Min_ATR       = 1.5;       // Min SL width in 4H ATR multiples (1.5 ~ $12-20 on XAUUSD)

input group "====== Take Profit & Exits ======"
input double   InpTP1_RR           = 1.0;       // TP1 Risk:Reward
input double   InpTP2_RR           = 2.0;       // TP2 Risk:Reward
input double   InpTP1_ClosePct     = 50.0;      // % to close at TP1
input double   InpTP2_ClosePct     = 60.0;      // % of remainder at TP2
input int      InpMaxTradeHours    = 8;         // Time exit if TP1 not hit (hours)

input group "====== Session Filter (Server Time) ======"
input int      InpGMTOffset        = 0;         // Broker server GMT offset
input int      InpLondonStart      = 7;         // London open (GMT)
input int      InpLondonEnd        = 12;        // London close (GMT)
input int      InpNYStart          = 13;        // NY overlap start (GMT)
input int      InpNYEnd            = 16;        // NY overlap end (GMT)
input bool     InpFridayFilter     = true;      // Stop trading Friday 15:00 GMT

input group "====== General ======"
input int      InpMagicNumber      = 20260202;  // Magic number

input group "====== Logging — CSV ======"
input bool     InpEnableCSV        = true;          // Write CSV log (always recommended)
input string   InpCSVFileName      = "MTFTR_TradeLog_XAUUSD.csv"; // CSV filename (in Common/Files)

input group "====== Logging — FastAPI (optional) ======"
input bool     InpEnableFastAPI    = false;         // POST events to FastAPI server
input string   InpFastAPIURL       = "http://127.0.0.1:8000/trade"; // FastAPI endpoint URL
input int      InpFastAPITimeout   = 2000;          // Request timeout ms (keep low — never blocks trades)

input group "====== Visual Display ======"
input bool     InpShowInfoPanel    = true;      // Show info panel on chart
input bool     InpShowHullBands    = true;      // Show Hull MA bands
input bool     InpShowEMAs         = true;      // Show EMA lines
input color    InpBullColor        = clrLime;   // Bullish color
input color    InpBearColor        = clrRed;    // Bearish color
input color    InpNeutralColor     = clrGray;   // Neutral color
input color    InpEMA200Color      = clrWhite;  // 4H EMA 200 color
input color    InpEMA50Color       = clrYellow; // 1H EMA 50 color
input color    InpEMA21Color       = clrAqua;   // 15M EMA 21 color

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  posInfo;

// --- Indicator handles: DAILY [NEW v2.0]
int d1_ema50_handle;
int d1_wma_half_handle;
int d1_wma_full_handle;

// --- Indicator handles: 4H
int h4_ema200_handle;
int h4_wma_half_handle;
int h4_wma_full_handle;
int h4_atr_handle;

// --- Indicator handles: 1H
int h1_ema50_handle;
int h1_wma_half_handle;
int h1_wma_full_handle;

// --- Indicator handles: 15M
int m15_ema21_handle;
int m15_rsi_handle;
int m15_atr_handle;

// --- Hull MA derived periods
int d1_hull_sqrt;
int h4_hull_sqrt;
int h1_hull_sqrt;

// --- New bar tracking
datetime lastBar_M15 = 0;
datetime lastBar_H1  = 0;
datetime lastBar_H4  = 0;

// --- Daily/weekly tracking
int    dailyTradeCount    = 0;
int    consecLosses       = 0;
double dayStartBalance    = 0;
double weekStartBalance   = 0;
int    lastTradeDay        = 0;
int    lastTradeWeekDay    = 0;
bool   dailyPauseActive   = false;

// --- Position tracking
struct TrackedPosition
{
   ulong    ticket;
   double   entryPrice;
   double   originalSL;
   double   tp1Price;
   double   tp2Price;
   double   originalVolume;
   int      state;
   int      direction;
   datetime entryTime;
   bool     active;
   // --- Extended fields for enriched close-event logging (v2.2)
   string   entryMethod;    // "EMA Bounce" / "Structure Break" / "Fibonacci 50/61.8%"
   string   session;        // "London" / "NY Overlap" / "Lunch" / "Off-Hours"
   string   d1Bias;         // "LONG" / "SHORT" / "NEUTRAL" at time of entry
   string   h4Bias;         // "LONG" / "SHORT" / "NEUTRAL" at time of entry
   double   entryBalance;   // Account balance at time of entry
};

TrackedPosition g_positions[];
int             g_posCount = 0;

// --- Visual display objects
string          g_panelName = "MTFTR_Panel";
string          g_hullPrefix = "MTFTR_Hull_";
string          g_emaPrefix = "MTFTR_EMA_";
datetime        g_lastVisualUpdate = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   d1_hull_sqrt = (int)MathFloor(MathSqrt(InpD1_Hull_Period));
   h4_hull_sqrt = (int)MathFloor(MathSqrt(InpH4_Hull_Period));
   h1_hull_sqrt = (int)MathFloor(MathSqrt(InpH1_Hull_Period));
   
   int d1_hull_half = (int)MathFloor(InpD1_Hull_Period / 2.0);
   int h4_hull_half = (int)MathFloor(InpH4_Hull_Period / 2.0);
   int h1_hull_half = (int)MathFloor(InpH1_Hull_Period / 2.0);
   
   // Daily handles
   d1_ema50_handle    = iMA(_Symbol, PERIOD_D1, InpD1_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   d1_wma_half_handle = iMA(_Symbol, PERIOD_D1, d1_hull_half, 0, MODE_LWMA, PRICE_CLOSE);
   d1_wma_full_handle = iMA(_Symbol, PERIOD_D1, InpD1_Hull_Period, 0, MODE_LWMA, PRICE_CLOSE);
   
   // 4H handles
   h4_ema200_handle   = iMA(_Symbol, PERIOD_H4, InpH4_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   h4_wma_half_handle = iMA(_Symbol, PERIOD_H4, h4_hull_half, 0, MODE_LWMA, PRICE_CLOSE);
   h4_wma_full_handle = iMA(_Symbol, PERIOD_H4, InpH4_Hull_Period, 0, MODE_LWMA, PRICE_CLOSE);
   h4_atr_handle      = iATR(_Symbol, PERIOD_H4, InpH4_ATR_Period);
   
   // 1H handles
   h1_ema50_handle    = iMA(_Symbol, PERIOD_H1, InpH1_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   h1_wma_half_handle = iMA(_Symbol, PERIOD_H1, h1_hull_half, 0, MODE_LWMA, PRICE_CLOSE);
   h1_wma_full_handle = iMA(_Symbol, PERIOD_H1, InpH1_Hull_Period, 0, MODE_LWMA, PRICE_CLOSE);
   
   // 15M handles
   m15_ema21_handle   = iMA(_Symbol, PERIOD_M15, InpM15_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   m15_rsi_handle     = iRSI(_Symbol, PERIOD_M15, InpM15_RSI_Period, PRICE_CLOSE);
   m15_atr_handle     = iATR(_Symbol, PERIOD_M15, InpM15_ATR_Period);
   
   if(d1_ema50_handle == INVALID_HANDLE || d1_wma_half_handle == INVALID_HANDLE ||
      d1_wma_full_handle == INVALID_HANDLE ||
      h4_ema200_handle == INVALID_HANDLE || h4_wma_half_handle == INVALID_HANDLE ||
      h4_wma_full_handle == INVALID_HANDLE || h4_atr_handle == INVALID_HANDLE ||
      h1_ema50_handle == INVALID_HANDLE || h1_wma_half_handle == INVALID_HANDLE ||
      h1_wma_full_handle == INVALID_HANDLE || m15_ema21_handle == INVALID_HANDLE ||
      m15_rsi_handle == INVALID_HANDLE || m15_atr_handle == INVALID_HANDLE)
   {
      Print("MTFTR ERROR: Failed to create indicator handles");
      return(INIT_FAILED);
   }
   
   dayStartBalance  = AccountInfoDouble(ACCOUNT_BALANCE);
   weekStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   LoadExistingPositions();
   ArrayResize(g_positions, 10);
   
   Print("MTFTR EA v2.0 initialized. Magic: ", InpMagicNumber);
   Print("D1 Hull sqrt: ", d1_hull_sqrt, " | 4H Hull sqrt: ", h4_hull_sqrt, " | 1H Hull sqrt: ", h1_hull_sqrt);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(d1_ema50_handle);
   IndicatorRelease(d1_wma_half_handle);
   IndicatorRelease(d1_wma_full_handle);
   IndicatorRelease(h4_ema200_handle);
   IndicatorRelease(h4_wma_half_handle);
   IndicatorRelease(h4_wma_full_handle);
   IndicatorRelease(h4_atr_handle);
   IndicatorRelease(h1_ema50_handle);
   IndicatorRelease(h1_wma_half_handle);
   IndicatorRelease(h1_wma_full_handle);
   IndicatorRelease(m15_ema21_handle);
   IndicatorRelease(m15_rsi_handle);
   IndicatorRelease(m15_atr_handle);
   
   CleanupVisualObjects();
   Print("MTFTR EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   UpdateVisualDisplay();
   ManageOpenPositions();
   CleanupClosedPositions();
   
   if(!IsNewBar(PERIOD_M15))
      return;
   
   ResetDailyStats();
   
   if(dailyPauseActive)     return;
   if(!CheckDailyLimits())  return;
   if(!CheckWeeklyLimits()) return;
   if(CountOpenPositions() >= InpMaxOpenTrades) return;
   if(dailyTradeCount >= InpMaxDailyTrades)     return;
   if(!IsWithinSession())   return;
   
   // === [NEW v2.0] STEP 0: Daily Master Bias ===
   ENUM_TREND_BIAS dailyBias = GetDailyTrendBias();
   
   // === STEP 1: 4H Trend Bias ===
   ENUM_TREND_BIAS bias = Get4HTrendBias();
   if(bias == BIAS_NEUTRAL)
      return;
   
   // === [NEW v2.0] Daily alignment check ===
   // Block if Daily is actively opposing the 4H direction
   // e.g., 4H says LONG but Daily says SHORT = avoid fighting the master trend
   if(dailyBias != BIAS_NEUTRAL && dailyBias != bias)
      return;
   
   // === STEP 2: 1H Confirmation ===
   if(!Check1HConfirmation(bias))
      return;
   
   // === STEP 3: 15M Entry Signal ===
   ENUM_ENTRY_METHOD entryMethod = Check15MEntry(bias);
   if(entryMethod == ENTRY_NONE)
      return;
   
   // === STEP 4: Execute Trade ===
   ExecuteTrade(bias, entryMethod);
}

//+------------------------------------------------------------------+
//| [NEW v2.0] STEP 0: Get Daily Master Trend Bias                   |
//| Daily EMA50 + Daily Hull MA(21)                                   |
//| The highest-timeframe filter — master trend direction             |
//+------------------------------------------------------------------+
ENUM_TREND_BIAS GetDailyTrendBias()
{
   double ema50d[];
   ArraySetAsSeries(ema50d, true);
   if(CopyBuffer(d1_ema50_handle, 0, 0, 1, ema50d) < 1)
      return BIAS_NEUTRAL;
   
   // Use bar[1] = last confirmed closed Daily bar
   double closeD1[];
   ArraySetAsSeries(closeD1, true);
   if(CopyClose(_Symbol, PERIOD_D1, 0, 2, closeD1) < 2)
      return BIAS_NEUTRAL;
   double price = closeD1[1];
   
   bool hullBullish = IsHullMABullish(d1_wma_half_handle, d1_wma_full_handle, d1_hull_sqrt);
   bool hullBearish = IsHullMABearish(d1_wma_half_handle, d1_wma_full_handle, d1_hull_sqrt);
   
   if(price > ema50d[0] && hullBullish)  return BIAS_LONG;
   if(price < ema50d[0] && hullBearish)  return BIAS_SHORT;
   
   return BIAS_NEUTRAL;
}

//+------------------------------------------------------------------+
//| STEP 1: Determine 4H Trend Bias                                  |
//| v2.0: Added Hull MA consecutive bars strength check               |
//+------------------------------------------------------------------+
ENUM_TREND_BIAS Get4HTrendBias()
{
   double ema200[];
   ArraySetAsSeries(ema200, true);
   if(CopyBuffer(h4_ema200_handle, 0, 0, InpH4_SlopeBars + 1, ema200) < InpH4_SlopeBars + 1)
      return BIAS_NEUTRAL;
   
   double ema200_current = ema200[0];
   double ema200_past    = ema200[InpH4_SlopeBars];
   double ema200_slope   = ema200_current - ema200_past;
   
   double close4h[];
   ArraySetAsSeries(close4h, true);
   if(CopyClose(_Symbol, PERIOD_H4, 0, 1, close4h) < 1)
      return BIAS_NEUTRAL;
   double price = close4h[0];
   
   double atr4h[];
   ArraySetAsSeries(atr4h, true);
   if(CopyBuffer(h4_atr_handle, 0, 0, 1, atr4h) < 1)
      return BIAS_NEUTRAL;
   double atr = atr4h[0];
   
   if(MathAbs(price - ema200_current) < InpH4_NoTradeATR * atr)
      return BIAS_NEUTRAL;
   
   if(MathAbs(ema200_slope) < InpH4_SlopeMin)
      return BIAS_NEUTRAL;
   
   bool hullBullish = IsHullMABullish(h4_wma_half_handle, h4_wma_full_handle, h4_hull_sqrt);
   bool hullBearish = IsHullMABearish(h4_wma_half_handle, h4_wma_full_handle, h4_hull_sqrt);
   
   // === [NEW v2.0] Hull Strength Gate ===
   // Ensure the trend is established, not just a single bar flip
   if(hullBullish)
   {
      int consecBars = GetHullConsecutiveBars(h4_wma_half_handle, h4_wma_full_handle,
                                              h4_hull_sqrt, true, InpH4_HullMinBars + 3);
      if(consecBars < InpH4_HullMinBars)
         return BIAS_NEUTRAL;
   }
   if(hullBearish)
   {
      int consecBars = GetHullConsecutiveBars(h4_wma_half_handle, h4_wma_full_handle,
                                              h4_hull_sqrt, false, InpH4_HullMinBars + 3);
      if(consecBars < InpH4_HullMinBars)
         return BIAS_NEUTRAL;
   }
   
   if(price > ema200_current && ema200_slope > InpH4_SlopeMin && hullBullish)
      return BIAS_LONG;
   
   if(price < ema200_current && ema200_slope < -InpH4_SlopeMin && hullBearish)
      return BIAS_SHORT;
   
   return BIAS_NEUTRAL;
}

//+------------------------------------------------------------------+
//| STEP 2: Check 1H Trend Confirmation                              |
//+------------------------------------------------------------------+
bool Check1HConfirmation(ENUM_TREND_BIAS bias)
{
   double ema50[];
   ArraySetAsSeries(ema50, true);
   if(CopyBuffer(h1_ema50_handle, 0, 0, 1, ema50) < 1)
      return false;
   
   double close1h[];
   ArraySetAsSeries(close1h, true);
   if(CopyClose(_Symbol, PERIOD_H1, 0, 1, close1h) < 1)
      return false;
   
   double price = close1h[0];
   double ema   = ema50[0];
   
   bool hullBullish = IsHullMABullish(h1_wma_half_handle, h1_wma_full_handle, h1_hull_sqrt);
   bool hullBearish = IsHullMABearish(h1_wma_half_handle, h1_wma_full_handle, h1_hull_sqrt);
   
   if(bias == BIAS_LONG)  return (price > ema && hullBullish);
   if(bias == BIAS_SHORT) return (price < ema && hullBearish);
   
   return false;
}

//+------------------------------------------------------------------+
//| STEP 3: Check 15M Entry Signal                                   |
//| v2.0: Added Method C — Fibonacci retracement entry               |
//+------------------------------------------------------------------+
ENUM_ENTRY_METHOD Check15MEntry(ENUM_TREND_BIAS bias)
{
   double open15[], high15[], low15[], close15[];
   ArraySetAsSeries(open15, true);
   ArraySetAsSeries(high15, true);
   ArraySetAsSeries(low15, true);
   ArraySetAsSeries(close15, true);
   
   if(CopyOpen(_Symbol, PERIOD_M15, 0, 30, open15) < 30)   return ENTRY_NONE;
   if(CopyHigh(_Symbol, PERIOD_M15, 0, 30, high15) < 30)   return ENTRY_NONE;
   if(CopyLow(_Symbol, PERIOD_M15, 0, 30, low15) < 30)     return ENTRY_NONE;
   if(CopyClose(_Symbol, PERIOD_M15, 0, 30, close15) < 30) return ENTRY_NONE;
   
   double ema21[];
   ArraySetAsSeries(ema21, true);
   if(CopyBuffer(m15_ema21_handle, 0, 0, 5, ema21) < 5)
      return ENTRY_NONE;
   
   double rsi[];
   ArraySetAsSeries(rsi, true);
   if(CopyBuffer(m15_rsi_handle, 0, 0, 3, rsi) < 3)
      return ENTRY_NONE;
   
   double rsiVal = rsi[1];
   bool rsiOK = false;
   
   if(bias == BIAS_LONG)
      rsiOK = (rsiVal >= InpRSI_Long_Min && rsiVal <= InpRSI_Long_Max);
   else if(bias == BIAS_SHORT)
      rsiOK = (rsiVal >= InpRSI_Short_Min && rsiVal <= InpRSI_Short_Max);
   
   if(!rsiOK) return ENTRY_NONE;
   
   // ===== METHOD A: EMA 21 Bounce =====
   bool pullbackToEMA = false;
   
   if(bias == BIAS_LONG)
   {
      pullbackToEMA = (low15[1] <= ema21[1] || low15[2] <= ema21[2]);
      if(pullbackToEMA)
      {
         if(IsBullishEngulfing(open15, high15, low15, close15, 1) ||
            IsBullishPinBar(open15, high15, low15, close15, 1))
            return ENTRY_EMA_BOUNCE;
      }
   }
   else if(bias == BIAS_SHORT)
   {
      pullbackToEMA = (high15[1] >= ema21[1] || high15[2] >= ema21[2]);
      if(pullbackToEMA)
      {
         if(IsBearishEngulfing(open15, high15, low15, close15, 1) ||
            IsBearishPinBar(open15, high15, low15, close15, 1))
            return ENTRY_EMA_BOUNCE;
      }
   }
   
   // ===== METHOD B: Structure Break =====
   if(bias == BIAS_LONG)
   {
      double swingLowPrice = 0, swingHighPrice = 0;
      int swingLowBar = 0, swingHighBar = 0;
      
      if(FindRecentSwingLow(low15, InpSwingLookback, swingLowPrice, swingLowBar))
         if(FindSwingHighBefore(high15, InpSwingLookback, swingLowBar, swingHighPrice, swingHighBar))
            if(close15[1] > swingHighPrice)
               return ENTRY_STRUCTURE;
   }
   else if(bias == BIAS_SHORT)
   {
      double swingHighPrice = 0, swingLowPrice = 0;
      int swingHighBar = 0, swingLowBar = 0;
      
      if(FindRecentSwingHigh(high15, InpSwingLookback, swingHighPrice, swingHighBar))
         if(FindSwingLowBefore(low15, InpSwingLookback, swingHighBar, swingLowPrice, swingLowBar))
            if(close15[1] < swingLowPrice)
               return ENTRY_STRUCTURE;
   }
   
   // ===== [NEW v2.0] METHOD C: Fibonacci 50%/61.8% Retracement =====
   if(InpFibEnabled)
   {
      if(CheckFibonacciEntry(bias, open15, high15, low15, close15))
         return ENTRY_FIBONACCI;
   }
   
   return ENTRY_NONE;
}

//+------------------------------------------------------------------+
//| [NEW v2.0] Method C: Fibonacci Retracement Entry                 |
//| Identifies the most recent significant swing on 1H               |
//| and checks if 15M price is at the 50% or 61.8% retracement      |
//| with a reversal candle confirming the level                       |
//|                                                                   |
//| Why 1H swings: More significant than 15M, gives better reference  |
//| Why 50/61.8%: The "golden zone" — highest probability of reversal |
//+------------------------------------------------------------------+
bool CheckFibonacciEntry(ENUM_TREND_BIAS bias,
                          const double &open15[], const double &high15[],
                          const double &low15[],  const double &close15[])
{
   double high1h[], low1h[];
   ArraySetAsSeries(high1h, true);
   ArraySetAsSeries(low1h, true);
   
   if(CopyHigh(_Symbol, PERIOD_H1, 0, InpFibSwingBarsH1, high1h) < InpFibSwingBarsH1) return false;
   if(CopyLow(_Symbol, PERIOD_H1, 0, InpFibSwingBarsH1, low1h) < InpFibSwingBarsH1)   return false;
   
   double fibSwingHigh = 0, fibSwingLow = 0;
   int swingHighBar = 0, swingLowBar = 0;
   
   if(bias == BIAS_LONG)
   {
      // Retracing a bullish impulse: swing low -> swing high -> pullback to Fib
      if(!FindRecentSwingHigh(high1h, 3, fibSwingHigh, swingHighBar)) return false;
      if(!FindSwingLowBefore(low1h, 3, swingHighBar, fibSwingLow, swingLowBar)) return false;
      
      double swingRange = fibSwingHigh - fibSwingLow;
      if(swingRange <= 0) return false;
      
      // Fib levels measured from the high downward
      double fib50  = fibSwingHigh - swingRange * InpFibLevel1;
      double fib618 = fibSwingHigh - swingRange * InpFibLevel2;
      double tolerance = swingRange * InpFibTolerance;
      
      double currentLow   = low15[1];
      double currentClose = close15[1];
      
      // Price must have touched the zone (50-61.8 golden zone)
      bool inFibZone = (currentLow >= fib618 - tolerance && currentLow <= fib50 + tolerance);
      if(!inFibZone) return false;
      
      // Must close above the deeper Fib level (not rejected through it)
      if(currentClose < fib618 - tolerance) return false;
      
      // Must have a bullish reversal candle at the zone
      bool hasReversal = IsBullishEngulfing(open15, high15, low15, close15, 1) ||
                         IsBullishPinBar(open15, high15, low15, close15, 1);
      if(!hasReversal) return false;
      
      Print("MTFTR FIB LONG: Range=", DoubleToString(swingRange, 2),
            " Fib50=", DoubleToString(fib50, _Digits),
            " Fib618=", DoubleToString(fib618, _Digits),
            " Low=", DoubleToString(currentLow, _Digits));
      return true;
   }
   else if(bias == BIAS_SHORT)
   {
      // Retracing a bearish impulse: swing high -> swing low -> rally to Fib
      if(!FindRecentSwingLow(low1h, 3, fibSwingLow, swingLowBar)) return false;
      if(!FindSwingHighBefore(high1h, 3, swingLowBar, fibSwingHigh, swingHighBar)) return false;
      
      double swingRange = fibSwingHigh - fibSwingLow;
      if(swingRange <= 0) return false;
      
      // Fib levels measured from the low upward
      double fib50  = fibSwingLow + swingRange * InpFibLevel1;
      double fib618 = fibSwingLow + swingRange * InpFibLevel2;
      double tolerance = swingRange * InpFibTolerance;
      
      double currentHigh  = high15[1];
      double currentClose = close15[1];
      
      bool inFibZone = (currentHigh >= fib50 - tolerance && currentHigh <= fib618 + tolerance);
      if(!inFibZone) return false;
      
      if(currentClose > fib618 + tolerance) return false;
      
      bool hasReversal = IsBearishEngulfing(open15, high15, low15, close15, 1) ||
                         IsBearishPinBar(open15, high15, low15, close15, 1);
      if(!hasReversal) return false;
      
      Print("MTFTR FIB SHORT: Range=", DoubleToString(swingRange, 2),
            " Fib50=", DoubleToString(fib50, _Digits),
            " Fib618=", DoubleToString(fib618, _Digits),
            " High=", DoubleToString(currentHigh, _Digits));
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Execute Trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_TREND_BIAS bias, ENUM_ENTRY_METHOD method)
{
   // === NETTING ACCOUNT PROTECTION ===
   // MetaQuotes Demo and many brokers use netting mode. In netting, opening
   // a second position on the same symbol just adds lots to the existing one,
   // bypassing MaxOpenTrades logic and creating oversized exposure.
   // Solution: if netting account, block entry when any position already open.
   long marginMode = (long)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   bool isNetting  = (marginMode == 0); // 0 = ACCOUNT_MARGIN_MODE_RETAIL_NETTING
   if(isNetting && CountOpenPositions() > 0)
   {
      // Don't log this — it fires every bar when a trade is open
      return;
   }
   
   // === USE 4H ATR FOR SL CALCULATION ===
   // Critical fix: M15 ATR on XAUUSD is often $1-2 in calm conditions,
   // creating SLs that are stopped out by normal tick noise within seconds.
   // 4H ATR gives $8-20 range which represents meaningful market structure.
   double atr4h[];
   ArraySetAsSeries(atr4h, true);
   if(CopyBuffer(h4_atr_handle, 0, 0, 1, atr4h) < 1) return;
   double atr = atr4h[0];  // 4H ATR used for ALL SL distance calculations
   
   double low15[], high15[];
   ArraySetAsSeries(low15, true);
   ArraySetAsSeries(high15, true);
   if(CopyLow(_Symbol, PERIOD_M15, 0, 20, low15) < 20)   return;
   if(CopyHigh(_Symbol, PERIOD_M15, 0, 20, high15) < 20) return;
   
   double entryPrice = 0, slPrice = 0, slDistance = 0;
   
   if(bias == BIAS_LONG)
   {
      entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double swingLow = 0; int swingBar = 0;
      if(FindRecentSwingLow(low15, InpSwingLookback, swingLow, swingBar))
         slPrice = swingLow - InpSL_ATR_Buffer * atr;
      else
         slPrice = entryPrice - InpSL_Min_ATR * atr;
      slDistance = entryPrice - slPrice;
   }
   else
   {
      entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double swingHigh = 0; int swingBar = 0;
      if(FindRecentSwingHigh(high15, InpSwingLookback, swingHigh, swingBar))
         slPrice = swingHigh + InpSL_ATR_Buffer * atr;
      else
         slPrice = entryPrice + InpSL_Min_ATR * atr;
      slDistance = slPrice - entryPrice;
   }
   
   if(slDistance <= 0) return;
   
   if(slDistance > InpSL_Max_ATR * atr)
   {
      Print("MTFTR: SL too wide. Trade skipped.");
      return;
   }
   
   if(slDistance < InpSL_Min_ATR * atr)
   {
      slDistance = InpSL_Min_ATR * atr;
      slPrice = (bias == BIAS_LONG) ? entryPrice - slDistance : entryPrice + slDistance;
   }
   
   double lots = CalculateLotSize(slDistance);
   if(lots <= 0) { Print("MTFTR: Invalid lot size. Trade skipped."); return; }
   
   double riskAmt = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   Print("MTFTR LOT CALC: RiskAmt=", DoubleToString(riskAmt, 2),
         " | 4H_ATR=", DoubleToString(atr, 2),
         " | SL_Dist=", DoubleToString(slDistance, _Digits),
         " | Lots=", DoubleToString(lots, 2));
   
   double tp1Price, tp2Price;
   if(bias == BIAS_LONG)
   {
      tp1Price = entryPrice + slDistance * InpTP1_RR;
      tp2Price = entryPrice + slDistance * InpTP2_RR;
   }
   else
   {
      tp1Price = entryPrice - slDistance * InpTP1_RR;
      tp2Price = entryPrice - slDistance * InpTP2_RR;
   }
   
   string methodCode = (method == ENTRY_EMA_BOUNCE) ? "A" : (method == ENTRY_STRUCTURE) ? "B" : "C";
   string comment    = "MTFTR_" + methodCode;
   bool result = false;
   
   if(bias == BIAS_LONG)
      result = trade.Buy(lots, _Symbol, entryPrice, slPrice, 0, comment);
   else
      result = trade.Sell(lots, _Symbol, entryPrice, slPrice, 0, comment);
   
   if(result && trade.ResultRetcode() == TRADE_RETCODE_DONE)
   {
      ulong ticket = trade.ResultOrder();
      
      string biasStr   = (bias == BIAS_LONG) ? "LONG" : "SHORT";
      string methodFull = (method == ENTRY_EMA_BOUNCE) ? "EMA Bounce" :
                          (method == ENTRY_STRUCTURE)   ? "Structure Break" :
                                                          "Fibonacci 50/61.8%";
      
      // Capture session and biases at the moment of entry (stored in struct for close events)
      string entrySess = GetCurrentSession();
      string d1Str     = BiasToString(GetDailyTrendBias());
      string h4Str     = BiasToString(bias);
      
      TrackNewPosition(ticket, entryPrice, slPrice, tp1Price, tp2Price, lots,
                       (bias == BIAS_LONG) ? 1 : -1,
                       methodFull, entrySess, d1Str, h4Str);
      dailyTradeCount++;
      
      Print("MTFTR OPEN: ", biasStr, " | ", methodFull,
            " | Entry: ", DoubleToString(entryPrice, _Digits),
            " | SL: ", DoubleToString(slPrice, _Digits),
            " | TP1: ", DoubleToString(tp1Price, _Digits),
            " | TP2: ", DoubleToString(tp2Price, _Digits),
            " | Lots: ", DoubleToString(lots, 2), " | #", ticket);
      
      // Log the OPEN event — CSV always, FastAPI if enabled
      int posIdx = FindPositionIndex(ticket);
      if(posIdx >= 0)
         LogEvent("OPEN", g_positions[posIdx], 0.0, "");
   }
   else
      Print("MTFTR ERROR: Order failed. Code: ", trade.ResultRetcode(),
            " | Msg: ", trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| Manage Open Positions (runs every tick)                           |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   for(int i = 0; i < g_posCount; i++)
   {
      if(!g_positions[i].active) continue;
      
      ulong ticket = g_positions[i].ticket;
      
      if(!PositionSelectByTicket(ticket))
      {
         g_positions[i].active = false;
         continue;
      }
      
      double posVolume = PositionGetDouble(POSITION_VOLUME);
      int    direction = g_positions[i].direction;
      int    state     = g_positions[i].state;
      
      // Time-based exit
      if(state == STATE_INITIAL &&
         TimeCurrent() - g_positions[i].entryTime > InpMaxTradeHours * 3600)
      {
         double exitPx = (g_positions[i].direction == 1) ?
            SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         Print("MTFTR: Time exit for ticket ", ticket);
         trade.PositionClose(ticket);
         LogEvent("TIME_EXIT", g_positions[i], exitPx, "");
         g_positions[i].active = false;
         continue;
      }
      
      // TP1
      if(state == STATE_INITIAL)
      {
         bool tp1Reached = (direction == 1) ?
            SymbolInfoDouble(_Symbol, SYMBOL_BID) >= g_positions[i].tp1Price :
            SymbolInfoDouble(_Symbol, SYMBOL_ASK) <= g_positions[i].tp1Price;
         
         if(tp1Reached)
         {
            double closeVol = NormalizeLots(g_positions[i].originalVolume * InpTP1_ClosePct / 100.0);
            
            if(closeVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) && closeVol < posVolume)
            {
               if(trade.PositionClosePartial(ticket, closeVol))
               {
                  trade.PositionModify(ticket, g_positions[i].entryPrice, 0);
                  g_positions[i].state = STATE_TP1_HIT;
                  consecLosses = 0;
                  GlobalVariableSet("MTFTR_" + IntegerToString(ticket) + "_STATE", (double)STATE_TP1_HIT);
                  Print("MTFTR TP1: Closed ", closeVol, " lots. BE set. Ticket: ", ticket);
                  LogEvent("TP1_HIT", g_positions[i], g_positions[i].tp1Price, "");
               }
            }
            else
            {
               trade.PositionClose(ticket);
               LogEvent("TP1_HIT", g_positions[i], g_positions[i].tp1Price, "Full close — lot too small to split");
               g_positions[i].active = false;
               consecLosses = 0;
            }
         }
      }
      
      // TP2
      else if(state == STATE_TP1_HIT)
      {
         bool tp2Reached = (direction == 1) ?
            SymbolInfoDouble(_Symbol, SYMBOL_BID) >= g_positions[i].tp2Price :
            SymbolInfoDouble(_Symbol, SYMBOL_ASK) <= g_positions[i].tp2Price;
         
         if(tp2Reached)
         {
            double currentVol = PositionGetDouble(POSITION_VOLUME);
            double trailVol   = NormalizeLots(g_positions[i].originalVolume * 0.20);
            double closeVol   = NormalizeLots(currentVol - trailVol);
            
            if(trailVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) && closeVol > 0)
            {
               if(trade.PositionClosePartial(ticket, closeVol))
               {
                  g_positions[i].state = STATE_TP2_HIT;
                  GlobalVariableSet("MTFTR_" + IntegerToString(ticket) + "_STATE", (double)STATE_TP2_HIT);
                  Print("MTFTR TP2: Closed ", closeVol, " lots. Trailing ", trailVol, ". Ticket: ", ticket);
                  LogEvent("TP2_HIT", g_positions[i], g_positions[i].tp2Price, "");
               }
            }
            else
            {
               trade.PositionClose(ticket);
               LogEvent("TP2_HIT", g_positions[i], g_positions[i].tp2Price, "Full close — lot too small to trail");
               g_positions[i].active = false;
            }
         }
      }
      
      // Trailing final 20% on 1H bar close
      else if(state == STATE_TP2_HIT)
      {
         datetime h1Time = iTime(_Symbol, PERIOD_H1, 0);
         if(h1Time != lastBar_H1)
         {
            lastBar_H1 = h1Time;
            
            bool hullFlipped = (direction == 1 &&
                                IsHullMABearish(h1_wma_half_handle, h1_wma_full_handle, h1_hull_sqrt)) ||
                               (direction == -1 &&
                                IsHullMABullish(h1_wma_half_handle, h1_wma_full_handle, h1_hull_sqrt));
            
            if(hullFlipped)
            {
               double exitPx = (direction == 1) ?
                  SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               trade.PositionClose(ticket);
               LogEvent("TRAIL_EXIT", g_positions[i], exitPx, "Hull MA flipped");
               g_positions[i].active = false;
               Print("MTFTR: Hull flipped. Closed trail. Ticket: ", ticket);
               continue;
            }
            
            double ema50[];
            ArraySetAsSeries(ema50, true);
            if(CopyBuffer(h1_ema50_handle, 0, 0, 1, ema50) >= 1)
            {
               double atr15trail[];
               ArraySetAsSeries(atr15trail, true);
               if(CopyBuffer(m15_atr_handle, 0, 0, 1, atr15trail) >= 1)
               {
                  double currentSL = PositionGetDouble(POSITION_SL);
                  double newSL = 0;
                  
                  if(direction == 1)
                  {
                     newSL = ema50[0] - atr15trail[0] * 0.5;
                     if(newSL > currentSL && newSL < SymbolInfoDouble(_Symbol, SYMBOL_BID))
                        trade.PositionModify(ticket, newSL, 0);
                  }
                  else
                  {
                     newSL = ema50[0] + atr15trail[0] * 0.5;
                     if(newSL < currentSL && newSL > SymbolInfoDouble(_Symbol, SYMBOL_ASK))
                        trade.PositionModify(ticket, newSL, 0);
                  }
               }
               
               double close1h[];
               ArraySetAsSeries(close1h, true);
               if(CopyClose(_Symbol, PERIOD_H1, 0, 2, close1h) >= 2)
               {
                  if(direction == 1 && close1h[1] < ema50[0])
                  {
                     double exitPx = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                     trade.PositionClose(ticket);
                     LogEvent("TRAIL_EXIT", g_positions[i], exitPx, "Price closed below 1H EMA50");
                     g_positions[i].active = false;
                     Print("MTFTR: Close below 1H EMA50. Exiting long trail. Ticket: ", ticket);
                  }
                  else if(direction == -1 && close1h[1] > ema50[0])
                  {
                     double exitPx = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                     trade.PositionClose(ticket);
                     LogEvent("TRAIL_EXIT", g_positions[i], exitPx, "Price closed above 1H EMA50");
                     g_positions[i].active = false;
                     Print("MTFTR: Close above 1H EMA50. Exiting short trail. Ticket: ", ticket);
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Hull MA core calculation                                          |
//+------------------------------------------------------------------+
double GetHullMAValue(int wmaHalfHandle, int wmaFullHandle, int sqrtPeriod, int shift)
{
   double wmaHalf[], wmaFull[];
   ArraySetAsSeries(wmaHalf, true);
   ArraySetAsSeries(wmaFull, true);
   
   if(CopyBuffer(wmaHalfHandle, 0, shift, sqrtPeriod, wmaHalf) < sqrtPeriod) return 0;
   if(CopyBuffer(wmaFullHandle, 0, shift, sqrtPeriod, wmaFull) < sqrtPeriod) return 0;
   
   double weightSum = 0, valueSum = 0;
   for(int i = 0; i < sqrtPeriod; i++)
   {
      double w = (double)(sqrtPeriod - i);
      valueSum  += (2.0 * wmaHalf[i] - wmaFull[i]) * w;
      weightSum += w;
   }
   
   return (weightSum == 0) ? 0 : valueSum / weightSum;
}

bool IsHullMABullish(int wmaHalfHandle, int wmaFullHandle, int sqrtPeriod)
{
   double h0 = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, 0);
   double h1 = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, 1);
   if(h0 == 0 || h1 == 0) return false;
   return (h0 > h1);
}

bool IsHullMABearish(int wmaHalfHandle, int wmaFullHandle, int sqrtPeriod)
{
   double h0 = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, 0);
   double h1 = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, 1);
   if(h0 == 0 || h1 == 0) return false;
   return (h0 < h1);
}

//+------------------------------------------------------------------+
//| [NEW v2.0] Count consecutive Hull MA bars in a direction          |
//| Used to gate the 4H Hull signal — trend must be established       |
//| bullish=true checks for rising bars, false for falling bars       |
//+------------------------------------------------------------------+
int GetHullConsecutiveBars(int wmaHalfHandle, int wmaFullHandle, int sqrtPeriod,
                            bool bullish, int maxLookback)
{
   int count = 0;
   for(int bar = 0; bar < maxLookback; bar++)
   {
      double hC = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, bar);
      double hP = GetHullMAValue(wmaHalfHandle, wmaFullHandle, sqrtPeriod, bar + 1);
      if(hC == 0 || hP == 0) break;
      
      bool rising = (hC > hP);
      if(bar == 0)
      {
         if(rising != bullish) return 0;
         count = 1;
      }
      else
      {
         if(rising == bullish) count++;
         else break;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Candlestick Patterns                                              |
//+------------------------------------------------------------------+
bool IsBullishEngulfing(const double &open[], const double &high[],
                        const double &low[], const double &close[], int idx)
{
   if(close[idx+1] >= open[idx+1]) return false;
   if(close[idx] <= open[idx])     return false;
   return (open[idx] <= close[idx+1] && close[idx] >= open[idx+1]);
}

bool IsBearishEngulfing(const double &open[], const double &high[],
                        const double &low[], const double &close[], int idx)
{
   if(close[idx+1] <= open[idx+1]) return false;
   if(close[idx] >= open[idx])     return false;
   return (open[idx] >= close[idx+1] && close[idx] <= open[idx+1]);
}

bool IsBullishPinBar(const double &open[], const double &high[],
                     const double &low[], const double &close[], int idx)
{
   double body = MathAbs(close[idx] - open[idx]);
   double range = high[idx] - low[idx];
   if(range <= 0) return false;
   double upperWick = high[idx] - MathMax(open[idx], close[idx]);
   double lowerWick = MathMin(open[idx], close[idx]) - low[idx];
   if(lowerWick >= 2.0 * body && upperWick <= body)
      return ((open[idx] + close[idx]) / 2.0 > low[idx] + range * 0.6);
   return false;
}

bool IsBearishPinBar(const double &open[], const double &high[],
                     const double &low[], const double &close[], int idx)
{
   double body = MathAbs(close[idx] - open[idx]);
   double range = high[idx] - low[idx];
   if(range <= 0) return false;
   double upperWick = high[idx] - MathMax(open[idx], close[idx]);
   double lowerWick = MathMin(open[idx], close[idx]) - low[idx];
   if(upperWick >= 2.0 * body && lowerWick <= body)
      return ((open[idx] + close[idx]) / 2.0 < low[idx] + range * 0.4);
   return false;
}

//+------------------------------------------------------------------+
//| Swing Detection                                                   |
//+------------------------------------------------------------------+
bool FindRecentSwingLow(const double &low[], int lookback, double &price, int &bar)
{
   int sz = ArraySize(low);
   if(sz < lookback * 2 + 1) return false;
   int maxBar = MathMin(25, sz - lookback - 1);
   for(int i = lookback; i < maxBar; i++)
   {
      bool ok = true;
      for(int j = 1; j <= lookback; j++)
      {
         if(i-j < 0 || i+j >= sz) { ok = false; break; }
         if(low[i] > low[i-j] || low[i] > low[i+j]) { ok = false; break; }
      }
      if(ok) { price = low[i]; bar = i; return true; }
   }
   return false;
}

bool FindRecentSwingHigh(const double &high[], int lookback, double &price, int &bar)
{
   int sz = ArraySize(high);
   if(sz < lookback * 2 + 1) return false;
   int maxBar = MathMin(25, sz - lookback - 1);
   for(int i = lookback; i < maxBar; i++)
   {
      bool ok = true;
      for(int j = 1; j <= lookback; j++)
      {
         if(i-j < 0 || i+j >= sz) { ok = false; break; }
         if(high[i] < high[i-j] || high[i] < high[i+j]) { ok = false; break; }
      }
      if(ok) { price = high[i]; bar = i; return true; }
   }
   return false;
}

bool FindSwingHighBefore(const double &high[], int lookback, int afterBar,
                         double &price, int &bar)
{
   int sz = ArraySize(high);
   if(sz < lookback * 2 + 1) return false;
   int startBar = MathMax(afterBar + 1, lookback);
   int maxBar   = MathMin(28, sz - lookback - 1);
   for(int i = startBar; i < maxBar; i++)
   {
      bool ok = true;
      for(int j = 1; j <= lookback; j++)
      {
         if(i-j < 0 || i+j >= sz) { ok = false; break; }
         if(high[i] < high[i-j] || high[i] < high[i+j]) { ok = false; break; }
      }
      if(ok) { price = high[i]; bar = i; return true; }
   }
   return false;
}

bool FindSwingLowBefore(const double &low[], int lookback, int afterBar,
                        double &price, int &bar)
{
   int sz = ArraySize(low);
   if(sz < lookback * 2 + 1) return false;
   int startBar = MathMax(afterBar + 1, lookback);
   int maxBar   = MathMin(28, sz - lookback - 1);
   for(int i = startBar; i < maxBar; i++)
   {
      bool ok = true;
      for(int j = 1; j <= lookback; j++)
      {
         if(i-j < 0 || i+j >= sz) { ok = false; break; }
         if(low[i] > low[i-j] || low[i] > low[i+j]) { ok = false; break; }
      }
      if(ok) { price = low[i]; bar = i; return true; }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Lot size calculation                                              |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
   if(slDistance <= 0) return 0;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmt = balance * InpRiskPercent / 100.0;
   
   double testProfit = 0;
   double ep = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, ep, ep - slDistance, testProfit))
   {
      double lossPerLot = MathAbs(testProfit);
      if(lossPerLot > 0) return NormalizeLots(riskAmt / lossPerLot);
   }
   
   double cs = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(cs > 0 && pt > 0)
   {
      double lpl = (slDistance / pt) * (cs * pt);
      if(lpl > 0) return NormalizeLots(riskAmt / lpl);
   }
   
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tv <= 0 || ts <= 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   
   double rpl = (slDistance / ts) * tv;
   if(rpl <= 0) return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   return NormalizeLots(riskAmt / rpl);
}

double NormalizeLots(double lots)
{
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lotStep <= 0) return 0;
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   return NormalizeDouble(lots, (int)MathCeil(-MathLog10(lotStep)));
}

//+------------------------------------------------------------------+
//| Session Filter                                                    |
//+------------------------------------------------------------------+
bool IsWithinSession()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   int gmtHour = dt.hour - InpGMTOffset;
   if(gmtHour < 0)   gmtHour += 24;
   if(gmtHour >= 24) gmtHour -= 24;
   if(InpFridayFilter && dt.day_of_week == 5 && gmtHour >= 15) return false;
   if(dt.day_of_week == 0 || dt.day_of_week == 6) return false;
   return (gmtHour >= InpLondonStart && gmtHour < InpLondonEnd) ||
          (gmtHour >= InpNYStart     && gmtHour < InpNYEnd);
}

bool IsNewBar(ENUM_TIMEFRAMES tf)
{
   datetime t = iTime(_Symbol, tf, 0);
   if(tf == PERIOD_M15) { if(t != lastBar_M15) { lastBar_M15 = t; return true; } }
   else if(tf == PERIOD_H1) { if(t != lastBar_H1) { lastBar_H1 = t; return true; } }
   else if(tf == PERIOD_H4) { if(t != lastBar_H4) { lastBar_H4 = t; return true; } }
   return false;
}

int CountOpenPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(posInfo.SelectByIndex(i))
         if(posInfo.Magic() == InpMagicNumber && posInfo.Symbol() == _Symbol)
            count++;
   return count;
}

//+------------------------------------------------------------------+
//| Position tracking                                                 |
//+------------------------------------------------------------------+
void TrackNewPosition(ulong ticket, double entry, double sl, double tp1,
                      double tp2, double volume, int direction,
                      string method, string session, string d1Bias, string h4Bias)
{
   int idx = -1;
   for(int i = 0; i < g_posCount; i++)
      if(!g_positions[i].active) { idx = i; break; }
   
   if(idx == -1)
   {
      idx = g_posCount++;
      if(g_posCount > ArraySize(g_positions))
         ArrayResize(g_positions, g_posCount + 5);
   }
   
   g_positions[idx].ticket         = ticket;
   g_positions[idx].entryPrice     = entry;
   g_positions[idx].originalSL     = sl;
   g_positions[idx].tp1Price       = tp1;
   g_positions[idx].tp2Price       = tp2;
   g_positions[idx].originalVolume = volume;
   g_positions[idx].state          = STATE_INITIAL;
   g_positions[idx].direction      = direction;
   g_positions[idx].entryTime      = TimeCurrent();
   g_positions[idx].active         = true;
   g_positions[idx].entryMethod    = method;
   g_positions[idx].session        = session;
   g_positions[idx].d1Bias         = d1Bias;
   g_positions[idx].h4Bias         = h4Bias;
   g_positions[idx].entryBalance   = AccountInfoDouble(ACCOUNT_BALANCE);
   
   string p = "MTFTR_" + IntegerToString(ticket);
   GlobalVariableSet(p + "_STATE", 0);
   GlobalVariableSet(p + "_ENTRY", entry);
   GlobalVariableSet(p + "_SL",    sl);
   GlobalVariableSet(p + "_TP1",   tp1);
   GlobalVariableSet(p + "_TP2",   tp2);
   GlobalVariableSet(p + "_VOL",   volume);
   GlobalVariableSet(p + "_DIR",   (double)direction);
   GlobalVariableSet(p + "_TIME",  (double)TimeCurrent());
}

void LoadExistingPositions()
{
   g_posCount = 0;
   ArrayResize(g_positions, 10);
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Magic() != InpMagicNumber || posInfo.Symbol() != _Symbol) continue;
      
      ulong ticket = posInfo.Ticket();
      string p = "MTFTR_" + IntegerToString(ticket);
      
      if(g_posCount >= ArraySize(g_positions))
         ArrayResize(g_positions, g_posCount + 5);
      
      g_positions[g_posCount].ticket    = ticket;
      g_positions[g_posCount].active    = true;
      g_positions[g_posCount].direction = (posInfo.PositionType() == POSITION_TYPE_BUY) ? 1 : -1;
      
      if(GlobalVariableCheck(p + "_STATE"))
      {
         g_positions[g_posCount].state          = (int)GlobalVariableGet(p + "_STATE");
         g_positions[g_posCount].entryPrice     = GlobalVariableGet(p + "_ENTRY");
         g_positions[g_posCount].originalSL     = GlobalVariableGet(p + "_SL");
         g_positions[g_posCount].tp1Price       = GlobalVariableGet(p + "_TP1");
         g_positions[g_posCount].tp2Price       = GlobalVariableGet(p + "_TP2");
         g_positions[g_posCount].originalVolume = GlobalVariableGet(p + "_VOL");
         g_positions[g_posCount].entryTime      = (datetime)(int)GlobalVariableGet(p + "_TIME");
      }
      else
      {
         double sl_dist = MathAbs(posInfo.PriceOpen() - posInfo.StopLoss());
         g_positions[g_posCount].entryPrice     = posInfo.PriceOpen();
         g_positions[g_posCount].originalSL     = posInfo.StopLoss();
         g_positions[g_posCount].originalVolume = posInfo.Volume();
         g_positions[g_posCount].entryTime      = posInfo.Time();
         g_positions[g_posCount].state          = STATE_INITIAL;
         g_positions[g_posCount].tp1Price = posInfo.PriceOpen() + (g_positions[g_posCount].direction * sl_dist * InpTP1_RR);
         g_positions[g_posCount].tp2Price = posInfo.PriceOpen() + (g_positions[g_posCount].direction * sl_dist * InpTP2_RR);
      }
      
      g_posCount++;
      Print("MTFTR: Loaded position #", ticket, " State: ", g_positions[g_posCount-1].state);
   }
}

void CleanupClosedPositions()
{
   for(int i = 0; i < g_posCount; i++)
   {
      if(!g_positions[i].active) continue;
      if(PositionSelectByTicket(g_positions[i].ticket)) continue;
      
      ulong ticket = g_positions[i].ticket;
      if(g_positions[i].state == STATE_INITIAL)
      {
         // Closed while in initial state = SL was hit before TP1 (full loss)
         consecLosses++;
         Print("MTFTR: Loss. Consecutive: ", consecLosses);
         LogEvent("SL_HIT", g_positions[i], g_positions[i].originalSL, "");
         if(consecLosses >= InpMaxConsecLosses)
         {
            dailyPauseActive = true;
            Print("MTFTR: Max consecutive losses. Pausing today.");
         }
      }
      else
      {
         consecLosses = 0;
         // TP1 was already hit; if still active at state >= TP1_HIT and now gone,
         // the trailing SL was triggered — log as TRAIL_EXIT with SL price as proxy
         if(g_positions[i].state == STATE_TP2_HIT)
            LogEvent("TRAIL_EXIT", g_positions[i], g_positions[i].originalSL, "SL triggered on trailing portion");
      }
      
      string p = "MTFTR_" + IntegerToString(ticket);
      GlobalVariableDel(p + "_STATE"); GlobalVariableDel(p + "_ENTRY");
      GlobalVariableDel(p + "_SL");    GlobalVariableDel(p + "_TP1");
      GlobalVariableDel(p + "_TP2");   GlobalVariableDel(p + "_VOL");
      GlobalVariableDel(p + "_DIR");   GlobalVariableDel(p + "_TIME");
      
      g_positions[i].active = false;
   }
}

void ResetDailyStats()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   int today = dt.year * 10000 + dt.mon * 100 + dt.day;
   if(today != lastTradeDay)
   {
      lastTradeDay     = today;
      dailyTradeCount  = 0;
      dailyPauseActive = false;
      dayStartBalance  = AccountInfoDouble(ACCOUNT_BALANCE);
      consecLosses     = 0;
      if(dt.day_of_week == 1 && lastTradeWeekDay != today)
      {
         weekStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
         lastTradeWeekDay = today;
      }
   }
}

bool CheckDailyLimits()
{
   if(dayStartBalance <= 0) return true;
   double dd = (dayStartBalance - AccountInfoDouble(ACCOUNT_BALANCE)) / dayStartBalance * 100.0;
   if(dd >= InpMaxDailyDD)
   {
      static datetime lp = 0;
      if(TimeCurrent() - lp > 300) { Print("MTFTR: Daily DD limit (", dd, "%)"); lp = TimeCurrent(); }
      return false;
   }
   return true;
}

bool CheckWeeklyLimits()
{
   if(weekStartBalance <= 0) return true;
   double dd = (weekStartBalance - AccountInfoDouble(ACCOUNT_BALANCE)) / weekStartBalance * 100.0;
   if(dd >= InpMaxWeeklyDD)
   {
      static datetime lp = 0;
      if(TimeCurrent() - lp > 300) { Print("MTFTR: Weekly DD limit (", dd, "%)"); lp = TimeCurrent(); }
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| LOGGING ENGINE v2.2 — Dual Layer: CSV + FastAPI                  |
//+------------------------------------------------------------------+

// Derive human-readable session string from current server time
string GetCurrentSession()
{
   MqlDateTime dt; TimeCurrent(dt);
   int gmt = dt.hour - InpGMTOffset;
   if(gmt < 0) gmt += 24;
   if(gmt >= 24) gmt -= 24;
   if(gmt >= InpLondonStart && gmt < InpLondonEnd) return "London";
   if(gmt >= InpNYStart     && gmt < InpNYEnd)     return "NY Overlap";
   if(gmt == 12 || gmt == 13)                       return "Lunch";
   return "Off-Hours";
}

// Convert bias enum to string
string BiasToString(ENUM_TREND_BIAS b)
{
   if(b == BIAS_LONG)  return "LONG";
   if(b == BIAS_SHORT) return "SHORT";
   return "NEUTRAL";
}

// Find the g_positions index for a ticket (-1 if not found)
int FindPositionIndex(ulong ticket)
{
   for(int i = 0; i < g_posCount; i++)
      if(g_positions[i].ticket == ticket && g_positions[i].active)
         return i;
   return -1;
}

// Determine outcome label from event type and position state
string GetOutcomeLabel(string eventType, const TrackedPosition &pos)
{
   if(eventType == "SL_HIT")    return "LOSS";
   if(eventType == "TP1_HIT")   return "WIN_PARTIAL";
   if(eventType == "TP2_HIT")   return "WIN_PARTIAL";
   if(eventType == "TRAIL_EXIT") return (pos.state >= STATE_TP1_HIT) ? "WIN" : "LOSS";
   if(eventType == "TIME_EXIT")
   {
      // Time exit: outcome depends on whether price moved in our favour
      // We log it as "PENDING" — the journal P&L column will show the actual result
      return "TIME_EXIT";
   }
   return "OPEN";
}

// Calculate P&L given exit price (0 = no exit yet)
double CalcPnL(const TrackedPosition &pos, double exitPrice)
{
   if(exitPrice <= 0) return 0;
   double rawPnL = (pos.direction == 1) ?
      (exitPrice - pos.entryPrice) * pos.originalVolume * 100.0 :
      (pos.entryPrice - exitPrice) * pos.originalVolume * 100.0;
   return NormalizeDouble(rawPnL, 2);
}

// Build the CSV row string for any event
// Build JSON payload for FastAPI
string BuildJSON(string eventType, const TrackedPosition &pos,
                 double exitPrice, string note)
{
   MqlDateTime dt; TimeCurrent(dt);
   string timestamp = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d",
                                   dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
   double pnl    = CalcPnL(pos, exitPrice);
   double slDist = MathAbs(pos.entryPrice - pos.originalSL);
   double rr     = (slDist > 0 && exitPrice > 0) ?
                   NormalizeDouble(pnl / (slDist * pos.originalVolume * 100.0), 2) : 0;
   string outcome = GetOutcomeLabel(eventType, pos);
   
   return StringFormat(
      "{"
      "\"timestamp\":\"%s\","
      "\"event\":\"%s\","
      "\"ticket\":%d,"
      "\"symbol\":\"%s\","
      "\"direction\":\"%s\","
      "\"method\":\"%s\","
      "\"session\":\"%s\","
      "\"entry\":%.5f,"
      "\"sl\":%.5f,"
      "\"tp1\":%.5f,"
      "\"tp2\":%.5f,"
      "\"lots\":%.2f,"
      "\"risk_pct\":%.1f,"
      "\"sl_dist\":%.5f,"
      "\"exit_price\":%.5f,"
      "\"pnl\":%.2f,"
      "\"rr\":%.2f,"
      "\"outcome\":\"%s\","
      "\"d1_bias\":\"%s\","
      "\"h4_bias\":\"%s\","
      "\"pos_state\":%d,"
      "\"balance\":%.2f,"
      "\"equity\":%.2f,"
      "\"note\":\"%s\""
      "}",
      timestamp, eventType, pos.ticket, _Symbol,
      (pos.direction == 1) ? "LONG" : "SHORT",
      pos.entryMethod, pos.session,
      pos.entryPrice, pos.originalSL, pos.tp1Price, pos.tp2Price,
      pos.originalVolume, InpRiskPercent, slDist,
      exitPrice, pnl, rr, outcome,
      pos.d1Bias, pos.h4Bias, pos.state,
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      note
   );
}

// Write one CSV row — append to Common file
void WriteCSV(string eventType, const TrackedPosition &pos, double exitPrice, string note)
{
   if(!InpEnableCSV) return;
   
   string fn = InpCSVFileName;
   int h = FileOpen(fn, FILE_WRITE | FILE_READ | FILE_CSV | FILE_COMMON, ',');
   if(h == INVALID_HANDLE)
   {
      Print("MTFTR LOG: Cannot open CSV file: ", fn);
      return;
   }
   
   // Write header if file is empty
   if(FileSize(h) == 0)
   {
      FileWrite(h,
         "Timestamp","Event","Ticket","Direction","Method","Session",
         "Entry","SL","TP1","TP2","Lots","Risk%","SL_Dist",
         "Exit_Price","PnL","RR","Outcome",
         "D1_Bias","H4_Bias","Pos_State",
         "Balance","Equity","Note");
   }
   
   FileSeek(h, 0, SEEK_END);
   
   MqlDateTime dt; TimeCurrent(dt);
   string ts = StringFormat("%04d.%02d.%02d %02d:%02d:%02d",
                            dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
   double pnl    = CalcPnL(pos, exitPrice);
   double slDist = MathAbs(pos.entryPrice - pos.originalSL);
   double rr     = (slDist > 0 && exitPrice > 0) ?
                   NormalizeDouble(pnl / (slDist * pos.originalVolume * 100.0), 2) : 0;
   string outcome = GetOutcomeLabel(eventType, pos);
   
   FileWrite(h,
      ts, eventType, (string)pos.ticket,
      (pos.direction == 1) ? "LONG" : "SHORT",
      pos.entryMethod, pos.session,
      DoubleToString(pos.entryPrice, _Digits),
      DoubleToString(pos.originalSL, _Digits),
      DoubleToString(pos.tp1Price, _Digits),
      DoubleToString(pos.tp2Price, _Digits),
      DoubleToString(pos.originalVolume, 2),
      DoubleToString(InpRiskPercent, 1),
      DoubleToString(slDist, _Digits),
      DoubleToString(exitPrice, _Digits),
      DoubleToString(pnl, 2),
      DoubleToString(rr, 2),
      outcome,
      pos.d1Bias, pos.h4Bias,
      (string)pos.state,
      DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
      DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
      note);
   
   FileClose(h);
}

// POST JSON to FastAPI — non-blocking: 2s timeout, silent on failure
void PostToFastAPI(string eventType, const TrackedPosition &pos,
                   double exitPrice, string note)
{
   if(!InpEnableFastAPI) return;
   
   string json    = BuildJSON(eventType, pos, exitPrice, note);
   string headers = "Content-Type: application/json\r\n";
   char   reqBody[], resBody[];
   string resHeaders;
   int    resCode;
   
   StringToCharArray(json, reqBody, 0, StringLen(json));
   
   resCode = WebRequest(
      "POST",
      InpFastAPIURL,
      headers,
      InpFastAPITimeout,
      reqBody,
      resBody,
      resHeaders
   );
   
   // Silent on failure — server may simply not be running
   if(resCode == -1)
   {
      // Only log at first failure per EA session to avoid log spam
      static bool warned = false;
      if(!warned)
      {
         Print("MTFTR LOG: FastAPI unreachable at ", InpFastAPIURL,
               " — CSV logging continues. Start mtftr_server.py to enable API logging.");
         warned = true;
      }
   }
   else if(resCode != 200 && resCode != 201)
   {
      Print("MTFTR LOG: FastAPI returned HTTP ", resCode, " for event ", eventType);
   }
}

// ── Master entry point ── called everywhere in the EA ──────────────
// eventType:  "OPEN" | "TP1_HIT" | "TP2_HIT" | "SL_HIT" | "TIME_EXIT" | "TRAIL_EXIT"
// exitPrice:  0.0 for OPEN events
// note:       optional free-text reason (e.g. "Hull MA flipped")
void LogEvent(string eventType, const TrackedPosition &pos,
              double exitPrice, string note)
{
   WriteCSV(eventType, pos, exitPrice, note);      // Layer 1: CSV (always)
   PostToFastAPI(eventType, pos, exitPrice, note); // Layer 2: FastAPI (optional)
}

//+------------------------------------------------------------------+
//| Visual Display                                                    |
//+------------------------------------------------------------------+
void UpdateVisualDisplay()
{
   datetime cb = iTime(_Symbol, PERIOD_M15, 0);
   if(cb == g_lastVisualUpdate) return;
   g_lastVisualUpdate = cb;
   
   ENUM_TREND_BIAS dailyBias = GetDailyTrendBias();
   ENUM_TREND_BIAS bias      = Get4HTrendBias();
   bool h1Confirmed = (bias != BIAS_NEUTRAL) ? Check1HConfirmation(bias) : false;
   
   double rsi[];
   ArraySetAsSeries(rsi, true);
   double rsiVal = 0;
   if(CopyBuffer(m15_rsi_handle, 0, 0, 1, rsi) >= 1) rsiVal = rsi[0];
   
   if(InpShowInfoPanel) UpdateInfoPanel(dailyBias, bias, h1Confirmed, rsiVal);
   if(InpShowHullBands) UpdateHullBands();
   if(InpShowEMAs)      UpdateEMALines();
   
   ChartRedraw(0);
}

void UpdateInfoPanel(ENUM_TREND_BIAS dailyBias, ENUM_TREND_BIAS bias, bool h1Confirmed, double rsi)
{
   int yPos = 30, yStep = 18;
   
   string bg = g_panelName + "_BG";
   if(ObjectFind(0, bg) < 0)
   {
      ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, bg, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
      ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, 20);
      ObjectSetInteger(0, bg, OBJPROP_XSIZE, 160);
      ObjectSetInteger(0, bg, OBJPROP_YSIZE, 220);
      ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, C'30,30,30');
      ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg, OBJPROP_BORDER_COLOR, clrDimGray);
   }
   
   CreateLabel(g_panelName+"_Title", "MTFTR v2.0", 155, yPos, clrWhite, 10, true); yPos += yStep+5;
   CreateLabel(g_panelName+"_Sep1", "-------------", 155, yPos, clrDimGray, 8, false); yPos += yStep;
   
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   CreateLabel(g_panelName+"_Acc",    "Account:",  155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_AccVal", "$"+DoubleToString(bal,2), 70, yPos, clrWhite, 9, false);
   yPos += yStep;
   
   double dpl = eq - dayStartBalance;
   CreateLabel(g_panelName+"_DPL",    "Daily P&L:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_DPLVal", ((dpl>=0)?"+":"")+DoubleToString(dpl,2), 70, yPos, (dpl>=0)?clrLime:clrRed, 9, false);
   yPos += yStep;
   
   CreateLabel(g_panelName+"_Trades",    "Trades:",   155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_TradesVal", IntegerToString(dailyTradeCount)+"/"+IntegerToString(InpMaxDailyTrades), 70, yPos, clrWhite, 9, false);
   yPos += yStep;
   
   CreateLabel(g_panelName+"_Sep2", "-------------", 155, yPos, clrDimGray, 8, false); yPos += yStep;
   
   bool inSess = IsWithinSession();
   CreateLabel(g_panelName+"_Sess",    "Session:",  155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_SessVal", inSess?"ACTIVE":"CLOSED", 70, yPos, inSess?clrLime:clrGray, 9, false);
   yPos += yStep;
   
   // Daily bias row (NEW)
   string dStr = (dailyBias==BIAS_LONG)?"BULL D1":(dailyBias==BIAS_SHORT)?"BEAR D1":"NEUTRAL";
   color  dClr = (dailyBias==BIAS_LONG)?InpBullColor:(dailyBias==BIAS_SHORT)?InpBearColor:InpNeutralColor;
   CreateLabel(g_panelName+"_D1",    "D1 Bias:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_D1Val", dStr,       70,  yPos, dClr,  9, true);
   yPos += yStep;
   
   string bStr = (bias==BIAS_LONG)?"BULL 4H":(bias==BIAS_SHORT)?"BEAR 4H":"NEUTRAL";
   color  bClr = (bias==BIAS_LONG)?InpBullColor:(bias==BIAS_SHORT)?InpBearColor:InpNeutralColor;
   CreateLabel(g_panelName+"_MTF",    "4H Trend:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_MTFVal", bStr,        70,  yPos, bClr, 9, true);
   yPos += yStep;
   
   string cStr = (bias==BIAS_NEUTRAL)?"N/A":(h1Confirmed?"YES":"NO");
   color  cClr = (bias==BIAS_NEUTRAL)?clrGray:(h1Confirmed?clrLime:clrOrange);
   CreateLabel(g_panelName+"_Conf",    "1H Confirm:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_ConfVal", cStr,          70,  yPos, cClr, 9, false);
   yPos += yStep;
   
   int trailCnt = 0, beCnt = 0;
   for(int i = 0; i < g_posCount; i++)
   {
      if(!g_positions[i].active) continue;
      if(g_positions[i].state == STATE_TP2_HIT) trailCnt++;
      if(g_positions[i].state >= STATE_TP1_HIT) beCnt++;
   }
   
   CreateLabel(g_panelName+"_Trail",    "Trailing:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_TrailVal", trailCnt>0?"ON":"OFF", 70, yPos, trailCnt>0?clrLime:clrGray, 9, false);
   yPos += yStep;
   
   CreateLabel(g_panelName+"_BE",    "Breakeven:", 155, yPos, clrGray, 9, false);
   CreateLabel(g_panelName+"_BEVal", beCnt>0?"ON":"OFF", 70, yPos, beCnt>0?clrLime:clrGray, 9, false);
}

void CreateLabel(string name, string text, int xDist, int yDist, color clr, int fontSize, bool bold)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT);
   }
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xDist);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yDist);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
}

void UpdateHullBands()
{
   int n = 100;
   double hull[];
   ArrayResize(hull, n);
   for(int i = 0; i < n; i++)
      hull[i] = GetHullMAValue(h1_wma_half_handle, h1_wma_full_handle, h1_hull_sqrt, i);
   
   double atr[];
   ArraySetAsSeries(atr, true);
   if(CopyBuffer(h4_atr_handle, 0, 0, 1, atr) < 1) return;
   double bw = atr[0] * 0.3;
   
   datetime time[];
   ArraySetAsSeries(time, true);
   if(CopyTime(_Symbol, PERIOD_M15, 0, n, time) < n) return;
   
   for(int i = 1; i < n - 1; i++)
   {
      if(hull[i] == 0 || hull[i-1] == 0) continue;
      bool bull = (hull[i-1] > hull[i]);
      color clr = bull ? InpBullColor : InpBearColor;
      
      string mid = g_hullPrefix+"M_"+IntegerToString(i);
      string up  = g_hullPrefix+"U_"+IntegerToString(i);
      string dn  = g_hullPrefix+"L_"+IntegerToString(i);
      
      // Mid line
      if(ObjectFind(0, mid) < 0) ObjectCreate(0, mid, OBJ_TREND, 0, time[i], hull[i], time[i-1], hull[i-1]);
      else { ObjectSetInteger(0,mid,OBJPROP_TIME,0,time[i]); ObjectSetDouble(0,mid,OBJPROP_PRICE,0,hull[i]);
             ObjectSetInteger(0,mid,OBJPROP_TIME,1,time[i-1]); ObjectSetDouble(0,mid,OBJPROP_PRICE,1,hull[i-1]); }
      ObjectSetInteger(0,mid,OBJPROP_COLOR,clr); ObjectSetInteger(0,mid,OBJPROP_WIDTH,2);
      ObjectSetInteger(0,mid,OBJPROP_STYLE,STYLE_SOLID); ObjectSetInteger(0,mid,OBJPROP_RAY_RIGHT,false);
      ObjectSetInteger(0,mid,OBJPROP_BACK,true);
      
      // Upper band
      if(ObjectFind(0, up) < 0) ObjectCreate(0, up, OBJ_TREND, 0, time[i], hull[i]+bw, time[i-1], hull[i-1]+bw);
      else { ObjectSetInteger(0,up,OBJPROP_TIME,0,time[i]); ObjectSetDouble(0,up,OBJPROP_PRICE,0,hull[i]+bw);
             ObjectSetInteger(0,up,OBJPROP_TIME,1,time[i-1]); ObjectSetDouble(0,up,OBJPROP_PRICE,1,hull[i-1]+bw); }
      ObjectSetInteger(0,up,OBJPROP_COLOR,clr); ObjectSetInteger(0,up,OBJPROP_WIDTH,1);
      ObjectSetInteger(0,up,OBJPROP_STYLE,STYLE_DOT); ObjectSetInteger(0,up,OBJPROP_RAY_RIGHT,false);
      ObjectSetInteger(0,up,OBJPROP_BACK,true);
      
      // Lower band
      if(ObjectFind(0, dn) < 0) ObjectCreate(0, dn, OBJ_TREND, 0, time[i], hull[i]-bw, time[i-1], hull[i-1]-bw);
      else { ObjectSetInteger(0,dn,OBJPROP_TIME,0,time[i]); ObjectSetDouble(0,dn,OBJPROP_PRICE,0,hull[i]-bw);
             ObjectSetInteger(0,dn,OBJPROP_TIME,1,time[i-1]); ObjectSetDouble(0,dn,OBJPROP_PRICE,1,hull[i-1]-bw); }
      ObjectSetInteger(0,dn,OBJPROP_COLOR,clr); ObjectSetInteger(0,dn,OBJPROP_WIDTH,1);
      ObjectSetInteger(0,dn,OBJPROP_STYLE,STYLE_DOT); ObjectSetInteger(0,dn,OBJPROP_RAY_RIGHT,false);
      ObjectSetInteger(0,dn,OBJPROP_BACK,true);
   }
}

void UpdateEMALines()
{
   int n = 200;
   datetime time[];
   ArraySetAsSeries(time, true);
   if(CopyTime(_Symbol, PERIOD_M15, 0, n, time) < n) return;
   
   // 4H EMA200 mapped to M15 (1 4H bar = 16 M15 bars)
   double ema200[];
   ArraySetAsSeries(ema200, true);
   if(CopyBuffer(h4_ema200_handle, 0, 0, 50, ema200) >= 50)
   {
      for(int i = 1; i < 50; i++)
      {
         int s = i*16, e = (i-1)*16;
         if(s >= n || e >= n) continue;
         string nm = g_emaPrefix+"200_"+IntegerToString(i);
         if(ObjectFind(0,nm) < 0) ObjectCreate(0,nm,OBJ_TREND,0,time[s],ema200[i],time[e],ema200[i-1]);
         else { ObjectSetInteger(0,nm,OBJPROP_TIME,0,time[s]); ObjectSetDouble(0,nm,OBJPROP_PRICE,0,ema200[i]);
                ObjectSetInteger(0,nm,OBJPROP_TIME,1,time[e]); ObjectSetDouble(0,nm,OBJPROP_PRICE,1,ema200[i-1]); }
         ObjectSetInteger(0,nm,OBJPROP_COLOR,InpEMA200Color); ObjectSetInteger(0,nm,OBJPROP_WIDTH,2);
         ObjectSetInteger(0,nm,OBJPROP_RAY_RIGHT,false); ObjectSetInteger(0,nm,OBJPROP_BACK,true);
      }
   }
   
   // 1H EMA50 (1 1H bar = 4 M15 bars)
   double ema50[];
   ArraySetAsSeries(ema50, true);
   if(CopyBuffer(h1_ema50_handle, 0, 0, 50, ema50) >= 50)
   {
      for(int i = 1; i < 50; i++)
      {
         int s = i*4, e = (i-1)*4;
         if(s >= n || e >= n) continue;
         string nm = g_emaPrefix+"50_"+IntegerToString(i);
         if(ObjectFind(0,nm) < 0) ObjectCreate(0,nm,OBJ_TREND,0,time[s],ema50[i],time[e],ema50[i-1]);
         else { ObjectSetInteger(0,nm,OBJPROP_TIME,0,time[s]); ObjectSetDouble(0,nm,OBJPROP_PRICE,0,ema50[i]);
                ObjectSetInteger(0,nm,OBJPROP_TIME,1,time[e]); ObjectSetDouble(0,nm,OBJPROP_PRICE,1,ema50[i-1]); }
         ObjectSetInteger(0,nm,OBJPROP_COLOR,InpEMA50Color); ObjectSetInteger(0,nm,OBJPROP_WIDTH,1);
         ObjectSetInteger(0,nm,OBJPROP_RAY_RIGHT,false); ObjectSetInteger(0,nm,OBJPROP_BACK,true);
      }
   }
   
   // 15M EMA21
   double ema21[];
   ArraySetAsSeries(ema21, true);
   if(CopyBuffer(m15_ema21_handle, 0, 0, n, ema21) >= n)
   {
      for(int i = 1; i < n-1; i++)
      {
         string nm = g_emaPrefix+"21_"+IntegerToString(i);
         if(ObjectFind(0,nm) < 0) ObjectCreate(0,nm,OBJ_TREND,0,time[i],ema21[i],time[i-1],ema21[i-1]);
         else { ObjectSetInteger(0,nm,OBJPROP_TIME,0,time[i]); ObjectSetDouble(0,nm,OBJPROP_PRICE,0,ema21[i]);
                ObjectSetInteger(0,nm,OBJPROP_TIME,1,time[i-1]); ObjectSetDouble(0,nm,OBJPROP_PRICE,1,ema21[i-1]); }
         ObjectSetInteger(0,nm,OBJPROP_COLOR,InpEMA21Color); ObjectSetInteger(0,nm,OBJPROP_WIDTH,1);
         ObjectSetInteger(0,nm,OBJPROP_RAY_RIGHT,false); ObjectSetInteger(0,nm,OBJPROP_BACK,true);
      }
   }
}

void CleanupVisualObjects()
{
   ObjectsDeleteAll(0, g_panelName);
   ObjectsDeleteAll(0, g_hullPrefix);
   ObjectsDeleteAll(0, g_emaPrefix);
}
//+------------------------------------------------------------------+