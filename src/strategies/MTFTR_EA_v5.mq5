//+------------------------------------------------------------------+
//| MTFTR_EA_v5.mq5                                                   |
//| Multi-Timeframe Trend Rider Expert Advisor                        |
//| Version: 5.0  (Disciplined — 1:4 to 1:6 R:R targets)             |
//+------------------------------------------------------------------+
//| v5.0 CHANGES FROM v4.0:                                           |
//|                                                                   |
//| PHILOSOPHY SHIFT:                                                 |
//|   v4.0 chased capacity. v5.0 chases edge. The Universal variant   |
//|   blew a $1k account from 2020-2026 by stripping filters in       |
//|   pursuit of more setups. v5.0 does the opposite: tighter         |
//|   filters, smaller size, longer holding times, higher RR.         |
//|                                                                   |
//| TP STRUCTURE (target 1:4-1:6 R:R blended winners):                |
//|   - TP1 at 2.0R, close 20% (was 1.5R / 25%)                       |
//|   - TP2 at 4.0R, close 30% (was 3.0R / 35%)                       |
//|   - TP3 at 6.0R, close remaining 50% (was trail 40%)              |
//|   - Best-case blended: 20%×2R + 30%×4R + 50%×6R = 4.6R            |
//|                                                                   |
//| BE / TRAIL TIMING:                                                |
//|   - After TP1: SL to entry (pure BE) — was +1.0R                  |
//|     ONLY after partial profit is banked. Never move SL before     |
//|     locking some R.                                               |
//|   - After TP2: SL to +2.0R (unchanged)                            |
//|   - BE offset fix: nudge BE by 0.1×M15 ATR to satisfy broker      |
//|     stop-level check when market sits at entry.                   |
//|                                                                   |
//| ENTRY QUALITY (tighter, not looser):                              |
//|   - ADX > 25 on H1 (was 20). Range markets killed v4 win rate.    |
//|   - RSI bands narrowed: 40-50 long, 50-60 short (was 40-55/45-65) |
//|   - HARD D1 alignment: D1 bias must MATCH H4 (was: D1 allowed     |
//|     neutral). No counter-D1 trades, no D1-neutral trades.         |
//|   - Min SL: 1.2 ATR (was 1.0) — slight breathing room             |
//|                                                                   |
//| RISK / CAPACITY (cut until data justifies more):                  |
//|   - Risk: 1% per trade (was 2%)                                   |
//|   - MaxOpenTrades: 2 (was 3)                                      |
//|   - MaxDailyTrades: 3 (was 5)                                     |
//|   - MaxTradeHours: 24 (was 16) — 6R targets need time             |
//|                                                                   |
//| RETAINED FROM v4.0:                                               |
//|   - Error 10018 cooldown, per-position H1/H4 tracking             |
//|   - Smart time exit, proximity filter, lastKnownSL                |
//|   - Safety state persistence across restarts                      |
//|   - Monthly withdrawal simulation & CSV logging                   |
//|   - Pivoted away from Fibonacci-by-default; still optional        |
//+------------------------------------------------------------------+
#property copyright "MTFTR Strategy v5.0"
#property version   "5.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//| Enumerations                                                      |
//+------------------------------------------------------------------+
enum ENUM_TREND_BIAS   { BIAS_LONG=1, BIAS_SHORT=-1, BIAS_NEUTRAL=0 };
enum ENUM_ENTRY_METHOD { ENTRY_NONE=0, ENTRY_EMA_BOUNCE=1, ENTRY_STRUCTURE=2, ENTRY_FIBONACCI=3 };
enum ENUM_POS_STATE    { STATE_INITIAL=0, STATE_TP1_HIT=1, STATE_TP2_HIT=2, STATE_TP3_TRAIL=3 };

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "====== Risk Management [v5: disciplined] ======"
input double   InpRiskPercent      = 1.0;       // Risk per trade (%) [v5: was 2.0]
input int      InpMaxOpenTrades    = 2;         // Max simultaneous trades [v5: was 3]
input int      InpMaxDailyTrades   = 3;         // Max trades per day [v5: was 5]
input int      InpMaxConsecLosses  = 3;         // Consecutive losses -> pause [v5: was 4]
input double   InpMaxDailyDD       = 3.0;       // Max daily drawdown (%) [v5: was 4.0]
input double   InpMaxWeeklyDD      = 6.0;       // Max weekly drawdown (%) [v5: was 8.0]
input double   InpMaxSpreadATR     = 0.15;      // Max spread as fraction of 15M ATR

input group "====== Monthly Withdrawal ======"
input bool     InpEnableWithdrawal = true;       // Simulate monthly withdrawal
input double   InpWithdrawPct      = 70.0;       // % of monthly profit to withdraw
input bool     InpLogWithdrawals   = true;       // Log withdrawal events to CSV

input group "====== Daily Master Bias [v5: HARD alignment] ======"
input int      InpD1_EMA_Period    = 50;
input int      InpD1_Hull_Period   = 21;
input bool     InpRequireD1Align   = true;       // [v5] HARD: D1 must match H4 (no neutral D1)

input group "====== 4H Trend Settings ======"
input int      InpH4_EMA_Period    = 200;
input int      InpH4_Hull_Period   = 55;
input int      InpH4_ATR_Period    = 14;
input int      InpH4_SlopeBars     = 5;
input double   InpH4_SlopeMin      = 0.5;
input double   InpH4_NoTradeATR    = 1.5;
input int      InpH4_HullMinBars   = 2;

input group "====== 1H Confirmation ======"
input int      InpH1_EMA_Period    = 50;
input int      InpH1_Hull_Period   = 34;

input group "====== ADX Trend Strength Filter [v5: tighter] ======"
input bool     InpADXEnabled       = true;       // Enable ADX filter (kills ranging chop)
input int      InpADX_Period       = 14;
input double   InpADX_Min          = 25.0;       // Min ADX for entry [v5: was 20]
input ENUM_TIMEFRAMES InpADX_TF    = PERIOD_H1;

input group "====== 15M Entry Settings [v5: tighter RSI] ======"
input int      InpM15_EMA_Period   = 21;
input int      InpM15_RSI_Period   = 14;
input int      InpM15_ATR_Period   = 14;
input double   InpRSI_Long_Min     = 40.0;       // [v5: unchanged]
input double   InpRSI_Long_Max     = 50.0;       // [v5: was 55] — tighter
input double   InpRSI_Short_Min    = 50.0;       // [v5: was 45] — tighter
input double   InpRSI_Short_Max    = 60.0;       // [v5: was 65] — tighter
input int      InpSwingLookback    = 3;

input group "====== Entry Precision ======"
input double   InpProximityATR     = 0.5;
input int      InpSwingLookbackSL  = 5;

input group "====== Fibonacci Entry (Method C) — optional ======"
input bool     InpFibEnabled       = false;      // [v5] disabled by default; v4 forward-test showed Fib added noise
input double   InpFibLevel1        = 0.500;
input double   InpFibLevel2        = 0.618;
input double   InpFibTolerance     = 0.010;
input int      InpFibSwingBarsH1   = 50;
input double   InpFibRSI_Long_Min  = 35.0;
input double   InpFibRSI_Long_Max  = 55.0;
input double   InpFibRSI_Short_Min = 45.0;
input double   InpFibRSI_Short_Max = 65.0;

input group "====== Stop Loss [v5: slightly wider min] ======"
input double   InpSL_ATR_Buffer    = 0.3;
input double   InpSL_Max_ATR       = 3.0;
input double   InpSL_Min_ATR       = 1.2;       // [v5: was 1.0] — breathing room

input group "====== 3-Tier Take Profit [v5: 2/4/6R, 20/30/50%] ======"
input double   InpTP1_RR           = 2.0;       // TP1 R:R [v5: was 1.5]
input double   InpTP2_RR           = 4.0;       // TP2 R:R [v5: was 3.0]
input double   InpTP3_RR           = 6.0;       // TP3 R:R (full close)
input double   InpTP1_ClosePct     = 20.0;      // % at TP1 [v5: was 25]
input double   InpTP2_ClosePct     = 30.0;      // % at TP2 [v5: was 35]
input double   InpTP1_SL_MoveToRR  = 0.0;       // After TP1: SL to entry (BE) [v5: was 1.0]
input double   InpTP2_SL_MoveToRR  = 2.0;       // After TP2: SL to +2.0R [unchanged]
input double   InpBE_OffsetATR     = 0.1;       // [v5] BE offset in M15 ATR (broker stop-level safety)
input int      InpMaxTradeHours    = 24;        // Time exit hours [v5: was 16]
input bool     InpSmartTimeExit    = true;

input group "====== Session Filter ======"
input int      InpGMTOffset        = 0;
input int      InpSessionStart     = 7;
input int      InpSessionEnd       = 16;
input bool     InpFridayFilter     = true;

input group "====== General ======"
input int      InpMagicNumber      = 20260526;  // [v5] new magic

input group "====== Logging ======"
input bool     InpEnableCSV        = true;
input string   InpCSVFileName      = "MTFTR_v5_TradeLog.csv";
input bool     InpEnableFastAPI    = false;
input string   InpFastAPIURL       = "http://127.0.0.1:8000/trade";
input int      InpFastAPITimeout   = 2000;

input group "====== Visual ======"
input bool     InpShowInfoPanel    = true;
input color    InpBullColor        = clrLime;
input color    InpBearColor        = clrRed;
input color    InpNeutralColor     = clrGray;

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade trade; CPositionInfo posInfo;

int d1_ema50_handle, d1_wma_half_handle, d1_wma_full_handle;
int h4_ema200_handle, h4_wma_half_handle, h4_wma_full_handle, h4_atr_handle;
int h1_ema50_handle, h1_wma_half_handle, h1_wma_full_handle;
int m15_ema21_handle, m15_rsi_handle, m15_atr_handle;
int h1_adx_handle;

int d1_hull_sqrt, h4_hull_sqrt, h1_hull_sqrt;
datetime lastBar_M15=0, lastBar_H4=0;

int    dailyTradeCount=0, consecLosses=0;
double dayStartBalance=0, weekStartBalance=0;
int    lastTradeDay=0, lastTradeWeekDay=0;
bool   dailyPauseActive=false;

int    currentMonth = 0;
double monthStartBalance = 0;
double totalWithdrawn = 0;

struct TrackedPosition
{
   ulong    ticket;
   double   entryPrice, originalSL, tp1Price, tp2Price, tp3Price;
   double   originalVolume, slDistance, lastKnownSL, entryBalance;
   int      state, direction;
   datetime entryTime, lastH1Check, lastH4Check, cooldownUntil;
   bool     active, timeExtended, closeCooldown;
   string   entryMethod, session, d1Bias, h4Bias;
};

TrackedPosition g_positions[];
int g_posCount = 0;
string g_panelName = "MTFTR5_Panel";
string g_safetyPrefix = "";

//+------------------------------------------------------------------+
//| Initialization                                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(30);

   long fillMode = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if(fillMode & SYMBOL_FILLING_FOK)      trade.SetTypeFilling(ORDER_FILLING_FOK);
   else if(fillMode & SYMBOL_FILLING_IOC) trade.SetTypeFilling(ORDER_FILLING_IOC);
   else                                    trade.SetTypeFilling(ORDER_FILLING_RETURN);

   d1_hull_sqrt = (int)MathSqrt(InpD1_Hull_Period);
   h4_hull_sqrt = (int)MathSqrt(InpH4_Hull_Period);
   h1_hull_sqrt = (int)MathSqrt(InpH1_Hull_Period);

   d1_ema50_handle    = iMA(_Symbol,PERIOD_D1,InpD1_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   d1_wma_half_handle = iMA(_Symbol,PERIOD_D1,(int)(InpD1_Hull_Period/2.0),0,MODE_LWMA,PRICE_CLOSE);
   d1_wma_full_handle = iMA(_Symbol,PERIOD_D1,InpD1_Hull_Period,0,MODE_LWMA,PRICE_CLOSE);
   h4_ema200_handle   = iMA(_Symbol,PERIOD_H4,InpH4_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   h4_wma_half_handle = iMA(_Symbol,PERIOD_H4,(int)(InpH4_Hull_Period/2.0),0,MODE_LWMA,PRICE_CLOSE);
   h4_wma_full_handle = iMA(_Symbol,PERIOD_H4,InpH4_Hull_Period,0,MODE_LWMA,PRICE_CLOSE);
   h4_atr_handle      = iATR(_Symbol,PERIOD_H4,InpH4_ATR_Period);
   h1_ema50_handle    = iMA(_Symbol,PERIOD_H1,InpH1_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   h1_wma_half_handle = iMA(_Symbol,PERIOD_H1,(int)(InpH1_Hull_Period/2.0),0,MODE_LWMA,PRICE_CLOSE);
   h1_wma_full_handle = iMA(_Symbol,PERIOD_H1,InpH1_Hull_Period,0,MODE_LWMA,PRICE_CLOSE);
   m15_ema21_handle   = iMA(_Symbol,PERIOD_M15,InpM15_EMA_Period,0,MODE_EMA,PRICE_CLOSE);
   m15_rsi_handle     = iRSI(_Symbol,PERIOD_M15,InpM15_RSI_Period,PRICE_CLOSE);
   m15_atr_handle     = iATR(_Symbol,PERIOD_M15,InpM15_ATR_Period);
   h1_adx_handle      = iADX(_Symbol,InpADX_TF,InpADX_Period);

   if(d1_ema50_handle==INVALID_HANDLE || d1_wma_half_handle==INVALID_HANDLE ||
      d1_wma_full_handle==INVALID_HANDLE || h4_ema200_handle==INVALID_HANDLE ||
      h4_wma_half_handle==INVALID_HANDLE || h4_wma_full_handle==INVALID_HANDLE ||
      h4_atr_handle==INVALID_HANDLE || h1_ema50_handle==INVALID_HANDLE ||
      h1_wma_half_handle==INVALID_HANDLE || h1_wma_full_handle==INVALID_HANDLE ||
      m15_ema21_handle==INVALID_HANDLE || m15_rsi_handle==INVALID_HANDLE ||
      m15_atr_handle==INVALID_HANDLE || h1_adx_handle==INVALID_HANDLE)
   { Print("MTFTR v5 ERROR: Indicator handles failed"); return INIT_FAILED; }

   dayStartBalance = weekStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   monthStartBalance = dayStartBalance;
   MqlDateTime dt; TimeCurrent(dt); currentMonth = dt.year*100+dt.mon;
   g_safetyPrefix = "MTFTR5_S_" + _Symbol + "_";

   ArrayResize(g_positions,10);
   LoadExistingPositions();
   LoadSafetyState();

   Print("MTFTR v5.0 DISCIPLINED | Risk=",InpRiskPercent,"% | TP=",InpTP1_RR,"/",InpTP2_RR,"/",InpTP3_RR,
         "R | Close=",InpTP1_ClosePct,"/",InpTP2_ClosePct,"/",(100-InpTP1_ClosePct-InpTP2_ClosePct),"%",
         " | ADX>",InpADX_Min," | RSI L:",InpRSI_Long_Min,"-",InpRSI_Long_Max,
         " S:",InpRSI_Short_Min,"-",InpRSI_Short_Max,
         " | D1Hard=",InpRequireD1Align," | Open=",InpMaxOpenTrades,"/Day=",InpMaxDailyTrades);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   SaveSafetyState();
   IndicatorRelease(d1_ema50_handle);   IndicatorRelease(d1_wma_half_handle);
   IndicatorRelease(d1_wma_full_handle); IndicatorRelease(h4_ema200_handle);
   IndicatorRelease(h4_wma_half_handle); IndicatorRelease(h4_wma_full_handle);
   IndicatorRelease(h4_atr_handle);      IndicatorRelease(h1_ema50_handle);
   IndicatorRelease(h1_wma_half_handle); IndicatorRelease(h1_wma_full_handle);
   IndicatorRelease(m15_ema21_handle);   IndicatorRelease(m15_rsi_handle);
   IndicatorRelease(m15_atr_handle);     IndicatorRelease(h1_adx_handle);
   ObjectsDeleteAll(0, g_panelName);
}

//+------------------------------------------------------------------+
//| Safety State                                                      |
//+------------------------------------------------------------------+
void SaveSafetyState()
{
   string p=g_safetyPrefix;
   GlobalVariableSet(p+"cL",(double)consecLosses);
   GlobalVariableSet(p+"dTC",(double)dailyTradeCount);
   GlobalVariableSet(p+"dP",dailyPauseActive?1:0);
   GlobalVariableSet(p+"lTD",(double)lastTradeDay);
   GlobalVariableSet(p+"dSB",dayStartBalance);
   GlobalVariableSet(p+"wSB",weekStartBalance);
   GlobalVariableSet(p+"mSB",monthStartBalance);
   GlobalVariableSet(p+"tW",totalWithdrawn);
   GlobalVariableSet(p+"cM",(double)currentMonth);
}

void LoadSafetyState()
{
   string p=g_safetyPrefix;
   if(!GlobalVariableCheck(p+"lTD")) return;
   MqlDateTime dt; TimeCurrent(dt);
   int today=dt.year*10000+dt.mon*100+dt.day;
   if((int)GlobalVariableGet(p+"lTD")!=today) return;
   consecLosses=(int)GlobalVariableGet(p+"cL");
   dailyTradeCount=(int)GlobalVariableGet(p+"dTC");
   dailyPauseActive=GlobalVariableGet(p+"dP")>0.5;
   dayStartBalance=GlobalVariableGet(p+"dSB");
   lastTradeDay=today;
   if(GlobalVariableCheck(p+"wSB")) weekStartBalance=GlobalVariableGet(p+"wSB");
   if(GlobalVariableCheck(p+"mSB")) monthStartBalance=GlobalVariableGet(p+"mSB");
   if(GlobalVariableCheck(p+"tW"))  totalWithdrawn=GlobalVariableGet(p+"tW");
   if(GlobalVariableCheck(p+"cM"))  currentMonth=(int)GlobalVariableGet(p+"cM");
}

//+------------------------------------------------------------------+
//| Monthly Withdrawal Simulation                                     |
//+------------------------------------------------------------------+
void CheckMonthlyWithdrawal()
{
   if(!InpEnableWithdrawal) return;
   MqlDateTime dt; TimeCurrent(dt);
   int thisMonth = dt.year*100+dt.mon;

   if(thisMonth != currentMonth)
   {
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double monthProfit = balance - monthStartBalance;

      if(monthProfit > 0)
      {
         double withdrawal = monthProfit * InpWithdrawPct / 100.0;
         totalWithdrawn += withdrawal;
         Print("MTFTR v5 WITHDRAWAL: Month ", currentMonth,
               " | Profit: $", DoubleToString(monthProfit, 2),
               " | Withdraw ",InpWithdrawPct,"%: $", DoubleToString(withdrawal, 2),
               " | Total withdrawn: $", DoubleToString(totalWithdrawn, 2),
               " | Account: $", DoubleToString(balance, 2));
         if(InpLogWithdrawals && InpEnableCSV)
            LogWithdrawal(currentMonth, monthProfit, withdrawal, balance);
      }
      else
      {
         Print("MTFTR v5: Month ", currentMonth, " closed at loss: $",
               DoubleToString(monthProfit, 2), " — no withdrawal");
      }

      currentMonth = thisMonth;
      monthStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      SaveSafetyState();
   }
}

void LogWithdrawal(int month, double profit, double withdrawal, double balance)
{
   int h = FileOpen(InpCSVFileName, FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON, ',');
   if(h == INVALID_HANDLE) return;
   FileSeek(h, 0, SEEK_END);
   MqlDateTime dt; TimeCurrent(dt);
   FileWrite(h,
      StringFormat("%04d.%02d.%02d %02d:%02d:%02d", dt.year,dt.mon,dt.day,dt.hour,dt.min,dt.sec),
      "WITHDRAWAL", (string)month, "---", "---", "---",
      "---","---","---","---","---","---","---",
      DoubleToString(withdrawal,2), DoubleToString(profit,2),
      "---", "WITHDRAWAL",
      "---","---","---",
      DoubleToString(balance,2), DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2),
      "TotalWithdrawn=$"+DoubleToString(totalWithdrawn,2));
   FileClose(h);
}

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageOpenPositions();
   CleanupClosedPositions();

   if(!IsNewBar(PERIOD_M15)) return;

   ResetDailyStats();
   CheckMonthlyWithdrawal();
   UpdateInfoPanel();

   if(dailyPauseActive) return;
   if(!CheckDailyLimits()) return;
   if(!CheckWeeklyLimits()) return;
   if(CountOpenPositions() >= InpMaxOpenTrades) return;
   if(dailyTradeCount >= InpMaxDailyTrades) return;
   if(!IsWithinSession()) return;

   // ADX trending filter
   if(InpADXEnabled)
   {
      double adx[]; ArraySetAsSeries(adx,true);
      if(CopyBuffer(h1_adx_handle,0,0,1,adx)<1) return;
      if(adx[0] < InpADX_Min) return;
   }

   ENUM_TREND_BIAS dailyBias = GetDailyTrendBias();
   ENUM_TREND_BIAS bias = Get4HTrendBias();
   if(bias == BIAS_NEUTRAL) return;

   // [v5] HARD D1 alignment: when enabled, D1 must match H4 exactly.
   // No D1-neutral trades allowed (v4 permitted them; data says they degrade quality).
   if(InpRequireD1Align)
   {
      if(dailyBias != bias) return;
   }
   else
   {
      // Legacy v4 behaviour
      if(dailyBias != BIAS_NEUTRAL && dailyBias != bias) return;
   }

   if(!Check1HConfirmation(bias)) return;

   ENUM_ENTRY_METHOD method = Check15MEntry(bias);
   if(method == ENTRY_NONE) return;

   double ema21Price = CheckProximity(bias);
   if(ema21Price == 0) return;

   ExecuteTrade(bias, method);
}

//+------------------------------------------------------------------+
//| Bias Functions — bar[1] confirmed only                            |
//+------------------------------------------------------------------+
ENUM_TREND_BIAS GetDailyTrendBias()
{
   double ema[]; ArraySetAsSeries(ema,true);
   if(CopyBuffer(d1_ema50_handle,0,1,1,ema)<1) return BIAS_NEUTRAL;
   double cl[]; ArraySetAsSeries(cl,true);
   if(CopyClose(_Symbol,PERIOD_D1,0,2,cl)<2) return BIAS_NEUTRAL;
   bool hB=IsHullBull(d1_wma_half_handle,d1_wma_full_handle,d1_hull_sqrt);
   bool hR=IsHullBear(d1_wma_half_handle,d1_wma_full_handle,d1_hull_sqrt);
   if(cl[1]>ema[0]&&hB) return BIAS_LONG;
   if(cl[1]<ema[0]&&hR) return BIAS_SHORT;
   return BIAS_NEUTRAL;
}

ENUM_TREND_BIAS Get4HTrendBias()
{
   double ema[]; ArraySetAsSeries(ema,true);
   if(CopyBuffer(h4_ema200_handle,0,1,InpH4_SlopeBars+1,ema)<InpH4_SlopeBars+1) return BIAS_NEUTRAL;
   double slope=ema[0]-ema[InpH4_SlopeBars];
   double cl[]; ArraySetAsSeries(cl,true);
   if(CopyClose(_Symbol,PERIOD_H4,0,2,cl)<2) return BIAS_NEUTRAL;
   double atr[]; ArraySetAsSeries(atr,true);
   if(CopyBuffer(h4_atr_handle,0,1,1,atr)<1) return BIAS_NEUTRAL;
   if(MathAbs(cl[1]-ema[0])<InpH4_NoTradeATR*atr[0]) return BIAS_NEUTRAL;
   if(MathAbs(slope)<InpH4_SlopeMin) return BIAS_NEUTRAL;
   bool hB=IsHullBull(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt);
   bool hR=IsHullBear(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt);
   if(hB && HullConsec(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt,true,InpH4_HullMinBars+3)<InpH4_HullMinBars) return BIAS_NEUTRAL;
   if(hR && HullConsec(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt,false,InpH4_HullMinBars+3)<InpH4_HullMinBars) return BIAS_NEUTRAL;
   if(cl[1]>ema[0]&&slope>InpH4_SlopeMin&&hB) return BIAS_LONG;
   if(cl[1]<ema[0]&&slope<-InpH4_SlopeMin&&hR) return BIAS_SHORT;
   return BIAS_NEUTRAL;
}

bool Check1HConfirmation(ENUM_TREND_BIAS bias)
{
   double ema[]; ArraySetAsSeries(ema,true);
   if(CopyBuffer(h1_ema50_handle,0,1,1,ema)<1) return false;
   double cl[]; ArraySetAsSeries(cl,true);
   if(CopyClose(_Symbol,PERIOD_H1,0,2,cl)<2) return false;
   if(bias==BIAS_LONG)  return cl[1]>ema[0]&&IsHullBull(h1_wma_half_handle,h1_wma_full_handle,h1_hull_sqrt);
   if(bias==BIAS_SHORT) return cl[1]<ema[0]&&IsHullBear(h1_wma_half_handle,h1_wma_full_handle,h1_hull_sqrt);
   return false;
}

//+------------------------------------------------------------------+
//| 15M Entry + Fibonacci                                             |
//+------------------------------------------------------------------+
ENUM_ENTRY_METHOD Check15MEntry(ENUM_TREND_BIAS bias)
{
   double o[],h[],l[],c[];
   ArraySetAsSeries(o,true);ArraySetAsSeries(h,true);ArraySetAsSeries(l,true);ArraySetAsSeries(c,true);
   if(CopyOpen(_Symbol,PERIOD_M15,0,30,o)<30) return ENTRY_NONE;
   if(CopyHigh(_Symbol,PERIOD_M15,0,30,h)<30) return ENTRY_NONE;
   if(CopyLow(_Symbol,PERIOD_M15,0,30,l)<30) return ENTRY_NONE;
   if(CopyClose(_Symbol,PERIOD_M15,0,30,c)<30) return ENTRY_NONE;
   double ema[]; ArraySetAsSeries(ema,true);
   if(CopyBuffer(m15_ema21_handle,0,0,5,ema)<5) return ENTRY_NONE;
   double rsi[]; ArraySetAsSeries(rsi,true);
   if(CopyBuffer(m15_rsi_handle,0,0,3,rsi)<3) return ENTRY_NONE;
   double rv=rsi[1];

   bool rsiAB =(bias==BIAS_LONG)?(rv>=InpRSI_Long_Min&&rv<=InpRSI_Long_Max):(rv>=InpRSI_Short_Min&&rv<=InpRSI_Short_Max);
   bool rsiFib=(bias==BIAS_LONG)?(rv>=InpFibRSI_Long_Min&&rv<=InpFibRSI_Long_Max):(rv>=InpFibRSI_Short_Min&&rv<=InpFibRSI_Short_Max);

   if(rsiAB)
   {
      if(bias==BIAS_LONG)
      {
         if((l[1]<=ema[1]||l[2]<=ema[2])&&(BullEngulf(o,h,l,c,1)||BullPin(o,h,l,c,1))) return ENTRY_EMA_BOUNCE;
         double swl=0,swh=0;int slb=0,shb=0;
         if(FindSwingLow(l,InpSwingLookback,swl,slb)&&FindSwingHighBefore(h,InpSwingLookback,slb,swh,shb)&&c[1]>swh) return ENTRY_STRUCTURE;
      }
      else
      {
         if((h[1]>=ema[1]||h[2]>=ema[2])&&(BearEngulf(o,h,l,c,1)||BearPin(o,h,l,c,1))) return ENTRY_EMA_BOUNCE;
         double swh=0,swl=0;int shb=0,slb=0;
         if(FindSwingHigh(h,InpSwingLookback,swh,shb)&&FindSwingLowBefore(l,InpSwingLookback,shb,swl,slb)&&c[1]<swl) return ENTRY_STRUCTURE;
      }
   }
   if(InpFibEnabled&&rsiFib&&CheckFib(bias,o,h,l,c)) return ENTRY_FIBONACCI;
   return ENTRY_NONE;
}

bool CheckFib(ENUM_TREND_BIAS bias,const double &o[],const double &h[],const double &l[],const double &c[])
{
   double hh[],ll[]; ArraySetAsSeries(hh,true); ArraySetAsSeries(ll,true);
   if(CopyHigh(_Symbol,PERIOD_H1,0,InpFibSwingBarsH1,hh)<InpFibSwingBarsH1) return false;
   if(CopyLow(_Symbol,PERIOD_H1,0,InpFibSwingBarsH1,ll)<InpFibSwingBarsH1) return false;
   double fH=0,fL=0; int shb=0,slb=0;
   if(bias==BIAS_LONG)
   {
      if(!FindSwingHigh(hh,3,fH,shb)||!FindSwingLowBefore(ll,3,shb,fL,slb)) return false;
      double r=fH-fL; if(r<=0) return false;
      double f50=fH-r*InpFibLevel1, f618=fH-r*InpFibLevel2, tol=r*InpFibTolerance;
      if(l[1]<f618-tol||l[1]>f50+tol) return false;
      if(c[1]<f618-tol) return false;
      return BullEngulf(o,h,l,c,1)||BullPin(o,h,l,c,1);
   }
   else
   {
      if(!FindSwingLow(ll,3,fL,slb)||!FindSwingHighBefore(hh,3,slb,fH,shb)) return false;
      double r=fH-fL; if(r<=0) return false;
      double f50=fL+r*InpFibLevel1, f618=fL+r*InpFibLevel2, tol=r*InpFibTolerance;
      if(h[1]<f50-tol||h[1]>f618+tol) return false;
      if(c[1]>f618+tol) return false;
      return BearEngulf(o,h,l,c,1)||BearPin(o,h,l,c,1);
   }
}

double CheckProximity(ENUM_TREND_BIAS bias)
{
   if(InpProximityATR<=0) return 1.0;
   double ema[],atr[]; ArraySetAsSeries(ema,true); ArraySetAsSeries(atr,true);
   if(CopyBuffer(m15_ema21_handle,0,0,1,ema)<1||CopyBuffer(m15_atr_handle,0,0,1,atr)<1) return 0;
   double px=(bias==BIAS_LONG)?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
   return (MathAbs(px-ema[0])<=InpProximityATR*atr[0]) ? ema[0] : 0;
}

//+------------------------------------------------------------------+
//| Execute Trade — 3-tier TP                                         |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_TREND_BIAS bias, ENUM_ENTRY_METHOD method)
{
   long mm=(long)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mm==0&&CountOpenPositions()>0) return;

   double atr15[]; ArraySetAsSeries(atr15,true);
   if(CopyBuffer(m15_atr_handle,0,0,1,atr15)<1) return;
   double atr=atr15[0];

   double spread=SymbolInfoDouble(_Symbol,SYMBOL_ASK)-SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(spread>atr*InpMaxSpreadATR) return;

   double lo[],hi[]; ArraySetAsSeries(lo,true); ArraySetAsSeries(hi,true);
   if(CopyLow(_Symbol,PERIOD_M15,0,20,lo)<20||CopyHigh(_Symbol,PERIOD_M15,0,20,hi)<20) return;

   double entry=0,sl=0,slDist=0;
   if(bias==BIAS_LONG)
   {
      entry=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
      double sw=0;int sb=0;
      sl=(FindSwingLow(lo,InpSwingLookbackSL,sw,sb))? sw-InpSL_ATR_Buffer*atr : entry-InpSL_Min_ATR*atr;
      slDist=entry-sl;
   }
   else
   {
      entry=SymbolInfoDouble(_Symbol,SYMBOL_BID);
      double sw=0;int sb=0;
      sl=(FindSwingHigh(hi,InpSwingLookbackSL,sw,sb))? sw+InpSL_ATR_Buffer*atr : entry+InpSL_Min_ATR*atr;
      slDist=sl-entry;
   }
   if(slDist<=0) return;
   if(slDist>InpSL_Max_ATR*atr) return;
   if(slDist<InpSL_Min_ATR*atr) { slDist=InpSL_Min_ATR*atr; sl=(bias==BIAS_LONG)?entry-slDist:entry+slDist; }
   if(!IsStopLevelValid(entry,sl)) return;

   double lots=CalcLots(slDist);
   if(lots<=0) return;

   int dir=(bias==BIAS_LONG)?1:-1;
   double tp1=entry+dir*slDist*InpTP1_RR;
   double tp2=entry+dir*slDist*InpTP2_RR;
   double tp3=entry+dir*slDist*InpTP3_RR;

   string mc=(method==ENTRY_EMA_BOUNCE)?"A":(method==ENTRY_STRUCTURE)?"B":"C";
   string mf=(method==ENTRY_EMA_BOUNCE)?"EMA Bounce":(method==ENTRY_STRUCTURE)?"Structure Break":"Fibonacci";

   bool ok=(bias==BIAS_LONG)?trade.Buy(lots,_Symbol,entry,sl,0,"MTFTR5_"+mc):trade.Sell(lots,_Symbol,entry,sl,0,"MTFTR5_"+mc);

   if(ok&&trade.ResultRetcode()==TRADE_RETCODE_DONE)
   {
      ulong t=trade.ResultOrder();
      TrackNew(t,entry,sl,tp1,tp2,tp3,lots,dir,mf,slDist);
      dailyTradeCount++; SaveSafetyState();
      Print("MTFTR v5 OPEN: ",(bias==BIAS_LONG?"LONG":"SHORT")," | ",mf,
            " | Entry:",DoubleToString(entry,_Digits),
            " | SL:",DoubleToString(sl,_Digits),
            " | TP1(",InpTP1_RR,"R):",DoubleToString(tp1,_Digits),
            " | TP2(",InpTP2_RR,"R):",DoubleToString(tp2,_Digits),
            " | TP3(",InpTP3_RR,"R):",DoubleToString(tp3,_Digits),
            " | Lots:",DoubleToString(lots,2)," | #",t);
      int idx=FindPos(t); if(idx>=0) LogEvent("OPEN",g_positions[idx],0,"");
   }
   else Print("MTFTR v5 ERROR: ",trade.ResultRetcode()," ",trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| Position Management — 3-tier exit with BE offset                  |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   for(int i=0;i<g_posCount;i++)
   {
      if(!g_positions[i].active) continue;
      ulong tk=g_positions[i].ticket;
      if(!PositionSelectByTicket(tk)) continue;

      int dir=g_positions[i].direction, st=g_positions[i].state;
      double slD=g_positions[i].slDistance;

      // --- Time exit with 10018 cooldown
      if(st==STATE_INITIAL)
      {
         int elapsed=(int)(TimeCurrent()-g_positions[i].entryTime);
         if(elapsed>InpMaxTradeHours*3600)
         {
            if(g_positions[i].closeCooldown&&TimeCurrent()<g_positions[i].cooldownUntil) continue;
            double px=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
            double unPnL=(dir==1)?(px-g_positions[i].entryPrice):(g_positions[i].entryPrice-px);
            if(InpSmartTimeExit&&unPnL>0&&!g_positions[i].timeExtended)
            { g_positions[i].entryTime+=InpMaxTradeHours*3600; g_positions[i].timeExtended=true;
              Print("MTFTR v5: #",tk," in profit (",DoubleToString(unPnL,_Digits),"). Extending."); continue; }
            if(!IsMarketOpen())
            { g_positions[i].closeCooldown=true; g_positions[i].cooldownUntil=TimeCurrent()+900; continue; }
            if(trade.PositionClose(tk))
            { LogEvent("TIME_EXIT",g_positions[i],px,""); g_positions[i].active=false; }
            else { int err=(int)trade.ResultRetcode();
              if(err==10018){g_positions[i].closeCooldown=true;g_positions[i].cooldownUntil=TimeCurrent()+900;} }
            continue;
         }
      }

      // --- TP1: Close 20%, SL to BE (with broker-safe offset)
      if(st==STATE_INITIAL)
      {
         bool hit=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID)>=g_positions[i].tp1Price
                          :SymbolInfoDouble(_Symbol,SYMBOL_ASK)<=g_positions[i].tp1Price;
         if(hit)
         {
            double cv=NormLots(g_positions[i].originalVolume*InpTP1_ClosePct/100.0);
            double pv=PositionGetDouble(POSITION_VOLUME);
            if(cv>=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN)&&cv<pv)
            {
               if(trade.PositionClosePartial(tk,cv))
               {
                  // [v5] BE offset fix: when InpTP1_SL_MoveToRR=0, newSL=entry exactly.
                  // If market sits right at entry the modify is rejected by stop-level.
                  // Apply max(SL_MoveToRR-distance, InpBE_OffsetATR × M15 ATR).
                  double atrBuf[]; ArraySetAsSeries(atrBuf,true);
                  double beOffset = 0;
                  if(CopyBuffer(m15_atr_handle,0,0,1,atrBuf) >= 1)
                     beOffset = atrBuf[0] * InpBE_OffsetATR;
                  double moveDist = MathMax(slD * InpTP1_SL_MoveToRR, beOffset);
                  double newSL = g_positions[i].entryPrice + dir * moveDist;
                  double mkt=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
                  if(IsStopLevelValid(mkt,newSL) && trade.PositionModify(tk,newSL,0))
                     g_positions[i].lastKnownSL=newSL;
                  g_positions[i].state=STATE_TP1_HIT;
                  consecLosses=0; SaveSafetyState();
                  Print("MTFTR v5 TP1(",InpTP1_RR,"R): -",cv," lots. SL→BE+",DoubleToString(moveDist,_Digits),
                        " (",DoubleToString(newSL,_Digits),") #",tk);
                  LogEvent("TP1_HIT",g_positions[i],g_positions[i].tp1Price,"");
               }
            }
            else { trade.PositionClose(tk); LogEvent("TP1_HIT",g_positions[i],g_positions[i].tp1Price,"Full close");
                   g_positions[i].active=false; consecLosses=0; SaveSafetyState(); }
         }
      }

      // --- TP2: Close 30%, SL to +2.0R
      else if(st==STATE_TP1_HIT)
      {
         bool hit=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID)>=g_positions[i].tp2Price
                          :SymbolInfoDouble(_Symbol,SYMBOL_ASK)<=g_positions[i].tp2Price;
         if(hit)
         {
            double pv=PositionGetDouble(POSITION_VOLUME);
            double cv=NormLots(g_positions[i].originalVolume*InpTP2_ClosePct/100.0);
            double tv=NormLots(pv-cv);
            if(tv>=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN)&&cv>0)
            {
               if(trade.PositionClosePartial(tk,cv))
               {
                  double newSL=g_positions[i].entryPrice+dir*slD*InpTP2_SL_MoveToRR;
                  double mkt=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
                  if(IsStopLevelValid(mkt,newSL) && trade.PositionModify(tk,newSL,0))
                     g_positions[i].lastKnownSL=newSL;
                  g_positions[i].state=STATE_TP2_HIT;
                  Print("MTFTR v5 TP2(",InpTP2_RR,"R): -",cv," lots. SL→+",InpTP2_SL_MoveToRR,"R. Trail ",tv," #",tk);
                  LogEvent("TP2_HIT",g_positions[i],g_positions[i].tp2Price,"");
               }
            }
            else { trade.PositionClose(tk); LogEvent("TP2_HIT",g_positions[i],g_positions[i].tp2Price,"Full close");
                   g_positions[i].active=false; }
         }
      }

      // --- TP3 (6R full close) OR H4 Hull trail
      else if(st==STATE_TP2_HIT)
      {
         bool tp3Hit=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID)>=g_positions[i].tp3Price
                             :SymbolInfoDouble(_Symbol,SYMBOL_ASK)<=g_positions[i].tp3Price;
         if(tp3Hit)
         {
            double px=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
            trade.PositionClose(tk);
            g_positions[i].state=STATE_TP3_TRAIL;
            LogEvent("TP3_HIT",g_positions[i],px,"6R target hit");
            g_positions[i].active=false;
            Print("MTFTR v5 TP3(",InpTP3_RR,"R): FULL CLOSE at ",DoubleToString(px,_Digits)," #",tk);
            continue;
         }

         // H4 Hull flip trail
         datetime h4t=iTime(_Symbol,PERIOD_H4,0);
         if(h4t!=g_positions[i].lastH4Check)
         {
            g_positions[i].lastH4Check=h4t;
            bool flip=(dir==1&&IsHullBear(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt))||
                      (dir==-1&&IsHullBull(h4_wma_half_handle,h4_wma_full_handle,h4_hull_sqrt));
            if(flip)
            {
               double px=(dir==1)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
               trade.PositionClose(tk);
               LogEvent("TRAIL_EXIT",g_positions[i],px,"H4 Hull flipped");
               g_positions[i].active=false;
               Print("MTFTR v5 H4 Trail exit #",tk," at ",DoubleToString(px,_Digits)); continue;
            }

            // Tighten SL using H1 EMA50
            double ema50[]; ArraySetAsSeries(ema50,true);
            double atr15[]; ArraySetAsSeries(atr15,true);
            if(CopyBuffer(h1_ema50_handle,0,0,1,ema50)>=1&&CopyBuffer(m15_atr_handle,0,0,1,atr15)>=1)
            {
               double curSL=PositionGetDouble(POSITION_SL), nSL=0;
               if(dir==1) { nSL=ema50[0]-atr15[0]*0.5;
                 if(nSL>curSL&&nSL<SymbolInfoDouble(_Symbol,SYMBOL_BID))
                            if(IsStopLevelValid(SymbolInfoDouble(_Symbol,SYMBOL_BID),nSL) && trade.PositionModify(tk,nSL,0)) g_positions[i].lastKnownSL=nSL; }
               else { nSL=ema50[0]+atr15[0]*0.5;
                 if(nSL<curSL&&nSL>SymbolInfoDouble(_Symbol,SYMBOL_ASK))
                            if(IsStopLevelValid(SymbolInfoDouble(_Symbol,SYMBOL_ASK),nSL) && trade.PositionModify(tk,nSL,0)) g_positions[i].lastKnownSL=nSL; }
            }
         }
      }
   }
}

bool IsMarketOpen()
{
   return (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE)==SYMBOL_TRADE_MODE_FULL;
}

//+------------------------------------------------------------------+
//| Hull MA                                                           |
//+------------------------------------------------------------------+
double HullVal(int wH,int wF,int sq,int sh)
{
   double a[],b[]; ArraySetAsSeries(a,true); ArraySetAsSeries(b,true);
   if(CopyBuffer(wH,0,sh,sq,a)<sq||CopyBuffer(wF,0,sh,sq,b)<sq) return 0;
   double ws=0,vs=0;
   for(int i=0;i<sq;i++){double w=(double)(sq-i);vs+=(2*a[i]-b[i])*w;ws+=w;}
   return ws==0?0:vs/ws;
}
bool IsHullBull(int h,int f,int s){double a=HullVal(h,f,s,0),b=HullVal(h,f,s,1);return a!=0&&b!=0&&a>b;}
bool IsHullBear(int h,int f,int s){double a=HullVal(h,f,s,0),b=HullVal(h,f,s,1);return a!=0&&b!=0&&a<b;}
int HullConsec(int wH,int wF,int sq,bool bull,int mx)
{
   int c=0;
   for(int b=0;b<mx;b++)
   { double hC=HullVal(wH,wF,sq,b),hP=HullVal(wH,wF,sq,b+1);
     if(hC==0||hP==0)break; bool r=hC>hP;
     if(b==0){if(r!=bull)return 0;c=1;}else{if(r==bull)c++;else break;}}
   return c;
}

//+------------------------------------------------------------------+
//| Candle Patterns                                                  |
//+------------------------------------------------------------------+
bool BullEngulf(const double &o[],const double &h[],const double &l[],const double &c[],int i)
{return c[i+1]<o[i+1]&&c[i]>o[i]&&o[i]<=c[i+1]&&c[i]>=o[i+1];}
bool BearEngulf(const double &o[],const double &h[],const double &l[],const double &c[],int i)
{return c[i+1]>o[i+1]&&c[i]<o[i]&&o[i]>=c[i+1]&&c[i]<=o[i+1];}
bool BullPin(const double &o[],const double &h[],const double &l[],const double &c[],int i)
{double bd=MathAbs(c[i]-o[i]),rg=h[i]-l[i];if(rg<=0)return false;
 return(MathMin(o[i],c[i])-l[i]>=2*bd&&h[i]-MathMax(o[i],c[i])<=bd&&(o[i]+c[i])/2>l[i]+rg*0.6);}
bool BearPin(const double &o[],const double &h[],const double &l[],const double &c[],int i)
{double bd=MathAbs(c[i]-o[i]),rg=h[i]-l[i];if(rg<=0)return false;
 return(h[i]-MathMax(o[i],c[i])>=2*bd&&MathMin(o[i],c[i])-l[i]<=bd&&(o[i]+c[i])/2<l[i]+rg*0.4);}

//+------------------------------------------------------------------+
//| Swing Detection                                                  |
//+------------------------------------------------------------------+
bool FindSwingLow(const double &d[],int lb,double &p,int &b)
{int sz=ArraySize(d);if(sz<lb*2+1)return false;
 for(int i=lb;i<MathMin(25,sz-lb-1);i++){bool ok=true;
   for(int j=1;j<=lb;j++){if(i-j<0||i+j>=sz||d[i]>d[i-j]||d[i]>d[i+j]){ok=false;break;}}
   if(ok){p=d[i];b=i;return true;}}return false;}
bool FindSwingHigh(const double &d[],int lb,double &p,int &b)
{int sz=ArraySize(d);if(sz<lb*2+1)return false;
 for(int i=lb;i<MathMin(25,sz-lb-1);i++){bool ok=true;
   for(int j=1;j<=lb;j++){if(i-j<0||i+j>=sz||d[i]<d[i-j]||d[i]<d[i+j]){ok=false;break;}}
   if(ok){p=d[i];b=i;return true;}}return false;}
bool FindSwingHighBefore(const double &d[],int lb,int af,double &p,int &b)
{int sz=ArraySize(d);if(sz<lb*2+1)return false;
 for(int i=MathMax(af+1,lb);i<MathMin(28,sz-lb-1);i++){bool ok=true;
   for(int j=1;j<=lb;j++){if(i-j<0||i+j>=sz||d[i]<d[i-j]||d[i]<d[i+j]){ok=false;break;}}
   if(ok){p=d[i];b=i;return true;}}return false;}
bool FindSwingLowBefore(const double &d[],int lb,int af,double &p,int &b)
{int sz=ArraySize(d);if(sz<lb*2+1)return false;
 for(int i=MathMax(af+1,lb);i<MathMin(28,sz-lb-1);i++){bool ok=true;
   for(int j=1;j<=lb;j++){if(i-j<0||i+j>=sz||d[i]>d[i-j]||d[i]>d[i+j]){ok=false;break;}}
   if(ok){p=d[i];b=i;return true;}}return false;}

//+------------------------------------------------------------------+
//| Lot Size                                                         |
//+------------------------------------------------------------------+
double CalcLots(double slD)
{
   if(slD<=0) return 0;
   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   double risk=bal*InpRiskPercent/100.0;
   double tp=0,ep=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   if(OrderCalcProfit(ORDER_TYPE_BUY,_Symbol,1.0,ep,ep-slD,tp)){double lpl=MathAbs(tp);if(lpl>0)return NormLots(risk/lpl);}
   double tv=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE),ts=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   if(tv<=0||ts<=0) return SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double rpl=(slD/ts)*tv; return rpl<=0?SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN):NormLots(risk/rpl);
}
double NormLots(double lots)
{
   double mn=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN),mx=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX),st=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(st<=0)return 0; lots=MathFloor(lots/st)*st;
   return NormalizeDouble(MathMin(MathMax(lots,mn),mx),(int)MathCeil(-MathLog10(st)));
}

bool IsStopLevelValid(double price,double sl)
{
   long stopLevel=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double minDist=stopLevel*SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   return MathAbs(price-sl)>=minDist;
}

//+------------------------------------------------------------------+
//| Session & Utility                                                 |
//+------------------------------------------------------------------+
bool IsWithinSession()
{
   MqlDateTime dt; TimeCurrent(dt);
   int gmt=dt.hour-InpGMTOffset; if(gmt<0)gmt+=24; if(gmt>=24)gmt-=24;
   if(InpFridayFilter&&dt.day_of_week==5&&gmt>=15) return false;
   if(dt.day_of_week==0||dt.day_of_week==6) return false;
   return (gmt>=InpSessionStart&&gmt<InpSessionEnd);
}
bool IsNewBar(ENUM_TIMEFRAMES tf)
{
   datetime t=iTime(_Symbol,tf,0);
   if(tf==PERIOD_M15){if(t!=lastBar_M15){lastBar_M15=t;return true;}}
   else if(tf==PERIOD_H4){if(t!=lastBar_H4){lastBar_H4=t;return true;}}
   return false;
}
int CountOpenPositions()
{
   int c=0;for(int i=PositionsTotal()-1;i>=0;i--)
     if(posInfo.SelectByIndex(i)&&posInfo.Magic()==InpMagicNumber&&posInfo.Symbol()==_Symbol)c++;
   return c;
}

//+------------------------------------------------------------------+
//| Position Tracking                                                 |
//+------------------------------------------------------------------+
void TrackNew(ulong tk,double entry,double sl,double tp1,double tp2,double tp3,
              double vol,int dir,string method,double slD)
{
   int idx=-1;
   for(int i=0;i<g_posCount;i++) if(!g_positions[i].active){idx=i;break;}
   if(idx==-1){idx=g_posCount++;if(g_posCount>ArraySize(g_positions))ArrayResize(g_positions,g_posCount+5);}

   g_positions[idx].ticket=tk; g_positions[idx].entryPrice=entry;
   g_positions[idx].originalSL=sl; g_positions[idx].tp1Price=tp1;
   g_positions[idx].tp2Price=tp2; g_positions[idx].tp3Price=tp3;
   g_positions[idx].originalVolume=vol; g_positions[idx].slDistance=slD;
   g_positions[idx].state=STATE_INITIAL; g_positions[idx].direction=dir;
   g_positions[idx].entryTime=TimeCurrent(); g_positions[idx].active=true;
   g_positions[idx].lastKnownSL=sl; g_positions[idx].entryBalance=AccountInfoDouble(ACCOUNT_BALANCE);
   g_positions[idx].lastH1Check=0; g_positions[idx].lastH4Check=0;
   g_positions[idx].timeExtended=false; g_positions[idx].closeCooldown=false; g_positions[idx].cooldownUntil=0;
   g_positions[idx].entryMethod=method; g_positions[idx].session=GetSession();
   g_positions[idx].d1Bias=BiasStr(GetDailyTrendBias()); g_positions[idx].h4Bias=BiasStr(Get4HTrendBias());

   string p="MTFTR5_"+IntegerToString(tk);
   GlobalVariableSet(p+"_ST",0); GlobalVariableSet(p+"_EN",entry);
   GlobalVariableSet(p+"_SL",sl); GlobalVariableSet(p+"_T1",tp1);
   GlobalVariableSet(p+"_T2",tp2); GlobalVariableSet(p+"_T3",tp3);
   GlobalVariableSet(p+"_VL",vol); GlobalVariableSet(p+"_DR",(double)dir);
   GlobalVariableSet(p+"_TM",(double)TimeCurrent()); GlobalVariableSet(p+"_SD",slD);
}

void LoadExistingPositions()
{
   g_posCount=0;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      if(!posInfo.SelectByIndex(i)||posInfo.Magic()!=InpMagicNumber||posInfo.Symbol()!=_Symbol) continue;
      ulong tk=posInfo.Ticket(); string p="MTFTR5_"+IntegerToString(tk);
      if(g_posCount>=ArraySize(g_positions)) ArrayResize(g_positions,g_posCount+5);
      g_positions[g_posCount].ticket=tk; g_positions[g_posCount].active=true;
      g_positions[g_posCount].direction=(posInfo.PositionType()==POSITION_TYPE_BUY)?1:-1;
      g_positions[g_posCount].lastH1Check=0; g_positions[g_posCount].lastH4Check=0;
      g_positions[g_posCount].timeExtended=false; g_positions[g_posCount].closeCooldown=false;
      if(GlobalVariableCheck(p+"_ST"))
      {
         g_positions[g_posCount].state=(int)GlobalVariableGet(p+"_ST");
         g_positions[g_posCount].entryPrice=GlobalVariableGet(p+"_EN");
         g_positions[g_posCount].originalSL=GlobalVariableGet(p+"_SL");
         g_positions[g_posCount].tp1Price=GlobalVariableGet(p+"_T1");
         g_positions[g_posCount].tp2Price=GlobalVariableGet(p+"_T2");
         g_positions[g_posCount].tp3Price=GlobalVariableCheck(p+"_T3")?GlobalVariableGet(p+"_T3"):0;
         g_positions[g_posCount].originalVolume=GlobalVariableGet(p+"_VL");
         g_positions[g_posCount].entryTime=(datetime)(int)GlobalVariableGet(p+"_TM");
         g_positions[g_posCount].slDistance=GlobalVariableCheck(p+"_SD")?GlobalVariableGet(p+"_SD"):MathAbs(posInfo.PriceOpen()-posInfo.StopLoss());
      }
      else
      {
         double sd=MathAbs(posInfo.PriceOpen()-posInfo.StopLoss());
         g_positions[g_posCount].entryPrice=posInfo.PriceOpen(); g_positions[g_posCount].originalSL=posInfo.StopLoss();
         g_positions[g_posCount].originalVolume=posInfo.Volume(); g_positions[g_posCount].entryTime=posInfo.Time();
         g_positions[g_posCount].state=STATE_INITIAL; g_positions[g_posCount].slDistance=sd;
         g_positions[g_posCount].tp1Price=posInfo.PriceOpen()+g_positions[g_posCount].direction*sd*InpTP1_RR;
         g_positions[g_posCount].tp2Price=posInfo.PriceOpen()+g_positions[g_posCount].direction*sd*InpTP2_RR;
         g_positions[g_posCount].tp3Price=posInfo.PriceOpen()+g_positions[g_posCount].direction*sd*InpTP3_RR;
      }
      g_positions[g_posCount].lastKnownSL=posInfo.StopLoss();
      g_posCount++;
   }
}

void CleanupClosedPositions()
{
   for(int i=0;i<g_posCount;i++)
   {
      if(!g_positions[i].active||PositionSelectByTicket(g_positions[i].ticket)) continue;
      ulong tk=g_positions[i].ticket;
      if(g_positions[i].state==STATE_INITIAL)
      { consecLosses++; LogEvent("SL_HIT",g_positions[i],g_positions[i].lastKnownSL,"");
        if(consecLosses>=InpMaxConsecLosses){dailyPauseActive=true;Print("MTFTR v5: Paused after ",consecLosses," losses");}}
      else
      { consecLosses=0;
        if(g_positions[i].state==STATE_TP1_HIT) LogEvent("BE_EXIT",g_positions[i],g_positions[i].lastKnownSL,"BE SL triggered");
        else if(g_positions[i].state>=STATE_TP2_HIT) LogEvent("TRAIL_EXIT",g_positions[i],g_positions[i].lastKnownSL,"Trail SL triggered");}
      SaveSafetyState();
      string p="MTFTR5_"+IntegerToString(tk);
      GlobalVariableDel(p+"_ST");GlobalVariableDel(p+"_EN");GlobalVariableDel(p+"_SL");
      GlobalVariableDel(p+"_T1");GlobalVariableDel(p+"_T2");GlobalVariableDel(p+"_T3");
      GlobalVariableDel(p+"_VL");GlobalVariableDel(p+"_DR");GlobalVariableDel(p+"_TM");GlobalVariableDel(p+"_SD");
      g_positions[i].active=false;
   }
}

void ResetDailyStats()
{
   MqlDateTime dt; TimeCurrent(dt);
   int today=dt.year*10000+dt.mon*100+dt.day;
   if(today!=lastTradeDay)
   { lastTradeDay=today; dailyTradeCount=0; dailyPauseActive=false;
     dayStartBalance=AccountInfoDouble(ACCOUNT_BALANCE); consecLosses=0;
     if(dt.day_of_week==1&&lastTradeWeekDay!=today){weekStartBalance=AccountInfoDouble(ACCOUNT_BALANCE);lastTradeWeekDay=today;}
     SaveSafetyState(); }
}
bool CheckDailyLimits()
{if(dayStartBalance<=0)return true;double dd=(dayStartBalance-AccountInfoDouble(ACCOUNT_BALANCE))/dayStartBalance*100;
 if(dd>=InpMaxDailyDD){static datetime lp=0;if(TimeCurrent()-lp>300){Print("MTFTR v5: Daily DD ",DoubleToString(dd,1),"%");lp=TimeCurrent();}return false;}return true;}
bool CheckWeeklyLimits()
{if(weekStartBalance<=0)return true;double dd=(weekStartBalance-AccountInfoDouble(ACCOUNT_BALANCE))/weekStartBalance*100;
 if(dd>=InpMaxWeeklyDD){static datetime lp=0;if(TimeCurrent()-lp>300){Print("MTFTR v5: Weekly DD ",DoubleToString(dd,1),"%");lp=TimeCurrent();}return false;}return true;}

//+------------------------------------------------------------------+
//| Logging                                                           |
//+------------------------------------------------------------------+
string GetSession(){MqlDateTime dt;TimeCurrent(dt);int g=dt.hour-InpGMTOffset;if(g<0)g+=24;if(g>=24)g-=24;
  if(g>=InpSessionStart&&g<12)return"London";if(g>=12&&g<InpSessionEnd)return"NY Overlap";return"Off-Hours";}
string BiasStr(ENUM_TREND_BIAS b){return(b==BIAS_LONG)?"LONG":(b==BIAS_SHORT)?"SHORT":"NEUTRAL";}
int FindPos(ulong tk){for(int i=0;i<g_posCount;i++)if(g_positions[i].ticket==tk&&g_positions[i].active)return i;return-1;}
string OutcomeLabel(string e,const TrackedPosition &p)
{if(e=="SL_HIT")return"LOSS";if(e=="BE_EXIT")return"BREAKEVEN";if(e=="TP1_HIT"||e=="TP2_HIT")return"WIN_PARTIAL";
 if(e=="TP3_HIT")return"WIN_FULL";if(e=="TRAIL_EXIT")return(p.state>=STATE_TP1_HIT)?"WIN":"LOSS";if(e=="TIME_EXIT")return"TIME_EXIT";return"OPEN";}
double CalcPnL(const TrackedPosition &p,double ex){if(ex<=0)return 0;return NormalizeDouble((p.direction==1?(ex-p.entryPrice):(p.entryPrice-ex))*p.originalVolume*100,2);}

void LogEvent(string evt,const TrackedPosition &pos,double ex,string note)
{
   if(!InpEnableCSV) return;
   int h=FileOpen(InpCSVFileName,FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON,',');
   if(h==INVALID_HANDLE) return;
   if(FileSize(h)==0) FileWrite(h,"Timestamp","Event","Ticket","Direction","Method","Session",
     "Entry","SL","TP1","TP2","TP3","Lots","Risk%","SL_Dist","Exit_Price","PnL","RR","Outcome",
     "D1_Bias","H4_Bias","State","Balance","Equity","TotalWithdrawn","Note");
   FileSeek(h,0,SEEK_END);
   MqlDateTime dt; TimeCurrent(dt);
   double pnl=CalcPnL(pos,ex), slD=MathAbs(pos.entryPrice-pos.originalSL);
   double rr=(slD>0&&ex>0)?NormalizeDouble(pnl/(slD*pos.originalVolume*100),2):0;
   FileWrite(h,
     StringFormat("%04d.%02d.%02d %02d:%02d:%02d",dt.year,dt.mon,dt.day,dt.hour,dt.min,dt.sec),
     evt,(string)pos.ticket,(pos.direction==1)?"LONG":"SHORT",pos.entryMethod,pos.session,
     DoubleToString(pos.entryPrice,_Digits),DoubleToString(pos.originalSL,_Digits),
     DoubleToString(pos.tp1Price,_Digits),DoubleToString(pos.tp2Price,_Digits),
     DoubleToString(pos.tp3Price,_Digits),DoubleToString(pos.originalVolume,2),
     DoubleToString(InpRiskPercent,1),DoubleToString(slD,_Digits),
     DoubleToString(ex,_Digits),DoubleToString(pnl,2),DoubleToString(rr,2),
     OutcomeLabel(evt,pos),pos.d1Bias,pos.h4Bias,(string)pos.state,
     DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2),
     DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2),
     DoubleToString(totalWithdrawn,2),note);
   FileClose(h);
}

//+------------------------------------------------------------------+
//| Info Panel                                                        |
//+------------------------------------------------------------------+
void UpdateInfoPanel()
{
   if(!InpShowInfoPanel) return;
   int y=30,s=17;
   string bg=g_panelName+"_BG";
   if(ObjectFind(0,bg)<0)
   {ObjectCreate(0,bg,OBJ_RECTANGLE_LABEL,0,0,0);
    ObjectSetInteger(0,bg,OBJPROP_CORNER,CORNER_RIGHT_UPPER);
    ObjectSetInteger(0,bg,OBJPROP_XDISTANCE,10);ObjectSetInteger(0,bg,OBJPROP_YDISTANCE,20);
    ObjectSetInteger(0,bg,OBJPROP_XSIZE,180);ObjectSetInteger(0,bg,OBJPROP_YSIZE,280);
    ObjectSetInteger(0,bg,OBJPROP_BGCOLOR,C'30,30,30');ObjectSetInteger(0,bg,OBJPROP_BORDER_TYPE,BORDER_FLAT);}

   ENUM_TREND_BIAS db=GetDailyTrendBias(),b4=Get4HTrendBias();
   double adx[]; ArraySetAsSeries(adx,true); double adxVal=0;
   if(CopyBuffer(h1_adx_handle,0,0,1,adx)>=1) adxVal=adx[0];

   PL(g_panelName+"_T","MTFTR v5.0 [1% 2/4/6R]",165,y,clrWhite,9,true); y+=s+4;
   PL(g_panelName+"_S","—————————",165,y,clrDimGray,8,false); y+=s;

   double bal=AccountInfoDouble(ACCOUNT_BALANCE);
   PL(g_panelName+"_B","Bal: $"+DoubleToString(bal,2),165,y,clrWhite,9,false); y+=s;
   double dpl=AccountInfoDouble(ACCOUNT_EQUITY)-dayStartBalance;
   PL(g_panelName+"_P","P&L: "+((dpl>=0)?"+":"")+DoubleToString(dpl,2),165,y,(dpl>=0)?clrLime:clrRed,9,false); y+=s;
   PL(g_panelName+"_W","Withdrawn: $"+DoubleToString(totalWithdrawn,2),165,y,clrYellow,9,false); y+=s;
   PL(g_panelName+"_S2","—————————",165,y,clrDimGray,8,false); y+=s;

   PL(g_panelName+"_TC","Trades: "+IntegerToString(dailyTradeCount)+"/"+IntegerToString(InpMaxDailyTrades),165,y,clrWhite,9,false); y+=s;
   PL(g_panelName+"_SS","Session: "+(IsWithinSession()?"ACTIVE":"CLOSED"),165,y,IsWithinSession()?clrLime:clrGray,9,false); y+=s;
   PL(g_panelName+"_AX","ADX: "+DoubleToString(adxVal,1)+(adxVal>=InpADX_Min?" OK":" FLAT"),165,y,(adxVal>=InpADX_Min)?clrLime:clrOrange,9,false); y+=s;

   string ds=(db==BIAS_LONG)?"BULL":(db==BIAS_SHORT)?"BEAR":"—";
   string bs=(b4==BIAS_LONG)?"BULL":(b4==BIAS_SHORT)?"BEAR":"—";
   bool aligned = (db == b4 && b4 != BIAS_NEUTRAL);
   PL(g_panelName+"_D1","D1:"+ds+" | 4H:"+bs+(aligned?" ✓":" ✗"),165,y,
      aligned?((b4==BIAS_LONG)?InpBullColor:InpBearColor):InpNeutralColor,9,true); y+=s;

   bool h1OK=(b4!=BIAS_NEUTRAL)?Check1HConfirmation(b4):false;
   PL(g_panelName+"_1H","1H: "+(b4==BIAS_NEUTRAL?"N/A":(h1OK?"CONFIRMED":"WAIT")),165,y,
      h1OK?clrLime:clrOrange,9,false); y+=s;

   int tr=0;for(int i=0;i<g_posCount;i++){if(!g_positions[i].active)continue;if(g_positions[i].state>=STATE_TP2_HIT)tr++;}
   PL(g_panelName+"_TL","Trailing: "+IntegerToString(tr)+" | Open: "+IntegerToString(CountOpenPositions())+"/"+IntegerToString(InpMaxOpenTrades),165,y,clrWhite,9,false);
}

void PL(string n,string t,int x,int y,color c,int fs,bool b)
{if(ObjectFind(0,n)<0){ObjectCreate(0,n,OBJ_LABEL,0,0,0);ObjectSetInteger(0,n,OBJPROP_CORNER,CORNER_RIGHT_UPPER);ObjectSetInteger(0,n,OBJPROP_ANCHOR,ANCHOR_RIGHT);}
 ObjectSetInteger(0,n,OBJPROP_XDISTANCE,x);ObjectSetInteger(0,n,OBJPROP_YDISTANCE,y);
 ObjectSetString(0,n,OBJPROP_TEXT,t);ObjectSetInteger(0,n,OBJPROP_COLOR,c);
 ObjectSetInteger(0,n,OBJPROP_FONTSIZE,fs);ObjectSetString(0,n,OBJPROP_FONT,b?"Arial Bold":"Arial");}
//+------------------------------------------------------------------+
