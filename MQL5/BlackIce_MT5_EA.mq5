//+------------------------------------------------------------------+
//|                                         BlackIce_REST_EA_V2.mq5  |
//|                          Optimized for 60% Institutional Models  |
//|                                   24 SMC Features, Weighted Ensemble |
//+------------------------------------------------------------------+
#property copyright "Black Ice AI"
#property version   "2.00"
#property description "Institutional Models - 60% Accuracy"
#property strict

// Include modular core functions
#include "core_functions.mqh"


enum    TrType          {  Highlow=1,        // Previous low or high 
                           FixedPips=2,      // Defined no of pips 
                           PctofPrice = 3    // Trail stoploss as % of distance from the TP 
                        }; 
                           
enum    SLType          {  Yes=0,            // Use Trailing stoploss
                           No=1              // Use fixed stoploss 
                        }; 
                        
enum    IntervalTime     {
                           newBar = 0,       // Once every new bar
                           xSeconds = 1      // Every X Seconds   
                        }; 
                        
input group "/--- Input Parameters ---/"

   input string            RestServerURL = "http://127.0.0.1:5000/predict";   // REST API URL
   input long              inpMagic = 123456;                                 // EA magic number
   input ENUM_TIMEFRAMES   AnalysisTimeframe = PERIOD_H4;                     // Fixed timeframe for analysis
   input IntervalTime      UpdateIntervalSeconds = 0;                         // Update interval
   input int               updateSeconds = 10;                                // Update interval in Seconds if "Every X Seconds" selected
   input int               BarsToSend = 200;                                  // Bars to send (increased for SMC)
   input double            MinConfidence = 0.55;                              // Minimum confidence (lowered from 0.70)
   input double            BaseLotSize = 0.01;                                // Base position size
   input bool              UseConfidenceScaling = true;                       // Scale lots by confidence
   input int               StopLossPips = 50;                                 // Stop loss in pips
   input double            MinRiskReward = 2.0;                               // Minimum R:R ratio
   input bool              EnableTrading = false;                             // Enable actual trading
   input bool              ShowDebugInfo = true;                              // Show debug logs
   input bool              PredictOnStart = true;                             // Make prediction on EA start

input group "/--- Risk & Order Management Inputs ---/"

   input bool     UseSystemRisk = true;                              // Let EA compute lot size based on confidence/regime
   input double   UserRiskPercent = 0.5;                             // If user-defined risk (percent of account) per trade
   input double   MaxRiskPercent = 2.0;                              // Maximum percent allowed
   input bool     EnableCompensatory = true;                         // Enable recovery multiplier after losing trades
   input int      RecoveryDepth = 3;                                 // Max consecutive recovery steps
   input double   RecoveryMultiplier = 1.5;                          // Multiplier per recovery step
   input double   ConfidenceHigh = 0.80;                             // High confidence threshold
   input double   ConfidenceMed = 0.70;                              // Medium confidence threshold
   input double   ConfidenceLow = 0.60;                              // Low confidence threshold

input group "/--- Core functions news & trailing configuration ---/" 

   input bool     NewsFilterOn = false;            // Enable news filtering
   input int      StartTradingMin = 0;             // Minimum minutes to wait after news
   input int      separator = 0;                   // 0=comma,1=semicolon for news keys
   input string   keyNews = "";                    // Comma/semicolon separated list of news keywords
   input string   NewsCurrencied = "";             // Comma-separated currencies to monitor (e.g., "USD,EUR")
   input int      DaysNewsLookup = 1;              // How many days ahead to look for calendar events
   input int      StopBeforeMin = 60;              // Minutes before news to avoid trading
   ushort sep_code;

input group "/--- Trailing params ---/"  
   input SLType   SLT = 0;                         // Use Trailing stoploss (TSL)? (No = Fixed Stoploss)
   input TrType   trailType = 1;                   // If using TSL, what type of TSL? 
   input int      BarsN = 20;                      // Number of bars to scan for highs and lows
   input int      HighLowBuffer = 3;               // Buffer from prev low or high to trail (If selected) 
   input int      trailFixedpips = 10;             // Number of pips to trail SL (If option is selected)
   input double   TslPercent = 1.0;                // Percentage of ptice to TSL
   input double   TslPercentTP = 50.0;             // start TSL at x% from TP

//--- Global Variables
datetime lastBarTime = 0;
datetime lastUpdateTime = 0;
int requestCount = 0;
int successCount = 0;
bool firstRun = true;
int consecutiveLosses = 0;
double lastTradeProfit = 0.0;

//--- SMC Context (extracted from response)
string smcOrderBlocks = "";
string smcFairValueGaps = "";
string smcStructure = "";
string smcRegime = "";

//--- Trade logging
string logFileName = "BlackIce_Trades.csv";


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{

   ChartSetInteger(0, CHART_SHOW_GRID, false); 
   
   Print("========================================");
   Print("BLACK ICE EA V2 - INSTITUTIONAL MODELS");
   Print("========================================");
   Print("Server: ", RestServerURL);
   Print("Analysis Timeframe: ", EnumToString(AnalysisTimeframe));
   Print("Chart Timeframe: ", EnumToString(PERIOD_CURRENT));
   
   if(UpdateIntervalSeconds == 0)
      Print("Update Mode: New Bar Only (every ", PeriodSeconds(AnalysisTimeframe), " seconds)");
   else
      Print("Update Mode: Every ", UpdateIntervalSeconds, " seconds");
   
   Print("Min Confidence: ", MinConfidence);
   Print("Bars to Send: ", BarsToSend);
   Print("Trading: ", EnableTrading ? "ENABLED" : "DEMO MODE");
   Print("========================================");
   
   // Check WebRequest permission
   if(!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED))
   {
      Alert("DLL imports must be allowed!");
      return(INIT_FAILED);
   }
   
   // Initialize trade log file
   InitializeTradeLog();
   
   lastBarTime = iTime(_Symbol, AnalysisTimeframe, 0);
   lastUpdateTime = TimeCurrent();
   
   // Make initial prediction if enabled
   if(PredictOnStart)
   {
      Print("Making initial prediction...");
      MakePrediction();
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("========================================");
   Print("EA Stopped. Stats:");
   Print("  Requests: ", requestCount);
   Print("  Success: ", successCount);
   Print("  Success Rate: ", (requestCount > 0 ? (double)successCount/requestCount*100 : 0), "%");
   Print("========================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Make prediction on first tick
   if(firstRun)
   {
      firstRun = false;
      if(!PredictOnStart)  // Only if not already done in OnInit
      {
         Print("Making first prediction...");
         MakePrediction();
      }
      return;
   }
   
   // Two update modes:
   // 1. New bar only (UpdateIntervalSeconds = 0)
   // 2. Timer-based (UpdateIntervalSeconds > 0)
   
   bool shouldUpdate = false;
   datetime currentBarTime = iTime(_Symbol, AnalysisTimeframe, 0);
   datetime currentTime = TimeCurrent();
   switch(UpdateIntervalSeconds){
      case 0:
         // Mode 1: Update on new bar only
         if(currentBarTime != lastBarTime){
            lastBarTime = currentBarTime;
            shouldUpdate = true;
         }
         break;
      case 1:
         //Mode 2: Update every X seconds
         if(currentTime - lastUpdateTime >= updateSeconds){
            lastUpdateTime = currentTime;
            shouldUpdate = true;
            
            // Also update lastBarTime to track bar changes
            lastBarTime = iTime(_Symbol, AnalysisTimeframe, 0);
         }
         break;
   }
   
   if(shouldUpdate)
   {
      MakePrediction();
   }
}

//+------------------------------------------------------------------+
//| Make prediction via REST API                                     |
//+------------------------------------------------------------------+
void MakePrediction()
{
   requestCount++;
   
   // Prepare OHLCV data
   string jsonData = PrepareOHLCVData();
   if(jsonData == "")
   {
      Print("ERROR: Failed to prepare OHLCV data");
      return;
   }
   
   // Send request
   char post[];
   char result[];
   string headers = "Content-Type: application/json\r\n";
   
   // If debug info requested, add debug flag to payload (server will return full result)
   if(ShowDebugInfo)
   {
      // Insert debug flag before closing brace of JSON
      int insertPos = StringFind(jsonData, "}", StringLen(jsonData) - 2);
      if(insertPos > 0)
      {
         string jsonWithDebug = StringSubstr(jsonData, 0, insertPos);
         jsonWithDebug += ",\"debug\":true";
         jsonWithDebug += StringSubstr(jsonData, insertPos, StringLen(jsonData) - insertPos);
         jsonData = jsonWithDebug;
      }
   }

   StringToCharArray(jsonData, post, 0, StringLen(jsonData));
   
   if(ShowDebugInfo)
      Print("Sending request to: ", RestServerURL);
   
   ResetLastError();
   int res = WebRequest(
      "POST",
      RestServerURL,
      headers,
      5000,  // timeout
      post,
      result,
      headers
   );
   
   if(res == -1)
   {
      int error = GetLastError();
      Print("ERROR: WebRequest failed. Error: ", error);
      if(error == 4060)
         Print("  URL not allowed. Add to Tools > Options > Expert Advisors");
      return;
   }
   
   // Parse response
   string response = CharArrayToString(result);
   
   if(ShowDebugInfo)
      Print("Response received: ", StringSubstr(response, 0, 200), "...");
   
   // Save full response to debug file for later inspection
   if(ShowDebugInfo)
   {
      int h = FileOpen("BlackIce_Response_Debug.jsonl", FILE_WRITE|FILE_READ|FILE_COMMON);
      if(h != INVALID_HANDLE)
      {
         FileSeek(h, 0, SEEK_END);
         FileWriteString(h, response + "\n");
         FileClose(h);
      }
   }
   
   ParseAndExecute(response);
   successCount++;
}

//+------------------------------------------------------------------+
//| Prepare OHLCV JSON data                                          |
//+------------------------------------------------------------------+
string PrepareOHLCVData()
{
   // Check if we have enough bars on ANALYSIS timeframe
   int barsAvailable = Bars(_Symbol, AnalysisTimeframe);
   if(barsAvailable < BarsToSend + 10)
   {
      Print("WARNING: Insufficient bars on ", EnumToString(AnalysisTimeframe), 
            ". Need ", BarsToSend, ", have ", barsAvailable);
      return "";
   }
   
   string json = "{\"ohlcv\":[";
   
   // Use ANALYSIS timeframe, not chart timeframe
   for(int i = BarsToSend - 1; i >= 0; i--)
   {
      datetime time = iTime(_Symbol, AnalysisTimeframe, i);
      double open = iOpen(_Symbol, AnalysisTimeframe, i);
      double high = iHigh(_Symbol, AnalysisTimeframe, i);
      double low = iLow(_Symbol, AnalysisTimeframe, i);
      double close = iClose(_Symbol, AnalysisTimeframe, i);
      long volume = iVolume(_Symbol, AnalysisTimeframe, i);
      
      json += "{";
      json += "\"time\":\"" + TimeToString(time, TIME_DATE|TIME_MINUTES) + "\",";
      json += "\"open\":" + DoubleToString(open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(volume);
      json += "}";
      
      if(i > 0) json += ",";
   }
   
   json += "]}";
   
   return json;
}

//+------------------------------------------------------------------+
//| Parse response and execute trade                                 |
//+------------------------------------------------------------------+
void ParseAndExecute(string response)
{
   // Simple JSON parsing (production would use proper JSON library)
   int prediction = -1;
   double confidence = 0.0;
   double prob_sell = 0.0;
   double prob_hold = 0.0;
   double prob_buy = 0.0;
   
   // Extract prediction
   int predPos = StringFind(response, "\"prediction\":");
   if(predPos >= 0)
   {
      string predStr = StringSubstr(response, predPos + 13, 1);
      prediction = (int)StringToInteger(predStr);
   }
   
   // Extract confidence
   int confPos = StringFind(response, "\"confidence\":");
   if(confPos >= 0)
   {
      int confEnd = StringFind(response, ",", confPos);
      string confStr = StringSubstr(response, confPos + 13, confEnd - confPos - 13);
      confidence = StringToDouble(confStr);
   }
   
   // Extract probabilities
   int sellPos = StringFind(response, "\"SELL\":");
   if(sellPos >= 0)
   {
      int sellEnd = StringFind(response, ",", sellPos);
      string sellStr = StringSubstr(response, sellPos + 7, sellEnd - sellPos - 7);
      prob_sell = StringToDouble(sellStr);
   }
   
   int holdPos = StringFind(response, "\"HOLD\":");
   if(holdPos >= 0)
   {
      int holdEnd = StringFind(response, ",", holdPos);
      string holdStr = StringSubstr(response, holdPos + 7, holdEnd - holdPos - 7);
      prob_hold = StringToDouble(holdStr);
   }
   
   int buyPos = StringFind(response, "\"BUY\":");
   if(buyPos >= 0)
   {
      int buyEnd = StringFind(response, "}", buyPos);
      string buyStr = StringSubstr(response, buyPos + 6, buyEnd - buyPos - 6);
      prob_buy = StringToDouble(buyStr);
   }
   
   // Normalize probabilities to sum to 100%
   double total = prob_sell + prob_hold + prob_buy;
   if(total > 0)
   {
      prob_sell = prob_sell / total;
      prob_hold = prob_hold / total;
      prob_buy = prob_buy / total;
   }
   
   // Extract SMC context from response
   ExtractSMCContext(response);
   
   // Display results
   string predLabel = (prediction == 0 ? "SELL" : (prediction == 1 ? "HOLD" : "BUY"));
   
   Print("========================================");
   Print("PREDICTION: ", predLabel);
   Print("Confidence: ", DoubleToString(confidence * 100, 1), "%");
   Print("Probabilities:");
   Print("  SELL: ", DoubleToString(prob_sell * 100, 1), "%");
   Print("  HOLD: ", DoubleToString(prob_hold * 100, 1), "%");
   Print("  BUY:  ", DoubleToString(prob_buy * 100, 1), "%");
   Print("SMC Context: ", smcOrderBlocks, " | ", smcFairValueGaps);
   Print("========================================");
   
   // Update chart comment with color coding
   DisplayColoredInfo(predLabel, confidence, prob_sell, prob_hold, prob_buy);
   
   // Check for exit signals first (always active)
   if(EnableTrading)
   {
      CheckExitSignals(prediction, confidence);
   }
   
   // Execute trade if confidence is sufficient AND conditions are good
   if(confidence >= MinConfidence && EnableTrading)
   {
      if(IsGoodTradingCondition())
      {
         ExecuteTrade(prediction, confidence);
      }
   }
   else if(confidence < MinConfidence)
   {
      Print("SKIPPED: Confidence ", DoubleToString(confidence * 100, 1), "% below minimum ", DoubleToString(MinConfidence * 100, 0), "%");
   }
}

//+------------------------------------------------------------------+
//| Execute trade based on prediction                                |
//+------------------------------------------------------------------+
void ExecuteTrade(int prediction, double confidence)
{
   // Close opposite positions first
   CloseOppositePositions(prediction);
   
   // Check if we already have a position in this direction
   if(HasPosition(prediction))
   {
      Print("Already have position in this direction");
      return;
   }
   
   // Calculate position size based on confidence, regime and recovery
   double lots = ComputeLotSize(prediction, confidence);
   
   // Normalize lot size
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   lots = MathMax(minLot, MathMin(maxLot, MathRound(lots / lotStep) * lotStep));
   
   // Calculate SL/TP
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double sl = 0, tp = 0;
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   ConfigureTradeRequest(request);
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lots;
   request.deviation = 10;
   request.magic = 123456;
   
   if(prediction == 2)  // BUY
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.type = ORDER_TYPE_BUY;
      request.price = ask;
      
      sl = ask - StopLossPips * point * 10;
      tp = ask + (StopLossPips * MinRiskReward) * point * 10;
      
      request.sl = NormalizeDouble(sl, digits);
      request.tp = NormalizeDouble(tp, digits);
      
      Print("EXECUTING BUY: ", lots, " lots at ", ask, " (SL:", sl, " TP:", tp, ")");
      LogTrade("BUY", confidence, ask, sl, tp);
   }
   else if(prediction == 0)  // SELL
   {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      request.type = ORDER_TYPE_SELL;
      request.price = bid;
      
      sl = bid + StopLossPips * point * 10;
      tp = bid - (StopLossPips * MinRiskReward) * point * 10;
      
      request.sl = NormalizeDouble(sl, digits);
      request.tp = NormalizeDouble(tp, digits);
      
      Print("EXECUTING SELL: ", lots, " lots at ", bid, " (SL:", sl, " TP:", tp, ")");
      LogTrade("SELL", confidence, bid, sl, tp);
   }
   else
   {
      Print("HOLD signal - no trade");
      return;
   }
   
   // Send order
   if(!OrderSend(request, result))
   {
      Print("ERROR: OrderSend failed. Error: ", GetLastError());
      Print("  Retcode: ", result.retcode);
   }
   else
   {
      Print("SUCCESS: Order placed. Ticket: ", result.order);
   }
}


//+------------------------------------------------------------------+
//| Compute lot size based on confidence, regime and recovery logic  |
//+------------------------------------------------------------------+
double ComputeLotSize(int prediction, double confidence)
{
   double base = BaseLotSize;

   // Start multiplier from confidence tiers
   double mult = 1.0;
   if(confidence >= ConfidenceHigh)
      mult = 2.0;
   else if(confidence >= ConfidenceMed)
      mult = 1.5;
   else if(confidence >= ConfidenceLow)
      mult = 1.0;
   else if(confidence >= MinConfidence)
      mult = 0.5;
   else
      mult = 0.0; // below min confidence, should not trade

   // Penalize trades that go against regime
   if(smcRegime == "Bearish" && prediction == 2)
      mult *= 0.5; // buy vs bearish
   if(smcRegime == "Bullish" && prediction == 0)
      mult *= 0.5; // sell vs bullish

   // Apply compensatory recovery multiplier
   if(EnableCompensatory && consecutiveLosses > 0)
   {
      int steps = MathMin(consecutiveLosses, RecoveryDepth);
      mult *= MathPow(RecoveryMultiplier, steps);
   }

   // Respect UseSystemRisk / UserRiskPercent
   if(!UseSystemRisk)
   {
      // If user controls risk, scale base by user percent relative to a nominal percent (0.5%)
      double nominal = 0.5;
      double factor = MathMax(0.01, UserRiskPercent / nominal);
      mult *= factor;
   }

   // Final lot calculation and normalization
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   double raw = base * mult;
   double rounded = MathRound(raw / lotStep) * lotStep;
   double lots = MathMax(minLot, MathMin(maxLot, rounded));

   if(ShowDebugInfo)
      PrintFormat("Lot sizing: base=%.4f mult=%.3f -> %.4f (min=%.4f max=%.4f)", base, mult, lots, minLot, maxLot);

   return lots;
}

//+------------------------------------------------------------------+
//| Check if we have a position in this direction                    |
//+------------------------------------------------------------------+
bool HasPosition(int prediction)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            if(prediction == 2 && posType == POSITION_TYPE_BUY)
               return true;
            if(prediction == 0 && posType == POSITION_TYPE_SELL)
               return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Close opposite positions                                         |
//+------------------------------------------------------------------+
void CloseOppositePositions(int prediction)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            bool shouldClose = false;
            if(prediction == 2 && posType == POSITION_TYPE_SELL)
               shouldClose = true;
            if(prediction == 0 && posType == POSITION_TYPE_BUY)
               shouldClose = true;
            
            if(shouldClose)
            {
               MqlTradeRequest request = {};
               MqlTradeResult result = {};
               ConfigureTradeRequest(request);
               
               request.action = TRADE_ACTION_DEAL;
               request.position = PositionGetInteger(POSITION_TICKET);
               request.symbol = _Symbol;
               request.volume = PositionGetDouble(POSITION_VOLUME);
               request.deviation = 10;
               request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
               request.price = (posType == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               
               if(OrderSend(request, result))
                  Print("Closed opposite position: ", request.position);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Extract SMC context from JSON response                           |
//+------------------------------------------------------------------+
void ExtractSMCContext(string response)
{
   // Reset SMC context
   smcOrderBlocks = "";
   smcFairValueGaps = "";
   smcStructure = "";
   smcRegime = "";
   
   // Extract Order Blocks
   if(StringFind(response, "\"bullish_present\":true") >= 0)
      smcOrderBlocks += "Bullish OB ";
   if(StringFind(response, "\"bearish_present\":true") >= 0)
      smcOrderBlocks += "Bearish OB ";
   if(smcOrderBlocks == "")
      smcOrderBlocks = "No OB";
   
   // Extract Fair Value Gaps
   int fvgPos = StringFind(response, "\"fair_value_gaps\"");
   if(fvgPos >= 0)
   {
      string fvgSection = StringSubstr(response, fvgPos, 200);
      if(StringFind(fvgSection, "\"bullish_present\":true") >= 0)
         smcFairValueGaps += "Bullish FVG ";
      if(StringFind(fvgSection, "\"bearish_present\":true") >= 0)
         smcFairValueGaps += "Bearish FVG ";
   }
   if(smcFairValueGaps == "")
      smcFairValueGaps = "No FVG";
   
   // Extract Structure
   if(StringFind(response, "\"bos_close_confirmed\":true") >= 0)
      smcStructure = "BOS Confirmed";
   else if(StringFind(response, "\"bos_wick_confirmed\":true") >= 0)
      smcStructure = "BOS Wick";
   else
      smcStructure = "No BOS";
   
   // Extract Regime
   int regimePos = StringFind(response, "\"trend_bias\":");
   if(regimePos >= 0)
   {
      // Find end of the numeric value; prefer comma, fallback to closing brace or end
      int regimeEnd = StringFind(response, ",", regimePos);
      if(regimeEnd < 0)
         regimeEnd = StringFind(response, "}", regimePos);

      // Safety: if still not found, use end of string
      if(regimeEnd < 0)
         regimeEnd = StringLen(response);

      string biasStr = StringSubstr(response, regimePos + StringLen("\"trend_bias\":"), regimeEnd - (regimePos + StringLen("\"trend_bias\":")));
      double bias = StringToDouble(biasStr);

      // Use same thresholds as server (0.1) to determine regime
      if(bias > 0.1)
         smcRegime = "Bullish";
      else if(bias < -0.1)
         smcRegime = "Bearish";
      else
         smcRegime = "Neutral";
   }
}

//+------------------------------------------------------------------+
//| Display colored information on chart                             |
//+------------------------------------------------------------------+
void DisplayColoredInfo(string predLabel, double confidence, double prob_sell, double prob_hold, double prob_buy)
{
   // Determine colors based on prediction
   color predColor = clrGray;
   if(predLabel == "BUY")
      predColor = clrLime;
   else if(predLabel == "SELL")
      predColor = clrRed;
   else
      predColor = clrOrange;
   
   // Confidence color
   color confColor = clrGray;
   if(confidence >= 0.70)
      confColor = clrLime;
   else if(confidence >= 0.60)
      confColor = clrYellow;
   else if(confidence >= 0.55)
      confColor = clrOrange;
   else
      confColor = clrRed;
   
   // Build comment - simpler format for better visibility
   string comment = "";
   comment += "BLACK ICE V2 - INSTITUTIONAL MODELS\n";
   comment += "═══════════════════════════════════════\n";
   comment += "\n";
   comment += "Analysis TF: " + EnumToString(AnalysisTimeframe) + "\n";
   
   switch(UpdateIntervalSeconds){
      case 0:
            comment += "Update: New Bar Only\n";
         break;
      case 1:
         comment += "Update: Every " + IntegerToString(updateSeconds) + "s\n";
         break;
   }
   
   comment += "\n";
   comment += "PREDICTION: " + predLabel + "\n";
   comment += "Confidence: " + DoubleToString(confidence * 100, 1) + "%\n";
   comment += "───────────────────────────────────────\n";
   comment += "SELL: " + DoubleToString(prob_sell * 100, 1) + "%\n";
   comment += "HOLD: " + DoubleToString(prob_hold * 100, 1) + "%\n";
   comment += "BUY:  " + DoubleToString(prob_buy * 100, 1) + "%\n";
   comment += "\n";
   comment += "SMC CONTEXT\n";
   comment += "───────────────────────────────────────\n";
   comment += "Order Blocks: " + smcOrderBlocks + "\n";
   comment += "Fair Value Gaps: " + smcFairValueGaps + "\n";
   comment += "Structure: " + smcStructure + "\n";
   comment += "Regime: " + smcRegime + "\n";
   comment += "\n";
   comment += "───────────────────────────────────────\n";
   comment += "Min Confidence: " + DoubleToString(MinConfidence * 100, 0) + "%\n";
   comment += "Trading: " + (EnableTrading ? "ENABLED" : "DEMO") + "\n";
   comment += "Requests: " + IntegerToString(requestCount) + " | Success: " + IntegerToString(successCount) + "\n";
   
   // Add signal status
   comment += "\n";
   if(confidence >= MinConfidence)
   {
      if(predLabel == "BUY")
         comment += ">>> SIGNAL: READY TO BUY <<<\n";
      else if(predLabel == "SELL")
         comment += ">>> SIGNAL: READY TO SELL <<<\n";
      else
         comment += ">>> SIGNAL: HOLD POSITION <<<\n";
   }
   else
   {
      comment += ">>> CONFIDENCE TOO LOW - WAITING <<<\n";
   }
   
   Comment(comment);
   
   // Set chart foreground color to make text more visible
   ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrYellow);
}
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| Initialize trade log file                                        |
//+------------------------------------------------------------------+
void InitializeTradeLog()
{
   int handle = FileOpen(logFileName, FILE_WRITE|FILE_CSV|FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      // Write header
      FileWrite(handle, "Timestamp", "Symbol", "Action", "Confidence", "Entry", "SL", "TP", 
                "OB_Context", "FVG_Context", "Regime", "ATR", "Spread");
      FileClose(handle);
      Print("✅ Trade log initialized: ", logFileName);
   }
   else
   {
      Print("❌ Failed to create trade log");
   }
}

//+------------------------------------------------------------------+
//| Configure default fields for trade requests                      |
//+------------------------------------------------------------------+
void ConfigureTradeRequest(MqlTradeRequest &request)
{
   // Set order time type to GTC by default
   request.type_time = ORDER_TIME_GTC;
   // Use a safe default filling mode. Some MT5 builds/brokers do not expose
   // SYMBOL_FILLING_MODE at compile-time, so avoid querying it here.
   request.type_filling = ORDER_FILLING_IOC;
}

//+------------------------------------------------------------------+
//| Log trade to CSV                                                 |
//+------------------------------------------------------------------+
void LogTrade(string action, double confidence, double entry, double sl, double tp)
{
   int handle = FileOpen(logFileName, FILE_WRITE|FILE_READ|FILE_CSV|FILE_COMMON);
   if(handle != INVALID_HANDLE)
   {
      FileSeek(handle, 0, SEEK_END);
      
      int atr_handle = iATR(_Symbol, AnalysisTimeframe, 14);
      double atr = 0;
      if(atr_handle != INVALID_HANDLE)
      {
         double atr_buffer[1];
         if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) == 1)
            atr = atr_buffer[0];
         IndicatorRelease(atr_handle);
      }
      
      double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      
      FileWrite(handle,
         TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES),
         _Symbol,
         action,
         DoubleToString(confidence, 3),
         DoubleToString(entry, _Digits),
         DoubleToString(sl, _Digits),
         DoubleToString(tp, _Digits),
         smcOrderBlocks,
         smcFairValueGaps,
         smcRegime,
         DoubleToString(atr, _Digits),
         DoubleToString(spread, _Digits)
      );
      
      FileClose(handle);
      Print("📝 Trade logged: ", action, " at ", entry);
   }
}

//+------------------------------------------------------------------+
//| Check if trading conditions are good                             |
//+------------------------------------------------------------------+
bool IsGoodTradingCondition()
{
   // Check volatility
   int atr_handle = iATR(_Symbol, AnalysisTimeframe, 14);
   double atr = 0;
   double atr_avg = 0;
   
   if(atr_handle == INVALID_HANDLE)
      return true;  // If can't check, allow trade
   
   double atr_array[];
   ArraySetAsSeries(atr_array, true);
   
   // Get current ATR and 100-period average
   if(CopyBuffer(atr_handle, 0, 0, 100, atr_array) == 100)
   {
      atr = atr_array[0];
      
      double atr_sum = 0;
      for(int i = 0; i < 100; i++)
         atr_sum += atr_array[i];
      atr_avg = atr_sum / 100;
   }
   
   IndicatorRelease(atr_handle);
   
   if(atr < atr_avg * 0.5)
   {
      Print("⚠️ Skipping trade: Volatility too low (ATR: ", atr, " < ", atr_avg * 0.5, ")");
      return false;
   }
   
   // Check spread
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(spread > atr * 0.3)
   {
      Print("⚠️ Skipping trade: Spread too wide (", spread, " > ", atr * 0.3, ")");
      return false;
   }
   
   // Check time (avoid low liquidity hours) - UTC time
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int hour = dt.hour;
   
   // Skip Asian session dead hours (23:00-06:00 UTC)
   if(hour >= 23 || hour < 6)
   {
      Print("⚠️ Skipping trade: Low liquidity hours (", hour, ":00 UTC)");
      return false;
   }
   
   return true;
}


//+------------------------------------------------------------------+
//| Check for exit signals on open positions                         |
//+------------------------------------------------------------------+
void CheckExitSignals(int prediction, double confidence)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            
            bool shouldExit = false;
            string exitReason = "";
            
            // Exit long if strong SELL signal
            if(posType == POSITION_TYPE_BUY && prediction == 0 && confidence >= 0.60)
            {
               shouldExit = true;
               exitReason = "AI Exit: Strong SELL signal";
            }
            
            // Exit short if strong BUY signal
            if(posType == POSITION_TYPE_SELL && prediction == 2 && confidence >= 0.60)
            {
               shouldExit = true;
               exitReason = "AI Exit: Strong BUY signal";
            }
            
            if(shouldExit)
            {
               MqlTradeRequest request = {};
               MqlTradeResult result = {};
               ConfigureTradeRequest(request);
               
               request.action = TRADE_ACTION_DEAL;
               request.position = ticket;
               request.symbol = _Symbol;
               request.volume = PositionGetDouble(POSITION_VOLUME);
               request.deviation = 10;
               request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
               request.price = (posType == POSITION_TYPE_BUY) ? 
                  SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                  SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               
               if(OrderSend(request, result))
               {
                  Print("✅ ", exitReason, " - Ticket: ", ticket);
                  LogTrade("EXIT_" + (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"), 
                          confidence, request.price, 0, 0);
                 // Bookkeeping: assume exit was taken; we can't know pnl here without OnTradeTransaction
                 consecutiveLosses = MathMax(0, consecutiveLosses - 1); // conservatively reduce recovery
               }
            }
         }
      }
   }
}

// Placeholder: this should be hooked to OnTradeTransaction to get actual closed trade pnl
void OnTradeClosed(ulong ticket, double profit)
{
   lastTradeProfit = profit;
   if(profit < 0)
      consecutiveLosses++;
   else
      consecutiveLosses = MathMax(0, consecutiveLosses - 1);

   if(ShowDebugInfo)
      PrintFormat("Trade closed: ticket=%I64u profit=%.2f consecutiveLosses=%d", ticket, profit, consecutiveLosses);
}
