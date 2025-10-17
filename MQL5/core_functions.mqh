//+------------------------------------------------------------------+
//| core_functions.mqh                                               |
//| Modularized core functions (news filter, trailing stops, etc.)   |
//+------------------------------------------------------------------+

// Required trading helper includes (PositionInfo and Trade)
#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>
#include "BlackIce_MT5_EA.mq5" 

bool           TrDisableNews;
datetime       LastNewsAvoided;
string         newsToAvoid[];


string TradingEnabledComm;

// Local helper objects used by the include
CPositionInfo  posinfo;
CTrade         trade;

///+------------------------------------------------------------------+
//| Function to Find High                                            |
//+------------------------------------------------------------------+
double findHigh(){
   double highestHigh = 0; 
      for (int i = 0; i < 200; i++){
         double high = iHigh(_Symbol, AnalysisTimeframe, i ); 
         if(i > BarsN && iHighest(_Symbol, AnalysisTimeframe, MODE_HIGH, BarsN*2+1, i-BarsN) == i){
            if(high > highestHigh){
               return high; 
            }
         }
         highestHigh = MathMax(high, highestHigh); 
      }
    return -1; 
}
//+------------------------------------------------------------------+
//| Function to Find Low                                             |
//+------------------------------------------------------------------+
double findLow(){
   double lowestLow = DBL_MAX; 
      for (int i = 0; i < 200; i++){
         double low = iLow(_Symbol, AnalysisTimeframe, i); 
         if(i > BarsN && iLowest(_Symbol, AnalysisTimeframe, MODE_LOW, BarsN*2+1, i-BarsN) == i){
            if(low < lowestLow){
               return low; 
         }
      }
      lowestLow = MathMin(low, lowestLow); 
    }
    return -1; 
}
//+------------------------------------------------------------------+
//| Function to Check Upcoming News                                  |
//+------------------------------------------------------------------+
bool IsUpcomingNews(){
   
   if(NewsFilterOn == false) return (false); 
   
   if(TrDisableNews && TimeCurrent()-LastNewsAvoided < StartTradingMin*PeriodSeconds(PERIOD_M1)) return true; 
   
   TrDisableNews = false; 
   
   string sep; 
   switch (separator){
      case 0: sep = ","; break; 
      case 1: sep = ";";
   }
   
   sep_code = StringGetCharacter(sep, 0); 
   
   int k = StringSplit(keyNews, sep_code,newsToAvoid); 
   
   MqlCalendarValue values[]; 
   datetime start_time = TimeCurrent(); //iTime(_Symbol, PERIOD_D1,0); 
   datetime end_time = start_time + PeriodSeconds(PERIOD_D1)*DaysNewsLookup; 
   
   CalendarValueHistory(values, start_time, end_time, NULL, NULL); 
   
   for (int i = 0; i < ArraySize(values); i++){
       MqlCalendarEvent event; 
       CalendarEventById(values[i].event_id, event); 
       MqlCalendarCountry country; 
       CalendarCountryById(event.country_id, country); 
       
       if(StringFind(NewsCurrencied, country.currency) < 0) continue; 
       
         for (int j = 0; j<k; j++){
            string currentevent = newsToAvoid[j]; 
            string currentnews = event.name; 
            if(StringFind(currentnews, currentevent) < 0) continue; 
            
            Comment("Next News: ", country.currency, ": ", event.name, "-> ", values[i].time); 
            if(values[i].time - TimeCurrent() < StopBeforeMin*PeriodSeconds(PERIOD_M1)){
               LastNewsAvoided = values[i].time; 
               TrDisableNews = true; 
               if(TradingEnabledComm=="" || TradingEnabledComm!="Printed"){
                  TradingEnabledComm = "Trading is disabled due to upcoming news: " + event.name; 
               }
               return true; 
            }
          return false; 
         }
       
   }
   return false;    
}
//+------------------------------------------------------------------+
// Trailing stoploss

void TrailSL(){
   // Get broker requirements
   long stopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minStopDistance = stopLevel * point;
   
   for(int i = PositionsTotal()-1; i>=0; i--){
      posinfo.SelectByIndex(i); 
      long magic = posinfo.Magic();
      ulong ticket = posinfo.Ticket(); 
      ENUM_POSITION_TYPE postype = posinfo.PositionType(); 
      string symbol = posinfo.Symbol(); 
      
      if(symbol == _Symbol && magic == inpMagic){
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID); 
         double currentSL = posinfo.StopLoss();
         double tp = posinfo.TakeProfit(); 
         double openPrice = posinfo.PriceOpen(); 
         double high = findHigh(); 
         double low  = findLow(); 
         double newSL = currentSL; // Initialize with current SL
         
         if (trailType==1){
            if(postype==POSITION_TYPE_BUY){
               if (price > posinfo.PriceOpen() && low>0){
                  newSL = low - HighLowBuffer*10*_Point; 
               }
            } else if(postype==POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen() && high>0){
                  newSL = high + HighLowBuffer*10*_Point; 
               } 
            }
         }
         if(trailType==2){
            if(postype == POSITION_TYPE_BUY){
               if(price > posinfo.PriceOpen()){
                  newSL = price - trailFixedpips*10*_Point; 
               }   
            } else if(postype == POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen()){
                  newSL = price + trailFixedpips*10*_Point; 
               }
            }
         }
         if(trailType == 3){
            if(postype==POSITION_TYPE_BUY){
               if (price > posinfo.PriceOpen()){
                  double tpCurrent = posinfo.TakeProfit();
                  double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID); 
                  if(tpCurrent > 0) {
                     double totalTpDistance = tpCurrent - posinfo.PriceOpen();
                     double currentProfit = bid - posinfo.PriceOpen();
                     double profitPercentage = (currentProfit / totalTpDistance) * 100;
                     double trailAmount = bid * (TslPercent/100);
                     if(profitPercentage >= TslPercentTP){
                        newSL = bid - trailAmount; 
                     }
                  }
               }
            } else if(postype==POSITION_TYPE_SELL){
               if(price < posinfo.PriceOpen()){
                  double tpCurrent = posinfo.TakeProfit();
                  double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK); 
                  if(tpCurrent > 0) {
                     double totalTpDistance = posinfo.PriceOpen() - tpCurrent;
                     double currentProfit = posinfo.PriceOpen() - ask;
                     double profitPercentage = (currentProfit / totalTpDistance) * 100;
                     double trailAmount = ask * (TslPercent/100);
                     if(profitPercentage >= TslPercentTP){
                        newSL = ask + trailAmount;
                     }
                  }
               }
            }
         }
         
         // Normalize and validate the new stop loss
         newSL = NormalizeDouble(newSL, _Digits);
         
         // Check if we need to modify the position
         bool shouldModify = false;
         if(postype == POSITION_TYPE_BUY && newSL > currentSL && newSL != currentSL) {
            // For buy positions, only move SL up (trailing)
            if(newSL <= price - minStopDistance) { // Ensure minimum distance from current price
               shouldModify = true;
            }
         } else if(postype == POSITION_TYPE_SELL && newSL < currentSL && newSL != currentSL) {
            // For sell positions, only move SL down (trailing)  
            if(newSL >= price + minStopDistance) { // Ensure minimum distance from current price
               shouldModify = true;
            }
         }
         
         // Only modify if conditions are met and values are different
         if(shouldModify && MathAbs(newSL - currentSL) > point) {
            if(trade.PositionModify(ticket, newSL, tp)) {
               printf("[TSL]: Position #%d modified successfully. Old SL: %.5f, New SL: %.5f", ticket, currentSL, newSL);
            } else {
               printf("[TSL]: Failed to modify position #%d. Error: %d", ticket, GetLastError());
            }
         }
      } // Closes if(symbol == _Symbol && magic == Magic)
   } // Closes for loop
} // Closes function