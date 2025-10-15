//+------------------------------------------------------------------+
//|                    Black Ice Intelligence MT5 EA                 |
//|                    Connects to Python REST API Server            |
//|                    Displays AI Trading Decisions                  |
//+------------------------------------------------------------------+
#property copyright "Black Ice Intelligence"
#property link      "https://github.com/Morriase/integrated-pipeline"
#property version   "1.00"
#property strict

//--- Input parameters
input string ServerURL = "http://localhost:5001";  // REST API Server URL
input int UpdateInterval = 60;                     // Update interval in seconds
input double RiskMultiplier = 1.0;                 // Risk multiplier for position sizing
input bool ShowModelBreakdown = true;              // Show individual model predictions
input color CommentColor = clrTeal;                // Color for chart comments

//--- Global variables
datetime lastUpdate = 0;
string lastDecision = "";
double lastConfidence = 0.0;
string lastModelBreakdown = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🧊 Black Ice Intelligence EA initialized");
    Print("📡 Connecting to server: ", ServerURL);
    Print("⏰ Update interval: ", UpdateInterval, " seconds");

    // Test server connection
    if(!TestServerConnection())
    {
        Print("❌ Cannot connect to server. Please ensure Python server is running.");
        return(INIT_FAILED);
    }

    Print("✅ Server connection successful");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🧊 Black Ice Intelligence EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if it's time to update
    if(TimeCurrent() - lastUpdate < UpdateInterval)
        return;

    lastUpdate = TimeCurrent();

    // Get prediction from server
    string response = GetPredictionFromServer();
    if(response == "")
    {
        Comment("❌ Failed to get prediction from server");
        return;
    }

    // Parse response and update chart
    ParseAndDisplayPrediction(response);
}

//+------------------------------------------------------------------+
//| Test server connection                                           |
//+------------------------------------------------------------------+
bool TestServerConnection()
{
    string url = ServerURL + "/health";
    uchar empty_data[];
    uchar result[];
    string result_headers;
    int res = WebRequest("GET", url, "", 5000, empty_data, result, result_headers);

    if(res != 200 || ArraySize(result) == 0)
        return false;

    return true;
}

//+------------------------------------------------------------------+
//| Get prediction from REST API server                             |
//+------------------------------------------------------------------+
string GetPredictionFromServer()
{
    // Prepare OHLCV data (last 50 candles for feature calculation)
    string ohlcv_json = PrepareOHLCVData(50);

    if(ohlcv_json == "")
        return "";

    // Prepare POST data
    string post_data = StringFormat("{\"ohlcv_data\": %s}", ohlcv_json);

    // Convert string to char array for WebRequest
    uchar post_data_array[];
    StringToCharArray(post_data, post_data_array);

    // Send request to server
    string url = ServerURL + "/predict";
    uchar result[];
    string result_headers;
    int res = WebRequest("POST", url, "Content-Type: application/json\r\n", 10000, post_data_array, result, result_headers);

    if(res != 200)
    {
        Print("WebRequest error: ", res);
        return "";
    }

    // Convert char array back to string
    string response = CharArrayToString(result);
    return response;
}

//+------------------------------------------------------------------+
//| Prepare OHLCV data as JSON                                       |
//+------------------------------------------------------------------+
string PrepareOHLCVData(int candles_count)
{
    if(Bars(_Symbol, _Period) < candles_count)
        return "";

    string json_array = "[";

    for(int i = candles_count - 1; i >= 0; i--)
    {
        datetime time = iTime(_Symbol, _Period, i);
        double open_price = iOpen(_Symbol, _Period, i);
        double high_price = iHigh(_Symbol, _Period, i);
        double low_price = iLow(_Symbol, _Period, i);
        double close_price = iClose(_Symbol, _Period, i);
        long volume = iVolume(_Symbol, _Period, i);

        string candle_json = StringFormat(
            "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
            TimeToString(time, TIME_DATE|TIME_MINUTES),
            open_price, high_price, low_price, close_price, volume
        );

        json_array += candle_json;

        if(i > 0)
            json_array += ",";
    }

    json_array += "]";

    return json_array;
}

//+------------------------------------------------------------------+
//| Parse prediction response and display on chart                  |
//+------------------------------------------------------------------+
void ParseAndDisplayPrediction(string response)
{
    // Parse JSON response (simplified parsing)
    string decision = ExtractValue(response, "decision");
    string confidence_str = ExtractValue(response, "confidence");
    double confidence = StringToDouble(confidence_str);

    if(decision == "" || confidence == 0.0)
    {
        Comment("❌ Failed to parse server response");
        return;
    }

    // Store for global access
    lastDecision = decision;
    lastConfidence = confidence;

    // Create natural language interpretation
    string interpretation = CreateNaturalLanguageInterpretation(decision, confidence);

    // Get model breakdown if enabled
    if(ShowModelBreakdown)
    {
        string model_breakdown = ExtractModelBreakdown(response);
        if(model_breakdown != "")
            interpretation += "\n\n🤖 Model Analysis:\n" + model_breakdown;
    }

    // Display on chart
    Comment(interpretation);

    // Log to terminal
    Print("🧊 AI Decision: ", decision, " (", DoubleToString(confidence * 100, 1), "% confidence)");
}

//+------------------------------------------------------------------+
//| Create natural language interpretation                           |
//+------------------------------------------------------------------+
string CreateNaturalLanguageInterpretation(string decision, double confidence)
{
    string interpretation = "🧊 BLACK ICE INTELLIGENCE ANALYSIS\n";
    interpretation += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Time and symbol info
    interpretation += StringFormat("📊 %s | %s\n", _Symbol, TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES));
    interpretation += StringFormat("💰 Current Price: %.5f\n\n", iClose(_Symbol, _Period, 0));

    // Decision with confidence
    string confidence_level = GetConfidenceLevel(confidence);
    string decision_icon = GetDecisionIcon(decision);

    interpretation += StringFormat("🎯 TRADING DECISION: %s %s\n", decision_icon, decision);
    interpretation += StringFormat("📈 Confidence Level: %.1f%% (%s)\n\n", confidence * 100, confidence_level);

    // Natural language explanation
    interpretation += "💭 ANALYSIS:\n";
    interpretation += GetDecisionExplanation(decision, confidence);

    return interpretation;
}

//+------------------------------------------------------------------+
//| Get confidence level description                                 |
//+------------------------------------------------------------------+
string GetConfidenceLevel(double confidence)
{
    if(confidence >= 0.8) return "Very High";
    if(confidence >= 0.7) return "High";
    if(confidence >= 0.6) return "Moderate";
    if(confidence >= 0.5) return "Low";
    return "Very Low";
}

//+------------------------------------------------------------------+
//| Get decision icon                                                |
//+------------------------------------------------------------------+
string GetDecisionIcon(string decision)
{
    if(decision == "BUY") return "🟢";
    if(decision == "SELL") return "🔴";
    return "🟡";
}

//+------------------------------------------------------------------+
//| Get natural language decision explanation                       |
//+------------------------------------------------------------------+
string GetDecisionExplanation(string decision, double confidence)
{
    string explanation = "";

    if(decision == "BUY")
    {
        if(confidence >= 0.8)
            explanation = "The AI ensemble strongly detects bullish momentum with high conviction. Multiple technical indicators align for upward movement. Consider entering long positions with confidence.";
        else if(confidence >= 0.6)
            explanation = "The AI detects potential buying opportunities with moderate confidence. Some indicators suggest upward pressure, though market conditions remain uncertain.";
        else
            explanation = "The AI leans toward buying but with limited confidence. Exercise caution and wait for confirmation from additional signals.";
    }
    else if(decision == "SELL")
    {
        if(confidence >= 0.8)
            explanation = "The AI ensemble strongly detects bearish momentum with high conviction. Multiple technical indicators align for downward movement. Consider entering short positions with confidence.";
        else if(confidence >= 0.6)
            explanation = "The AI detects potential selling opportunities with moderate confidence. Some indicators suggest downward pressure, though market conditions remain uncertain.";
        else
            explanation = "The AI leans toward selling but with limited confidence. Exercise caution and wait for confirmation from additional signals.";
    }
    else // HOLD
    {
        if(confidence >= 0.7)
            explanation = "The AI recommends maintaining current positions or staying out of the market. No strong directional bias detected. Current market conditions favor patience over action.";
        else
            explanation = "The AI cannot determine a clear directional bias. Market conditions are mixed or unclear. Consider reducing exposure or waiting for stronger signals.";
    }

    return explanation;
}

//+------------------------------------------------------------------+
//| Extract model breakdown from response                           |
//+------------------------------------------------------------------+
string ExtractModelBreakdown(string response)
{
    string breakdown = "";

    // Extract individual model predictions (simplified)
    string models[] = {"deep_nn", "wide_nn", "compact_nn", "regularized_nn",
                      "random_forest_deep", "random_forest_wide",
                      "gradient_boosting", "logistic_regression"};

    for(int i = 0; i < ArraySize(models); i++)
    {
        string model_response = ExtractModelResponse(response, models[i]);
        if(model_response != "")
            breakdown += StringFormat("• %s: %s\n", models[i], model_response);
    }

    return breakdown;
}

//+------------------------------------------------------------------+
//| Extract individual model response                               |
//+------------------------------------------------------------------+
string ExtractModelResponse(string response, string model_name)
{
    // Simplified JSON parsing - in production, use proper JSON library
    string search_pattern = StringFormat("\"%s\":{\"action\":\"", model_name);
    int start_pos = StringFind(response, search_pattern);

    if(start_pos < 0)
        return "";

    start_pos += StringLen(search_pattern);
    int end_pos = StringFind(response, "\"", start_pos);

    if(end_pos < 0)
        return "";

    string action = StringSubstr(response, start_pos, end_pos - start_pos);

    // Find confidence
    string conf_pattern = "\"confidence\":";
    int conf_start = StringFind(response, conf_pattern, end_pos);
    if(conf_start >= 0)
    {
        conf_start += StringLen(conf_pattern);
        int conf_end = StringFind(response, "}", conf_start);
        if(conf_end >= 0)
        {
            string conf_str = StringSubstr(response, conf_start, conf_end - conf_start);
            double conf = StringToDouble(conf_str);
            return StringFormat("%s (%.1f%%)", action, conf * 100);
        }
    }

    return action;
}

//+------------------------------------------------------------------+
//| Extract value from JSON response                                 |
//+------------------------------------------------------------------+
string ExtractValue(string response, string key)
{
    string pattern = StringFormat("\"%s\":\"", key);
    int start_pos = StringFind(response, pattern);

    if(start_pos < 0)
    {
        // Try numeric value
        pattern = StringFormat("\"%s\":", key);
        start_pos = StringFind(response, pattern);
        if(start_pos < 0)
            return "";
        start_pos += StringLen(pattern);

        // Find end of number
        int end_pos = start_pos;
        while(end_pos < StringLen(response) &&
              (StringGetCharacter(response, end_pos) == '.' ||
               (StringGetCharacter(response, end_pos) >= '0' &&
                StringGetCharacter(response, end_pos) <= '9')))
            end_pos++;

        return StringSubstr(response, start_pos, end_pos - start_pos);
    }

    start_pos += StringLen(pattern);
    int end_pos = StringFind(response, "\"", start_pos);

    if(end_pos < 0)
        return "";

    return StringSubstr(response, start_pos, end_pos - start_pos);
}

//+------------------------------------------------------------------+
//| Execute trading decision                                         |
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| Execute trading decision                                         |
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//| Execute trading decision                                         |
//+------------------------------------------------------------------+
void ExecuteTrade(string decision, double confidence)
{
    // Only execute if confidence is above threshold
    if(confidence < 0.6)
        return;

    double lot_size = CalculateLotSize(confidence);

    if(decision == "BUY")
    {
        // Execute buy order
        // OrderSend(Symbol(), OP_BUY, lot_size, Ask, 3, 0, 0, "BlackIce AI", 0, 0, clrGreen);
        Print("🟢 BUY SIGNAL: Would execute buy order with ", DoubleToString(lot_size, 2), " lots");
    }
    else if(decision == "SELL")
    {
        // Execute sell order
        // OrderSend(Symbol(), OP_SELL, lot_size, Bid, 3, 0, 0, "BlackIce AI", 0, 0, clrRed);
        Print("🔴 SELL SIGNAL: Would execute sell order with ", DoubleToString(lot_size, 2), " lots");
    }
}

//+------------------------------------------------------------------+
//| Calculate position size based on confidence                      |
//+------------------------------------------------------------------+
double CalculateLotSize(double confidence)
{
    // Base lot size (adjust according to your risk management)
    double base_lot = 0.01;

    // Scale lot size with confidence
    double scaled_lot = base_lot * confidence * RiskMultiplier;

    // Ensure minimum lot size
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    if(scaled_lot < min_lot)
        scaled_lot = min_lot;

    // Ensure maximum lot size
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    if(scaled_lot > max_lot)
        scaled_lot = max_lot;

    return NormalizeDouble(scaled_lot, 2);
}
//+------------------------------------------------------------------+