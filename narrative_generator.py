"""
Trading Narrative Generator
Generates human-readable explanations of AI trading decisions
"""

def generate_trading_narrative(prediction_result: dict, smc_context: dict, features) -> str:
    """
    Generate comprehensive trading narrative for display panel
    
    Args:
        prediction_result: Dictionary with prediction, signal, confidence, etc.
        smc_context: Dictionary with SMC context features
        features: Feature DataFrame (single row)
    
    Returns:
        Multi-paragraph narrative explaining the trading decision
    """
    signal = prediction_result['signal']
    confidence = prediction_result['confidence']
    consensus = prediction_result['consensus']
    models = prediction_result['models']
    
    # Extract context
    ob = smc_context['order_blocks']
    regime = smc_context['regime']
    indicators = smc_context['indicators']
    structure = smc_context['structure']
    
    # Count model agreement
    agreement_count = sum(1 for pred in models.values() if pred == prediction_result['prediction'])
    
    # Build narrative paragraphs
    paragraphs = []
    
    # Paragraph 1: Main recommendation
    if signal == "HOLD":
        paragraphs.append(f"The AI ensemble recommends HOLDING with {confidence*100:.1f}% confidence. "
                         f"Models do not see a high-probability setup at this time.")
    else:
        setup_type = "bearish order block" if ob['bearish_present'] else "bullish order block" if ob['bullish_present'] else "price action"
        structure_conf = ""
        if structure['bos_close_confirmed']:
            structure_conf = " confirmed by break of structure"
        elif structure['choch_detected']:
            structure_conf = " with change of character detected"
        
        paragraphs.append(f"The AI ensemble recommends a {signal} with {confidence*100:.1f}% confidence "
                         f"based on a {setup_type} setup{structure_conf}. "
                         f"{agreement_count} of 3 models predict this trade will be profitable.")
    
    # Paragraph 2: Model consensus details
    model_details = []
    for name, pred in models.items():
        outcome = "WIN" if pred == 1 else "LOSS" if pred == -1 else "TIMEOUT"
        model_details.append(f"{name}: {outcome}")
    
    if consensus:
        if agreement_count == 3:
            paragraphs.append(f"All three models agree on the outcome. {', '.join(model_details)}.")
        else:
            majority = [name for name, pred in models.items() if pred == prediction_result['prediction']]
            minority = [name for name, pred in models.items() if pred != prediction_result['prediction']]
            paragraphs.append(f"Majority consensus: {' and '.join(majority)} predict profitability, "
                             f"while {' and '.join(minority)} suggest{'s' if len(minority)==1 else ''} caution.")
    else:
        paragraphs.append(f"Models are split: {', '.join(model_details)}. No clear consensus.")
    
    # Paragraph 3: Market conditions
    regime_desc = regime['regime_label']
    rsi = indicators['rsi']
    macd_hist = indicators['macd_hist']
    momentum = indicators['momentum']
    vol_ratio = indicators['volume_ma_ratio']
    
    # RSI interpretation
    if rsi > 70:
        rsi_desc = "overbought"
    elif rsi < 30:
        rsi_desc = "oversold"
    elif 45 <= rsi <= 55:
        rsi_desc = "neutral"
    else:
        rsi_desc = "moderate"
    
    # Momentum interpretation
    if abs(momentum) < 0.01:
        mom_desc = "flat momentum"
    elif momentum > 0:
        mom_desc = "bullish momentum"
    else:
        mom_desc = "bearish momentum"
    
    # Volume interpretation
    if vol_ratio > 1.5:
        vol_desc = "significantly above average volume"
    elif vol_ratio > 1.2:
        vol_desc = "above average volume"
    elif vol_ratio < 0.8:
        vol_desc = "below average volume"
    else:
        vol_desc = "normal volume"
    
    paragraphs.append(f"Market conditions show a {regime_desc.lower()} regime with {rsi_desc} RSI ({rsi:.1f}), "
                     f"{mom_desc}, and {vol_desc} ({vol_ratio:.2f}x). "
                     f"MACD histogram is {'positive' if macd_hist >= 0 else 'negative'} ({macd_hist:+.4f}), "
                     f"{'supporting' if (macd_hist > 0 and signal == 'BUY') or (macd_hist < 0 and signal == 'SELL') else 'contradicting'} the {signal} signal.")
    
    # Paragraph 4: Risk assessment
    if confidence >= 0.70:
        risk_level = "high confidence"
    elif confidence >= 0.60:
        risk_level = "moderate confidence"
    elif confidence >= 0.55:
        risk_level = "acceptable confidence"
    else:
        risk_level = "low confidence"
    
    if signal != "HOLD":
        ob_quality = ob.get('bearish_quality' if ob['bearish_present'] else 'bullish_quality', 0)
        ob_age = ob.get('age', 0)
        
        paragraphs.append(f"This is a {risk_level} setup. The order block has {ob_quality*100:.1f}% quality "
                         f"and is {ob_age} bars old. "
                         f"{'The setup aligns with the current regime.' if (regime_desc == 'Bullish' and signal == 'BUY') or (regime_desc == 'Bearish' and signal == 'SELL') else 'Note: This is a counter-trend setup.'}")
    
    # Join paragraphs with line breaks
    return " ".join(paragraphs)
