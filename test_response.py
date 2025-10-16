import requests
import json

test_data = {
    'ohlcv': [
        {'time': 1638360000, 'open': 1.0, 'high': 1.01,
            'low': 0.99, 'close': 1.005, 'volume': 1000},
        {'time': 1638360060, 'open': 1.005, 'high': 1.015,
            'low': 0.995, 'close': 1.01, 'volume': 1100},
        {'time': 1638360120, 'open': 1.01, 'high': 1.02,
            'low': 1.0, 'close': 1.008, 'volume': 900}
    ]
}

response = requests.post('http://127.0.0.1:5001/predict', json=test_data)
data = response.json()

print('Full server response:')
print(json.dumps(data, indent=2))

# Test EA parsing logic
response_str = json.dumps(data)
print('\nEA parsing simulation:')
print('Looking for prediction...')
pred_pos = response_str.find('"prediction":')
if pred_pos >= 0:
    pred_str = response_str[pred_pos + 13:pred_pos + 14]
    print(f'Found prediction: {pred_str}')

print('Looking for confidence...')
conf_pos = response_str.find('"confidence":')
if conf_pos >= 0:
    conf_end = response_str.find(',', conf_pos)
    conf_str = response_str[conf_pos + 13:conf_end]
    print(f'Found confidence: {conf_str}')

print('Looking for SELL probability...')
sell_pos = response_str.find('"SELL":')
if sell_pos >= 0:
    sell_end = response_str.find(',', sell_pos)
    sell_str = response_str[sell_pos + 7:sell_end]
    print(f'Found SELL prob: {sell_str}')

print('Looking for HOLD probability...')
hold_pos = response_str.find('"HOLD":')
if hold_pos >= 0:
    hold_end = response_str.find(',', hold_pos)
    hold_str = response_str[hold_pos + 7:hold_end]
    print(f'Found HOLD prob: {hold_str}')

print('Looking for BUY probability...')
buy_pos = response_str.find('"BUY":')
if buy_pos >= 0:
    buy_end = response_str.find('}', buy_pos)
    buy_str = response_str[buy_pos + 6:buy_end]
    print(f'Found BUY prob: {buy_str}')
