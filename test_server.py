import requests
import json
import time

# Test data - sample OHLCV data
test_ohlcv = [
    {"time": "2025-10-15 10:00:00", "open": 1.0500, "high": 1.0520,
        "low": 1.0490, "close": 1.0515, "volume": 1000},
    {"time": "2025-10-15 10:01:00", "open": 1.0515, "high": 1.0530,
        "low": 1.0505, "close": 1.0525, "volume": 1200},
    {"time": "2025-10-15 10:02:00", "open": 1.0525, "high": 1.0540,
        "low": 1.0510, "close": 1.0535, "volume": 1100},
    {"time": "2025-10-15 10:03:00", "open": 1.0535, "high": 1.0550,
        "low": 1.0520, "close": 1.0545, "volume": 1300},
    {"time": "2025-10-15 10:04:00", "open": 1.0545, "high": 1.0560,
        "low": 1.0530, "close": 1.0555, "volume": 1400},
    {"time": "2025-10-15 10:05:00", "open": 1.0555, "high": 1.0570,
        "low": 1.0540, "close": 1.0565, "volume": 1500},
    {"time": "2025-10-15 10:06:00", "open": 1.0565, "high": 1.0580,
        "low": 1.0550, "close": 1.0575, "volume": 1600},
    {"time": "2025-10-15 10:07:00", "open": 1.0575, "high": 1.0590,
        "low": 1.0560, "close": 1.0585, "volume": 1700},
    {"time": "2025-10-15 10:08:00", "open": 1.0585, "high": 1.0600,
        "low": 1.0570, "close": 1.0595, "volume": 1800},
    {"time": "2025-10-15 10:09:00", "open": 1.0595, "high": 1.0610,
        "low": 1.0580, "close": 1.0605, "volume": 1900},
    {"time": "2025-10-15 10:10:00", "open": 1.0605, "high": 1.0620,
        "low": 1.0590, "close": 1.0615, "volume": 2000},
    {"time": "2025-10-15 10:11:00", "open": 1.0615, "high": 1.0630,
        "low": 1.0600, "close": 1.0625, "volume": 2100},
    {"time": "2025-10-15 10:12:00", "open": 1.0625, "high": 1.0640,
        "low": 1.0610, "close": 1.0635, "volume": 2200},
    {"time": "2025-10-15 10:13:00", "open": 1.0635, "high": 1.0650,
        "low": 1.0620, "close": 1.0645, "volume": 2300},
    {"time": "2025-10-15 10:14:00", "open": 1.0645, "high": 1.0660,
        "low": 1.0630, "close": 1.0655, "volume": 2400},
    {"time": "2025-10-15 10:15:00", "open": 1.0655, "high": 1.0670,
        "low": 1.0640, "close": 1.0665, "volume": 2500},
    {"time": "2025-10-15 10:16:00", "open": 1.0665, "high": 1.0680,
        "low": 1.0650, "close": 1.0675, "volume": 2600},
    {"time": "2025-10-15 10:17:00", "open": 1.0675, "high": 1.0690,
        "low": 1.0660, "close": 1.0685, "volume": 2700},
    {"time": "2025-10-15 10:18:00", "open": 1.0685, "high": 1.0700,
        "low": 1.0670, "close": 1.0695, "volume": 2800},
    {"time": "2025-10-15 10:19:00", "open": 1.0695, "high": 1.0710,
        "low": 1.0680, "close": 1.0705, "volume": 2900},
    {"time": "2025-10-15 10:20:00", "open": 1.0705, "high": 1.0720,
        "low": 1.0690, "close": 1.0715, "volume": 3000},
    {"time": "2025-10-15 10:21:00", "open": 1.0715, "high": 1.0730,
        "low": 1.0700, "close": 1.0725, "volume": 3100},
    {"time": "2025-10-15 10:22:00", "open": 1.0725, "high": 1.0740,
        "low": 1.0710, "close": 1.0735, "volume": 3200},
    {"time": "2025-10-15 10:23:00", "open": 1.0735, "high": 1.0750,
        "low": 1.0720, "close": 1.0745, "volume": 3300},
    {"time": "2025-10-15 10:24:00", "open": 1.0745, "high": 1.0760,
        "low": 1.0730, "close": 1.0755, "volume": 3400},
    {"time": "2025-10-15 10:25:00", "open": 1.0755, "high": 1.0770,
        "low": 1.0740, "close": 1.0765, "volume": 3500},
    {"time": "2025-10-15 10:26:00", "open": 1.0765, "high": 1.0780,
        "low": 1.0750, "close": 1.0775, "volume": 3600},
    {"time": "2025-10-15 10:27:00", "open": 1.0775, "high": 1.0790,
        "low": 1.0760, "close": 1.0785, "volume": 3700},
    {"time": "2025-10-15 10:28:00", "open": 1.0785, "high": 1.0800,
        "low": 1.0770, "close": 1.0795, "volume": 3800},
    {"time": "2025-10-15 10:29:00", "open": 1.0795, "high": 1.0810,
        "low": 1.0780, "close": 1.0805, "volume": 3900},
    {"time": "2025-10-15 10:30:00", "open": 1.0805, "high": 1.0820,
        "low": 1.0790, "close": 1.0815, "volume": 4000},
    {"time": "2025-10-15 10:30:00", "open": 1.0815, "high": 1.0830,
        "low": 1.0800, "close": 1.0825, "volume": 4100},
    {"time": "2025-10-15 10:31:00", "open": 1.0825, "high": 1.0840,
        "low": 1.0810, "close": 1.0835, "volume": 4200},
    {"time": "2025-10-15 10:32:00", "open": 1.0835, "high": 1.0850,
        "low": 1.0820, "close": 1.0845, "volume": 4300},
    {"time": "2025-10-15 10:33:00", "open": 1.0845, "high": 1.0860,
        "low": 1.0830, "close": 1.0855, "volume": 4400},
    {"time": "2025-10-15 10:34:00", "open": 1.0855, "high": 1.0870,
        "low": 1.0840, "close": 1.0865, "volume": 4500},
    {"time": "2025-10-15 10:35:00", "open": 1.0865, "high": 1.0880,
        "low": 1.0850, "close": 1.0875, "volume": 4600},
    {"time": "2025-10-15 10:36:00", "open": 1.0875, "high": 1.0890,
        "low": 1.0860, "close": 1.0885, "volume": 4700},
    {"time": "2025-10-15 10:37:00", "open": 1.0885, "high": 1.0900,
        "low": 1.0870, "close": 1.0895, "volume": 4800},
    {"time": "2025-10-15 10:38:00", "open": 1.0895, "high": 1.0910,
        "low": 1.0880, "close": 1.0905, "volume": 4900},
    {"time": "2025-10-15 10:39:00", "open": 1.0905, "high": 1.0920,
        "low": 1.0890, "close": 1.0915, "volume": 5000},
    {"time": "2025-10-15 10:40:00", "open": 1.0915, "high": 1.0930,
        "low": 1.0900, "close": 1.0925, "volume": 5100},
    {"time": "2025-10-15 10:41:00", "open": 1.0925, "high": 1.0940,
        "low": 1.0910, "close": 1.0935, "volume": 5200},
    {"time": "2025-10-15 10:42:00", "open": 1.0935, "high": 1.0950,
        "low": 1.0920, "close": 1.0945, "volume": 5300},
    {"time": "2025-10-15 10:43:00", "open": 1.0945, "high": 1.0960,
        "low": 1.0930, "close": 1.0955, "volume": 5400},
    {"time": "2025-10-15 10:44:00", "open": 1.0955, "high": 1.0970,
        "low": 1.0940, "close": 1.0965, "volume": 5500},
    {"time": "2025-10-15 10:45:00", "open": 1.0965, "high": 1.0980,
        "low": 1.0950, "close": 1.0975, "volume": 5600},
    {"time": "2025-10-15 10:46:00", "open": 1.0975, "high": 1.0990,
        "low": 1.0960, "close": 1.0985, "volume": 5700},
    {"time": "2025-10-15 10:47:00", "open": 1.0985, "high": 1.1000,
        "low": 1.0970, "close": 1.0995, "volume": 5800},
    {"time": "2025-10-15 10:48:00", "open": 1.0995, "high": 1.1010,
        "low": 1.0980, "close": 1.1005, "volume": 5900},
    {"time": "2025-10-15 10:49:00", "open": 1.1005, "high": 1.1020,
        "low": 1.0990, "close": 1.1015, "volume": 6000},
]


def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check Successful!")
            print(
                f"Models loaded: PyTorch={data['models_loaded']['pytorch']}, sklearn={data['models_loaded']['sklearn']}, temporal={data['models_loaded']['temporal']}")
            print(f"Total models: {data['total_models']}")
            print(f"Scalers loaded: {data['scalers_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_prediction():
    """Test prediction endpoint"""
    try:
        payload = {
            "ohlcv": test_ohlcv,
            "symbol": "EURUSD",
            "timeframe": "1m"
        }

        response = requests.post(
            "http://localhost:5001/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "success" in data and data["success"]:
                prediction = data["prediction"]
                print("✅ Prediction Successful!")
                print(f"Action: {prediction['action']}")
                print(f"Confidence: {prediction['confidence']:.1%}")
                print(f"Should trade: {prediction['should_trade']}")
                print(f"Models used: {prediction['models_used']}")
                print(f"Consensus: {prediction['consensus']}")
                return True
            else:
                print(f"❌ Prediction failed: {data}")
                return False
        else:
            print(
                f"❌ Prediction request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def test_models_endpoint():
    """Test models endpoint"""
    try:
        response = requests.get("http://localhost:5001/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint successful!")
            print(f"PyTorch models: {data['models']['pytorch']}")
            print(f"sklearn models: {data['models']['sklearn']}")
            print(f"Temporal models: {data['models']['temporal']}")
            print(f"Ensemble weights: {len(data['ensemble_weights'])} models")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Black Ice Intelligence REST Server")
    print("=" * 50)

    # Wait a moment for server to be ready
    time.sleep(2)

    # Test health
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health()

    if health_ok:
        # Test models endpoint
        print("\n2. Testing Models Endpoint...")
        test_models_endpoint()

        # Test prediction
        print("\n3. Testing Prediction Endpoint...")
        test_prediction()

    print("\n" + "=" * 50)
    print("🧪 Testing Complete!")
