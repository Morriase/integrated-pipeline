#!/usr/bin/env python3
"""
BlackIce AI REST Server V3
V3 = V2 Server (60% accuracy) + EA V3 (Advanced Risk Management)
The "upgrade" is in the EA, not the server.
"""

import sys
sys.path.append('Python')

# Just use the working V2 server
from model_rest_server_v2 import app

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("BLACKICE AI REST SERVER V3")
    print("=" * 70)
    print("\nUsing V2 institutional models (60% accuracy)")
    print("Upgrade: EA V3 with advanced risk management")
    print("  - Dynamic ATR-based position sizing")
    print("  - Trailing stops")
    print("  - Partial profit taking")
    print("  - Breakeven protection")
    print("=" * 70)
    
    app.run(host='127.0.0.1', port=5000, debug=False)
