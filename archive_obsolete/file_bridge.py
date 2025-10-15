"""
Multi-Symbol File Bridge for MT5
Handles multiple symbols simultaneously
"""

import time
from pathlib import Path
from model_rest_server import ModelServer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MT5 Files directory
MT5_FILES = Path.home() / \
    "AppData/Roaming/MetaQuotes/Terminal/776D2ACDFA4F66FAF3C8985F75FA9FF6/MQL5/Files"


class MultiSymbolBridge:
    def __init__(self):
        self.model_server = ModelServer()
        self.last_processed = {}  # Track per-file processing
        logger.info("✅ Multi-symbol file bridge initialized")
        logger.info(f"📁 Watching directory: {MT5_FILES}")

    def run(self):
        """Main loop - watch for request files from multiple symbols"""
        logger.info("🔄 Bridge running - supports multiple symbols")
        logger.info("Waiting for requests from MT5...")

        while True:
            try:
                # Scan for all request files (ai_request_*.csv)
                request_files = list(MT5_FILES.glob("ai_request_*.csv"))

                for request_file in request_files:
                    # Check if file is new or modified
                    try:
                        mtime = request_file.stat().st_mtime
                        file_key = str(request_file)

                        if file_key not in self.last_processed or mtime != self.last_processed[file_key]:
                            self.process_request(request_file)
                            self.last_processed[file_key] = mtime
                    except FileNotFoundError:
                        # File was deleted, skip
                        pass

                time.sleep(0.1)  # Check every 100ms

            except KeyboardInterrupt:
                logger.info("\n🛑 Bridge stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)

    def process_request(self, request_file):
        """Process a single request file"""
        try:
            # Read request
            with open(request_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                return

            # Parse (MT5 CSV uses tabs)
            header = [h.strip() for h in lines[0].split('\t')]
            data = [d.strip() for d in lines[1].split('\t')]
            row_dict = dict(zip(header, data))

            symbol = row_dict.get('symbol', 'UNKNOWN')
            features_str = row_dict.get('features', '')

            if not features_str:
                return

            # Parse features
            features = [float(x.strip())
                        for x in features_str.split(';') if x.strip()]

            # Log features to verify they're different per symbol
            logger.info(f"📥 {symbol}: {len(features)} features")
            logger.info(f"   First 5: {[round(f, 4) for f in features[:5]]}")
            logger.info(f"   Last 5: {[round(f, 4) for f in features[-5:]]}")

            # Get prediction
            result = self.model_server.predict(features)

            # DEBUG: Print raw result
            print(f"DEBUG {symbol}: result = {result}")

            if 'error' in result:
                logger.error(f"❌ {symbol}: {result['error']}")
                return

            # Write response to matching file
            action = result['action']
            confidence = result['confidence']

            response_file = request_file.parent / \
                request_file.name.replace('ai_request', 'ai_response')

            with open(response_file, 'w') as f:
                f.write(f"{action},{confidence}\n")

            logger.info(f"📤 {symbol}: {action} ({confidence:.3f})")

            # Clean up request
            request_file.unlink()

        except Exception as e:
            logger.error(f"Failed to process {request_file.name}: {e}")


if __name__ == "__main__":
    print("="*60)
    print("BLACKICE MULTI-SYMBOL FILE BRIDGE")
    print("="*60)
    print("Supports multiple symbols simultaneously")
    print("Each symbol uses its own request/response files")
    print("="*60)

    bridge = MultiSymbolBridge()
    bridge.run()
