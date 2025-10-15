"""
Recovery Mechanism Module
Implements capital protection and recovery strategies for the Black Ice system
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: str
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    exit_price: float
    volume: float
    profit_loss: float
    confidence: float
    signal_strength: float
    recovery_mode: bool = False

class RecoveryManager:
    """
    Manages recovery mechanisms and capital protection strategies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        self.config = {
            'recovery_threshold': -0.02,  # -2% equity loss triggers recovery
            'recovery_max_trades': 10,    # Max trades in recovery mode
            'recovery_max_duration': 24,  # Max hours in recovery mode
            'recovery_confidence_threshold': 0.9,  # Higher confidence required
            'recovery_position_multiplier': 0.5,   # Reduce position size
            'normal_confidence_threshold': 0.7,    # Normal confidence threshold
            'trade_history_file': 'Model_output/trade_history.csv',
            'recovery_log_file': 'Model_output/recovery_log.json'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize state
        self.recovery_mode = False
        self.recovery_start_time = None
        self.recovery_trades_count = 0
        self.trade_history = []
        
        # Ensure directories exist
        Path(self.config['trade_history_file']).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config['recovery_log_file']).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing trade history
        self._load_trade_history()
    
    def _load_trade_history(self):
        """Load existing trade history from file"""
        try:
            if Path(self.config['trade_history_file']).exists():
                df = pd.read_csv(self.config['trade_history_file'])
                self.trade_history = df.to_dict('records')
                print(f"✅ Loaded {len(self.trade_history)} historical trades")
            else:
                print("📝 No existing trade history found, starting fresh")
        except Exception as e:
            print(f"❌ Failed to load trade history: {e}")
            self.trade_history = []
    
    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade
        
        Args:
            trade: TradeRecord instance
        """
        # Add to memory
        trade_dict = {
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'action': trade.action,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'volume': trade.volume,
            'profit_loss': trade.profit_loss,
            'confidence': trade.confidence,
            'signal_strength': trade.signal_strength,
            'recovery_mode': trade.recovery_mode
        }
        
        self.trade_history.append(trade_dict)
        
        # Save to file
        self._save_trade_history()
        
        # Check if recovery mode should be triggered
        self._check_recovery_trigger()
        
        print(f"📊 Trade recorded: {trade.action} {trade.symbol} P/L: {trade.profit_loss:.2f}")
    
    def _save_trade_history(self):
        """Save trade history to CSV file"""
        try:
            df = pd.DataFrame(self.trade_history)
            df.to_csv(self.config['trade_history_file'], index=False)
        except Exception as e:
            print(f"❌ Failed to save trade history: {e}")
    
    def _check_recovery_trigger(self):
        """Check if recovery mode should be triggered based on recent losses"""
        if len(self.trade_history) < 5:  # Need minimum trades to assess
            return
        
        # Calculate recent performance (last 10 trades or 24 hours)
        recent_trades = self._get_recent_trades(hours=24, max_trades=10)
        
        if not recent_trades:
            return
        
        # Calculate net P/L
        total_pnl = sum(trade['profit_loss'] for trade in recent_trades)
        
        # Calculate equity percentage (assuming starting equity, this should be configurable)
        equity_change_pct = total_pnl / 10000  # Assuming $10,000 starting equity
        
        if equity_change_pct <= self.config['recovery_threshold'] and not self.recovery_mode:
            self._activate_recovery_mode(equity_change_pct, recent_trades)
        elif self.recovery_mode:
            self._check_recovery_exit(equity_change_pct, recent_trades)
    
    def _get_recent_trades(self, hours: int = 24, max_trades: int = 10) -> List[Dict]:
        """Get recent trades within specified time window"""
        if not self.trade_history:
            return []
        
        # Get recent trades by time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_trades = []
        for trade in reversed(self.trade_history):  # Start from most recent
            try:
                trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                if trade_time >= cutoff_time:
                    recent_trades.append(trade)
                else:
                    break  # Trades are in chronological order
            except:
                continue
        
        # Limit to max_trades
        return recent_trades[:max_trades]
    
    def _activate_recovery_mode(self, equity_change_pct: float, recent_trades: List[Dict]):
        """Activate recovery mode"""
        self.recovery_mode = True
        self.recovery_start_time = datetime.now(timezone.utc)
        self.recovery_trades_count = 0
        
        recovery_log = {
            'event': 'recovery_activated',
            'timestamp': self.recovery_start_time.isoformat(),
            'trigger_equity_change': equity_change_pct,
            'recent_trades_count': len(recent_trades),
            'total_recent_pnl': sum(trade['profit_loss'] for trade in recent_trades)
        }
        
        self._log_recovery_event(recovery_log)
        
        print(f"🚨 RECOVERY MODE ACTIVATED - Equity change: {equity_change_pct:.2%}")
        print(f"   Recent P/L: ${sum(trade['profit_loss'] for trade in recent_trades):.2f}")
    
    def _check_recovery_exit(self, equity_change_pct: float, recent_trades: List[Dict]):
        """Check if recovery mode should be exited"""
        should_exit = False
        exit_reason = ""
        
        # Check if losses have been recovered
        if equity_change_pct >= 0:
            should_exit = True
            exit_reason = "losses_recovered"
        
        # Check max trades limit
        elif self.recovery_trades_count >= self.config['recovery_max_trades']:
            should_exit = True
            exit_reason = "max_trades_reached"
        
        # Check max duration
        elif self.recovery_start_time:
            duration_hours = (datetime.now(timezone.utc) - self.recovery_start_time).total_seconds() / 3600
            if duration_hours >= self.config['recovery_max_duration']:
                should_exit = True
                exit_reason = "max_duration_reached"
        
        if should_exit:
            self._deactivate_recovery_mode(exit_reason, equity_change_pct)
    
    def _deactivate_recovery_mode(self, reason: str, equity_change_pct: float):
        """Deactivate recovery mode"""
        recovery_log = {
            'event': 'recovery_deactivated',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': reason,
            'final_equity_change': equity_change_pct,
            'recovery_trades_executed': self.recovery_trades_count,
            'recovery_duration_hours': (datetime.now(timezone.utc) - self.recovery_start_time).total_seconds() / 3600 if self.recovery_start_time else 0
        }
        
        self._log_recovery_event(recovery_log)
        
        self.recovery_mode = False
        self.recovery_start_time = None
        self.recovery_trades_count = 0
        
        print(f"✅ RECOVERY MODE DEACTIVATED - Reason: {reason}")
        print(f"   Final equity change: {equity_change_pct:.2%}")
    
    def _log_recovery_event(self, event: Dict):
        """Log recovery events to file"""
        try:
            # Load existing log
            log_file = Path(self.config['recovery_log_file'])
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'events': []}
            
            # Add new event
            log_data['events'].append(event)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to log recovery event: {e}")
    
    def should_accept_signal(self, signal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a trading signal should be accepted based on recovery mode
        
        Args:
            signal: Trading signal dictionary with confidence, action, etc.
        
        Returns:
            Tuple of (should_accept, modified_signal)
        """
        if not self.recovery_mode:
            # Normal mode - use standard confidence threshold
            should_accept = signal.get('confidence', 0) >= self.config['normal_confidence_threshold']
            return should_accept, signal
        
        # Recovery mode - stricter criteria
        confidence = signal.get('confidence', 0)
        signal_strength = signal.get('signal_strength', 0)
        
        # Higher confidence threshold in recovery mode
        confidence_ok = confidence >= self.config['recovery_confidence_threshold']
        
        # Only accept strong signals
        strength_ok = signal_strength >= 0.8
        
        # Avoid SELL signals in recovery mode (focus on recovery)
        action_ok = signal.get('action') != 'SELL'
        
        should_accept = confidence_ok and strength_ok and action_ok
        
        # Modify signal for recovery mode
        modified_signal = signal.copy()
        if should_accept:
            # Reduce position size
            modified_signal['position_multiplier'] = self.config['recovery_position_multiplier']
            modified_signal['recovery_mode'] = True
            
            # Increment recovery trades counter
            self.recovery_trades_count += 1
        
        return should_accept, modified_signal
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get current recovery status and statistics
        
        Returns:
            Recovery status dictionary
        """
        recent_trades = self._get_recent_trades(hours=24, max_trades=10)
        recent_pnl = sum(trade['profit_loss'] for trade in recent_trades) if recent_trades else 0
        
        status = {
            'recovery_mode_active': self.recovery_mode,
            'total_trades': len(self.trade_history),
            'recent_trades_24h': len(recent_trades),
            'recent_pnl_24h': recent_pnl,
            'recovery_trades_count': self.recovery_trades_count if self.recovery_mode else 0,
            'recovery_duration_hours': (datetime.now(timezone.utc) - self.recovery_start_time).total_seconds() / 3600 if self.recovery_start_time else 0
        }
        
        if recent_trades:
            status['win_rate_24h'] = sum(1 for t in recent_trades if t['profit_loss'] > 0) / len(recent_trades)
            status['avg_trade_pnl_24h'] = recent_pnl / len(recent_trades)
        
        return status
    
    def generate_recovery_report(self) -> str:
        """
        Generate a comprehensive recovery report
        
        Returns:
            Path to generated report
        """
        report_path = Path(self.config['recovery_log_file']).parent / 'recovery_report.txt'
        
        status = self.get_recovery_status()
        recent_trades = self._get_recent_trades(hours=168, max_trades=50)  # Last week
        
        with open(report_path, 'w') as f:
            f.write("=== BLACK ICE RECOVERY SYSTEM REPORT ===\n\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            
            f.write("--- Current Status ---\n")
            f.write(f"Recovery Mode Active: {status['recovery_mode_active']}\n")
            f.write(f"Total Trades: {status['total_trades']}\n")
            f.write(f"Recent 24h Trades: {status['recent_trades_24h']}\n")
            f.write(f"Recent 24h P/L: ${status['recent_pnl_24h']:.2f}\n")
            
            if 'win_rate_24h' in status:
                f.write(f"24h Win Rate: {status['win_rate_24h']:.1%}\n")
                f.write(f"24h Avg Trade P/L: ${status['avg_trade_pnl_24h']:.2f}\n")
            
            if status['recovery_mode_active']:
                f.write(f"Recovery Trades Executed: {status['recovery_trades_count']}\n")
                f.write(f"Recovery Duration: {status['recovery_duration_hours']:.1f} hours\n")
            
            f.write("\n--- Recent Trades (Last 7 Days) ---\n")
            for trade in recent_trades[-20:]:  # Last 20 trades
                f.write(f"{trade['timestamp'][:19]} | {trade['action']} {trade['symbol']} | P/L: ${trade['profit_loss']:.2f} | Conf: {trade['confidence']:.2f}\n")
            
            f.write(f"\n--- Configuration ---\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📊 Recovery report generated: {report_path}")
        return str(report_path)


# Integration functions
def integrate_recovery_mechanism(system, signal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Integrate recovery mechanism with the main trading system
    
    Args:
        system: IntegratedSMCSystem instance
        signal: Trading signal to evaluate
    
    Returns:
        Tuple of (should_trade, modified_signal)
    """
    if not hasattr(system, 'recovery_manager'):
        system.recovery_manager = RecoveryManager()
    
    return system.recovery_manager.should_accept_signal(signal)


def create_sample_trade_history():
    """Create sample trade history for testing"""
    recovery_manager = RecoveryManager()
    
    # Simulate some trades
    sample_trades = [
        TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol="EURUSD",
            action="BUY",
            entry_price=1.0850,
            exit_price=1.0870,
            volume=0.1,
            profit_loss=20.0,
            confidence=0.85,
            signal_strength=0.9
        ),
        TradeRecord(
            timestamp=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            symbol="GBPUSD",
            action="SELL",
            entry_price=1.2650,
            exit_price=1.2620,
            volume=0.1,
            profit_loss=30.0,
            confidence=0.78,
            signal_strength=0.82
        )
    ]
    
    for trade in sample_trades:
        recovery_manager.record_trade(trade)
    
    return recovery_manager


if __name__ == "__main__":
    # Test the recovery mechanism
    print("=== Testing Recovery Mechanism ===")
    
    recovery_manager = create_sample_trade_history()
    status = recovery_manager.get_recovery_status()
    
    print(f"Recovery Status: {status}")
    
    # Test signal evaluation
    test_signal = {
        'action': 'BUY',
        'confidence': 0.75,
        'signal_strength': 0.85
    }
    
    should_accept, modified_signal = recovery_manager.should_accept_signal(test_signal)
    print(f"Should accept signal: {should_accept}")
    print(f"Modified signal: {modified_signal}")
    
    # Generate report
    recovery_manager.generate_recovery_report()