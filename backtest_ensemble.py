"""
Backtest Black Ice AI Ensemble
Test the models on historical data to see performance
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class SimpleBacktester:
    def __init__(self, initial_balance=10000, lot_size=0.01, stop_loss_pips=50, take_profit_pips=100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_size = lot_size
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.trades = []
        self.equity_curve = []
        
    def calculate_pips(self, entry_price, exit_price, action):
        """Calculate pips for a trade"""
        if action == "BUY":
            pips = (exit_price - entry_price) * 10000
        else:  # SELL
            pips = (entry_price - exit_price) * 10000
        return pips
    
    def execute_trade(self, action, entry_price, exit_price, confidence):
        """Simulate a trade"""
        if action == "HOLD":
            return
        
        pips = self.calculate_pips(entry_price, exit_price, action)
        
        # Simple P&L calculation (1 pip = $1 for 0.01 lot on EURUSD)
        pip_value = 0.1  # $0.10 per pip for 0.01 lot
        profit = pips * pip_value
        
        self.balance += profit
        
        self.trades.append({
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pips': pips,
            'profit': profit,
            'confidence': confidence,
            'balance': self.balance
        })
        
        self.equity_curve.append(self.balance)
    
    def run_backtest(self, df, predictions):
        """Run backtest on historical data"""
        print("🧪 Starting Backtest...")
        print(f"Initial Balance: ${self.initial_balance}")
        print(f"Lot Size: {self.lot_size}")
        print(f"Stop Loss: {self.stop_loss_pips} pips")
        print(f"Take Profit: {self.take_profit_pips} pips")
        print("="*70)
        
        for i in range(len(predictions) - 1):
            action = predictions[i]['action']
            confidence = predictions[i]['confidence']
            
            if action == "HOLD" or confidence < 0.70:
                continue
            
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + 1]['close']  # Exit on next bar
            
            self.execute_trade(action, entry_price, exit_price, confidence)
        
        self.print_results()
    
    def print_results(self):
        """Print backtest results"""
        if len(self.trades) == 0:
            print("⚠️ No trades executed!")
            print("   AI was too conservative or no clear signals")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pips'] > 0])
        losing_trades = len(trades_df[trades_df['pips'] < 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pips = trades_df['pips'].sum()
        avg_win = trades_df[trades_df['pips'] > 0]['pips'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pips'] < 0]['pips'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pips'] > 0]['pips'].sum() / 
                           trades_df[trades_df['pips'] < 0]['pips'].sum()) if losing_trades > 0 else 0
        
        final_balance = self.balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades}")
        print(f"")
        print(f"Total Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print("="*70)
        
        # Plot equity curve
        if len(self.equity_curve) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve, linewidth=2)
            plt.axhline(y=self.initial_balance, color='gray', linestyle='--', label='Initial Balance')
            plt.title('Black Ice AI - Equity Curve', fontsize=14, fontweight='bold')
            plt.xlabel('Trade Number')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('backtest_equity_curve.png', dpi=150)
            print("\n📈 Equity curve saved: backtest_equity_curve.png")
            plt.show()

def load_models_and_predict(df, model_dir="Model_output"):
    """Load models and generate predictions"""
    print("🤖 Loading models...")
    
    model_dir = Path(model_dir)
    models = {}
    
    # Load sklearn models
    for model_file in model_dir.glob("*_sklearn.pkl"):
        if 'scaler' in model_file.name:
            continue
        try:
            model_name = model_file.stem.replace('_sklearn', '')
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"✅ Loaded: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_file.name}: {e}")
    
    if len(models) == 0:
        print("❌ No models found!")
        return None
    
    # Load scaler
    scaler_path = Path("Python/feature_scaler_29.pkl")
    if not scaler_path.exists():
        print("❌ Feature scaler not found!")
        return None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Loaded scaler: {scaler.n_features_in_} features")
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = []
    
    feature_cols = [col for col in df.columns if col not in ['time', 'close', 'open', 'high', 'low', 'volume', 'target']]
    
    for idx, row in df.iterrows():
        features = row[feature_cols].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Get predictions from all models
        probs_list = []
        for model_name, model in models.items():
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                probs_list.append(probs)
        
        # Ensemble average
        if len(probs_list) > 0:
            avg_probs = np.mean(probs_list, axis=0)
            pred = np.argmax(avg_probs)
            conf = np.max(avg_probs)
            
            action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = action_map[pred]
            
            predictions.append({
                'action': action,
                'confidence': conf,
                'prob_sell': avg_probs[0],
                'prob_hold': avg_probs[1],
                'prob_buy': avg_probs[2]
            })
    
    print(f"✅ Generated {len(predictions)} predictions")
    return predictions

def main():
    """Run backtest"""
    print("🧊 Black Ice AI - Backtesting")
    print("="*70)
    
    # Load historical data
    data_path = "Data/EURUSD_M15_processed.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Run the data processing pipeline first")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} bars of historical data")
    
    # Generate predictions
    predictions = load_models_and_predict(df)
    
    if predictions is None:
        return
    
    # Run backtest
    backtester = SimpleBacktester(
        initial_balance=10000,
        lot_size=0.01,
        stop_loss_pips=50,
        take_profit_pips=100
    )
    
    backtester.run_backtest(df, predictions)
    
    # Analyze signal distribution
    actions = [p['action'] for p in predictions]
    print("\n📊 Signal Distribution:")
    print(f"   SELL: {actions.count('SELL')} ({actions.count('SELL')/len(actions)*100:.1f}%)")
    print(f"   HOLD: {actions.count('HOLD')} ({actions.count('HOLD')/len(actions)*100:.1f}%)")
    print(f"   BUY:  {actions.count('BUY')} ({actions.count('BUY')/len(actions)*100:.1f}%)")

if __name__ == "__main__":
    main()
