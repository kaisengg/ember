"""
Brent Crude Oil Trading System
==============================

A complete machine learning system for predicting BUY/SELL signals for Brent crude oil

"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')
sys.stdout.flush()

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Technical Analysis
import talib




class BrentTradingSystem:
    """
    
    WORKFLOW:
    ---------
    1. DATA EXTRACTION
       - Downloads Brent (BZ=F) and WTI (CL=F) crude oil prices from Yahoo Finance
       - Includes High, Low, Close, Volume for both instruments
       - Period: July 2020 to August 2025
    
    2. FEATURE ENGINEERING
       - Creates 50+ technical indicators: RSI, MACD, ADX, SMA, EMA, ATR
       - Lagged returns (1-10 days)
       - Volatility measures (10, 20, 30, 60 days)
       - Cross-market features (Brent-WTI spread, correlation)
       - Volume indicators (changes, ratios, moving averages)
    
    3. LABEL CREATION
       - Forward-looking labels based on 5-day returns
       - BUY if forward return > 5%, SELL if < -5%, else HOLD
       - Only BUY/SELL labels used for binary classification
    
    4. FEATURE SELECTION
       - Recursive Feature Elimination (RFE) with Random Forest
       - Selects top 15 most predictive features
    
    5. MODEL TRAINING
       - XGBoost Classifier 
       - Handles class imbalance with scale_pos_weight
       - Training period: July 2020 - July 2025
    
    6. SIGNAL GENERATION
       - Predicts P(Buy) for each day
       - Converts to conviction: |P(Buy) - 0.5| × 2
       - Filters signals with conviction < 30%  
    
    7. POSITION SIZING
       - Base allocation: 15% of equity
       - Scaled by conviction (1.0x to 2.5x multiplier)
       - Max position: 60% of equity
    
    8. BACKTESTING
       - Simulates actual trading with position management
       - Tracks equity curve, trades, P&L
       - Calculates performance metrics (Sharpe, Calmar, max drawdown)
    
    9. VISUALIZATION
       - Plots actual labels vs predicted signals
       - Equity curve with profit/loss regions
       - Performance metrics summary
       - Marks low conviction (filtered) signals
    
    PERFORMANCE METRICS:
    --------------------
    - Annualized Return, Volatility, Sharpe Ratio
    - Maximum Drawdown, Calmar Ratio
    - Win Rate, Average Win/Loss, Profit Factor
    - Trade statistics and CSV export
    """
    
    def __init__(self, initial_capital=10_000_000):
        """
        Initialize the trading system.
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in USD
        """
        self.initial_capital = initial_capital
        self.data = None
        self.features = None
        self.labels = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def extract_data(self, start_date="2020-07-01", end_date='2025-08-31'):
        """
        Extract Brent and WTI crude oil data from Yahoo Finance.
        
        Downloads OHLCV (Open, High, Low, Close, Volume) data for both Brent (BZ=F)
        and WTI (CL=F) crude oil futures contracts.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame: OHLCV data with columns like 'Brent_Close', 'WTI_High', etc.
        """
        try:
            tickers = ["BZ=F", "CL=F"]
            fields  = ["Close", "High", "Low", "Volume"]
            label   = {"BZ=F": "Brent", "CL=F": "WTI"}
            
            print("Extracting data from Yahoo Finance...")
            # Download data for both Brent and WTI
            df = yf.download(tickers, start=start_date, end=end_date, progress=False)[fields]
            
            # Ensure datetime index
            df.index = pd.to_datetime(df.index, errors="coerce")
            df.index.name = "Date"
            
            # Restructure MultiIndex columns: (Field, Ticker) → (Ticker, Field)
            df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
            # Rename columns to readable format: 'Brent_Close', 'WTI_High', etc.
            df.columns = [f"{label[t]}_{f}" for (t, f) in df.columns]
            
            # Reorder columns for consistency
            ordered = ["Brent_Close","Brent_High","Brent_Low","Brent_Volume",
                      "WTI_Close","WTI_High","WTI_Low","WTI_Volume"]
            df = df[ordered]
            df = df.dropna(how="all")  # Remove completely empty rows
            
            # Create price columns for convenience
            df['Brent_Price'] = df['Brent_Close']
            df['WTI_Price'] = df['WTI_Close']
            
            print(f"✓ Data extracted: {df.index.min().date()} to {df.index.max().date()} ({len(df)} observations)")
            
            self.data = df
            return df
        except Exception as e:
            print(f"✗ Error extracting data: {e}")
            import traceback
            traceback.print_exc()
            return None



    
    def create_labels(self, forward_days=5, buy_threshold=0.05, sell_threshold=-0.05):
        """
        Create gradient-based labels using forward returns.
        
        Labels are based on future price movement:
        - BUY (1): If forward return >= +5% (expect price to rise)
        - SELL (0): If forward return <= -5% (expect price to fall)
        - HOLD: If forward return between -5% and +5% (uncertain, excluded from training)
        
        Parameters:
        -----------
        forward_days : int, default=5
            Number of days to look forward for returns
        buy_threshold : float, default=0.05
            Minimum forward return (5%) to signal BUY
        sell_threshold : float, default=-0.05
            Maximum forward return (-5%) to signal SELL
            
        Sets:
        -----
        self.labels : pd.Series
            Binary labels (1=BUY, 0=SELL) with HOLD periods dropped
        self.forward_returns : pd.Series
            Raw forward returns used for label creation
        """
        print("Creating labels...")
        
        # Calculate forward log returns (what happens in next N days)
        forward_returns = np.log(self.data['Brent_Price'].shift(-forward_days) / self.data['Brent_Price'])
        
        # Create categorical labels based on thresholds
        labels = pd.Series(index=self.data.index, dtype='object')
        labels[forward_returns >= buy_threshold] = 'BUY'   # Strong upward movement
        labels[forward_returns <= sell_threshold] = 'SELL'  # Strong downward movement
        labels[(forward_returns > sell_threshold) & (forward_returns < buy_threshold)] = 'HOLD'  # Uncertain
        
        # Convert to binary classification (exclude HOLD for training)
        binary_labels = labels.copy()
        binary_labels[labels == 'BUY'] = 1   # Positive class
        binary_labels[labels == 'SELL'] = 0  # Negative class
        binary_labels[labels == 'HOLD'] = np.nan  # Excluded from training
        
        # Store labels and ensure integer type for classification
        self.labels = binary_labels.dropna().astype(int)
        self.forward_returns = forward_returns
        
        print(f"✓ Labels created: {(labels == 'BUY').sum()} BUY, {(labels == 'SELL').sum()} SELL, {(labels == 'HOLD').sum()} HOLD")
    
    def create_features(self):
        """
        Create comprehensive feature set for ML model.
        
        Features include:
        1. Lagged Returns: 1-10 day returns for both Brent and WTI
        2. Volume Features: Changes, moving averages, ratios
        3. Price Range Features: High-low range, close position in range
        4. Technical Indicators:
           - Momentum: RSI (14)
           - Trend: MACD, ADX (14), SMA (10,20,50,100,200), EMA (12,26)
           - Volatility: ATR (14), Bollinger Bands (20,2)
        5. Volatility Measures: Rolling std (10,20,30,60 days)
        6. Cross-Market Features: Brent-WTI spread, rolling correlation
        
        """
        print("Creating features...")
        
        features_df = pd.DataFrame(index=self.data.index)
        
        # 1. Lagged returns (capture momentum and mean reversion patterns)
        for lag in range(1, 11):
            features_df[f'Brent_Return_Lag{lag}'] = self.data['Brent_Price'].pct_change(lag)
            features_df[f'WTI_Return_Lag{lag}'] = self.data['WTI_Price'].pct_change(lag)
        
        # 1b. Volume and High/Low features
        # Volume changes
        features_df['Brent_Volume_Change'] = self.data['Brent_Volume'].pct_change()
        features_df['WTI_Volume_Change'] = self.data['WTI_Volume'].pct_change()
        
        # High-Low range 
        features_df['Brent_HL_Range'] = (self.data['Brent_High'] - self.data['Brent_Low']) / self.data['Brent_Close']
        features_df['WTI_HL_Range'] = (self.data['WTI_High'] - self.data['WTI_Low']) / self.data['WTI_Close']
        
        # Price position in daily range
        features_df['Brent_Close_Position'] = (self.data['Brent_Close'] - self.data['Brent_Low']) / (self.data['Brent_High'] - self.data['Brent_Low'] + 1e-6)
        features_df['WTI_Close_Position'] = (self.data['WTI_Close'] - self.data['WTI_Low']) / (self.data['WTI_High'] - self.data['WTI_Low'] + 1e-6)
        
        # Rolling volume averages
        features_df['Brent_Volume_MA10'] = self.data['Brent_Volume'].rolling(10).mean()
        features_df['Brent_Volume_MA20'] = self.data['Brent_Volume'].rolling(20).mean()
        features_df['Brent_Volume_Ratio'] = self.data['Brent_Volume'] / (features_df['Brent_Volume_MA20'] + 1)
        
        # 2. Technical indicators on Brent (using actual High/Low/Close data)
        brent_close = self.data['Brent_Close'].values
        brent_high = self.data['Brent_High'].values
        brent_low = self.data['Brent_Low'].values
        
        # RSI
        features_df['RSI_14'] = talib.RSI(brent_close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(brent_close, fastperiod=12, slowperiod=26, signalperiod=9)
        features_df['MACD'] = macd
        features_df['MACD_Signal'] = macd_signal
        features_df['MACD_Histogram'] = macd_hist
        
        # ADX (using actual high, low, close)
        features_df['ADX_14'] = talib.ADX(brent_high, brent_low, brent_close, timeperiod=14)
        
        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            features_df[f'SMA_{period}'] = talib.SMA(brent_close, timeperiod=period)
        
        for period in [12, 26]:
            features_df[f'EMA_{period}'] = talib.EMA(brent_close, timeperiod=period)
        
        # Moving average ratios and slopes
        features_df['SMA_Ratio'] = (features_df['SMA_50'] - features_df['SMA_200']) / features_df['SMA_200']
        features_df['SMA50_Slope'] = features_df['SMA_50'].diff(10)
        
        # 3. Volatility measures
        features_df['Volatility_10d'] = self.data['Brent_Price'].rolling(10).std()
        features_df['Volatility_20d'] = self.data['Brent_Price'].rolling(20).std()
        
        # ATR (using actual high, low, close)
        features_df['ATR_14'] = talib.ATR(brent_high, brent_low, brent_close, timeperiod=14)
        
        # 4. Cross features
        # Brent-WTI spread
        features_df['Brent_WTI_Spread'] = self.data['Brent_Price'] - self.data['WTI_Price']
        features_df['Spread_Change_5d'] = features_df['Brent_WTI_Spread'].diff(5)
        
        # Rolling correlation
        brent_returns = self.data['Brent_Price'].pct_change()
        wti_returns = self.data['WTI_Price'].pct_change()
        features_df['Correlation_20d'] = brent_returns.rolling(20).corr(wti_returns)
        
        # Clean features - handle inf and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.ffill().bfill()
        
        # Drop warmup rows (first 30 rows to ensure all indicators are calculated)
        features_df = features_df.iloc[200:]
        
        # Final check for any remaining inf or nan
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        self.features = features_df
        self.feature_names = features_df.columns.tolist()
        
        print(f"✓ Features created: {len(self.feature_names)} total features")
    
    def select_features(self, n_features=15):
        """
        Apply Recursive Feature Elimination to select most predictive features.
        
        Uses Random Forest classifier with RFE to iteratively remove weakest features
        and select the top N most predictive features for the model.
        
        Parameters:
        -----------
        n_features : int, default=15
            Number of features to select (from ~50+ total features)
            
 
        """
        print(f"Selecting top {n_features} features using RFE...")
        
        # Prepare training data for feature selection (July 2020 - July 2025)
        train_start = '2020-07-01'
        train_end = '2025-07-31'
        
        # Align features and labels (only use dates where both exist)
        train_features = self.features.loc[
            (self.features.index >= train_start) & 
            (self.features.index <= train_end)
        ]
        
        # Match features with labels (inner join on dates)
        train_labels = self.labels.loc[train_features.index.intersection(self.labels.index)]
        train_features = train_features.loc[train_labels.index]
        
        # Clean: Remove any rows with NaN in features or labels
        valid_mask = ~(train_features.isna().any(axis=1) | train_labels.isna())
        train_features = train_features[valid_mask]
        train_labels = train_labels[valid_mask]
        
        # Initialize Random Forest for feature importance ranking
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=n_features)
        
        # Fit RFE: Iteratively removes least important features
        rfe.fit(train_features, train_labels)
        
        # Extract selected feature names and update dataset
        selected_features = train_features.columns[rfe.support_].tolist()
        self.features = self.features[selected_features]  # Keep only selected
        self.feature_names = selected_features
        
        # Get feature importances from the final RF model
        self.rfe_feature_importances = pd.Series(
            rfe.estimator_.feature_importances_,
            index=selected_features
        ).sort_values(ascending=False)
        
        print(f"✓ Feature selection completed: {len(selected_features)} features selected")
    
    def plot_feature_importance(self):
        """
        Plot feature importance from RFE selection.
        
        Creates a horizontal bar chart showing the importance of all selected features
        from the Random Forest model used in RFE.
        """
        if not hasattr(self, 'rfe_feature_importances'):
            print("⚠️ No feature importances available. Run select_features() first.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Sort features by importance (ascending for horizontal bar chart)
        sorted_importances = self.rfe_feature_importances.sort_values(ascending=True)
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_importances)))
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(sorted_importances)), sorted_importances.values, 
                       color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_importances.index, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Feature Importance (RFE Selection)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_importances.values)):
            ax.text(value, i, f' {value:.4f}', 
                   va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'feature_importance_rfe.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved: {plot_filename}")
    
    def train_model(self):
        """
        Train XGBoost classifier with time series cross-validation.
        
        Training process:
        1. Prepares training data (July 2020 - July 2025)
        2. Scales features using StandardScaler
        3. Handles class imbalance with scale_pos_weight
        4. Performs 5-fold time series cross-validation
        5. Trains final model on full training set
        
        Model hyperparameters:
        - n_estimators: 100
        - max_depth: 6
        - learning_rate: 0.1
        - scale_pos_weight: Ratio of SELL/BUY samples
        
        Sets:
        -----
        self.model : xgb.XGBClassifier
            Trained model ready for prediction
        self.scaler : StandardScaler
            Fitted scaler for feature normalization
        """
        print("Training XGBoost model...")
        
        # Prepare training data (July 2020 - July 2025)
        train_start = '2020-07-01'
        train_end = '2025-07-31'
        
        # Align features and labels using inner join
        train_features = self.features.loc[
            (self.features.index >= train_start) & 
            (self.features.index <= train_end)
        ]
        
        # Get labels that match feature indices
        train_labels = self.labels.loc[train_features.index.intersection(self.labels.index)]
        train_features = train_features.loc[train_labels.index]
        
        # Remove any remaining NaN values
        valid_mask = ~(train_features.isna().any(axis=1) | train_labels.isna())
        train_features = train_features[valid_mask]
        train_labels = train_labels[valid_mask]
        
        # Scale features to mean=0, std=1 (improves model convergence)
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Handle class imbalance: Weight minority class higher
        n_buy = (train_labels == 1).sum()
        n_sell = (train_labels == 0).sum()
        scale_pos_weight = n_sell / n_buy if n_buy > 0 else 1
        
        # Time series cross-validation: Respect temporal order
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(train_features_scaled):
            X_train, X_val = train_features_scaled[train_idx], train_features_scaled[val_idx]
            y_train, y_val = train_labels.iloc[train_idx], train_labels.iloc[val_idx]
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_pred)
            cv_scores.append(val_score)
        
        # Train final model on all training data
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        self.model.fit(train_features_scaled, train_labels)
        
        print(f"✓ Model training completed: Mean CV AUC = {np.mean(cv_scores):.4f}")
    
    def generate_signals(self, period='validation'):
        """
        Generate trading signals using the trained XGBoost model.
        
        CONVICTION CALCULATION:
        -----------------------
        1. Model predicts P(Buy) for each day (probability from 0 to 1)
        2. Convert to signal_raw = P(Buy) - P(Sell) = P(Buy) - (1 - P(Buy))
           - signal_raw = 2*P(Buy) - 1, ranges from -1 (strong SELL) to +1 (strong BUY)
        3. Conviction = |signal_raw| = absolute confidence in the signal
           - Example: P(Buy) = 0.9 → signal_raw = 0.8 → Conviction = 0.8 (80%)
           - Example: P(Buy) = 0.1 → signal_raw = -0.8 → Conviction = 0.8 (80%)
           - Example: P(Buy) = 0.5 → signal_raw = 0.0 → Conviction = 0.0 (uncertain)
        
        Parameters
        ----------
        period : str
            'training' (2020-07-01 to 2025-07-31) or 'validation' (2025-08-01 to 2025-08-31)
            
        Returns
        -------
        pd.DataFrame
            Signals with columns:
            - p_buy: Probability of BUY (0 to 1)
            - signal_raw: Direction * conviction (-1 to +1)
            - direction: 1 for BUY, -1 for SELL
            - flat_signal: True if conviction < 1% (too uncertain to trade)
        """
        if period == 'training':
            start = '2020-07-01'
            end = '2025-07-31'
        else:
            start = '2025-08-01'
            end = '2025-08-31'
        
        period_mask = (self.features.index >= start) & (self.features.index <= end)
        period_features = self.features[period_mask]
        
        if period_features.empty:
            print(f"No {period} data available")
            return None
        
        # Scale features
        period_features_scaled = self.scaler.transform(period_features)
        
        # Generate predictions
        # Step 1: Get probability that price will go UP (BUY signal)
        p_buy = self.model.predict_proba(period_features_scaled)[:, 1]
        
        # Step 2: Convert to directional signal with confidence
        # signal_raw = P(Buy) - P(Sell) = P(Buy) - (1 - P(Buy)) = 2*P(Buy) - 1
        signal_raw = p_buy - (1 - p_buy)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'p_buy': p_buy,
            'signal_raw': signal_raw,
            'direction': np.where(p_buy >= 0.5, 1, -1),
            'flat_signal': np.abs(signal_raw) < 0.01
        }, index=period_features.index)
        
        # Add price data
        signals_df['Brent_Price'] = self.data.loc[period_features.index, 'Brent_Price']
        signals_df['WTI_Price'] = self.data.loc[period_features.index, 'WTI_Price']
        
        if period == 'training':
            self.train_signals = signals_df
        else:
            self.test_signals = signals_df
        
        print(f"✓ Signals generated for {period} period: {len(signals_df)} days")
        
        return signals_df
    
    def backtest_strategy(self, signals_df, period_name='validation'):
        """
        Backtest the trading strategy with dynamic position sizing.
        
        PROFIT-TAKING LOGIC:
        --------------------
        The system uses PYRAMIDING with SIGNAL REVERSAL for profit-taking:
        
        1. SIGNAL REVERSAL (Direction Changes)
           - When signal changes direction (BUY→SELL or SELL→BUY):
             a) CLOSE ALL existing positions → REALIZE P&L
             b) OPEN NEW position in opposite direction based on new conviction
           - Example: LONG position → SELL signal → Close LONG (realize), Open SHORT
           - This locks in profits/losses when the model changes its view
        
        2. SIGNAL CONFIRMATION (Same Direction)
           - When signal stays same direction but conviction changes:
             → ADD ON to existing position (pyramid/scale in)
             → Do NOT close existing position
             → Increase position size based on new conviction level
           - Example: LONG 40k units → LONG signal again → Add 20k more units (total 60k)
           - Unrealized P&L continues to accumulate on full position
        
        3. POSITION MANAGEMENT SCENARIOS
           - LONG → LONG (higher conviction): ADD ON, increase to larger position
           - LONG → LONG (lower conviction): ADD ON, but add fewer units (or reduce)
           - LONG → SHORT: CLOSE LONG (realize P&L), OPEN SHORT (new position)
           - SHORT → LONG: CLOSE SHORT (realize P&L), OPEN LONG (new position)
           - LONG/SHORT → Low conviction (<30%): HOLD position (no action, maintain state)
        
       
        -------------------------
        Step 1: CONVICTION FILTERING
            - Only trade if conviction >= 30% (min_conviction = 0.30)
            - Conviction < 30% → Skip trade (no action, hold existing position)
        
        Step 2: POSITION SIZING FORMULA (for conviction >= 30%)
            a) Base Size = 15% of current equity
            b) Normalized Conviction = (conviction - 0.30) / (1.0 - 0.30)
               - Maps conviction from [0.30, 1.0] to [0.0, 1.0]
            c) Conviction Multiplier = 1.0 + (Normalized Conviction × 1.5)
               - Ranges from 1.0 (at 30% conviction) to 2.5 (at 100% conviction)
            d) Position Fraction = Direction × Base × Multiplier
               - Direction: +1 for BUY, -1 for SELL (SHORT)
               - Capped at ±60% of equity
            e) Target Notional = Position Fraction × Current Equity
            f) Target Units = Target Notional / Current Price
        
        EXAMPLES:
        ---------
        Scenario 1: High Conviction BUY
            - Equity: $10,000,000
            - Conviction: 0.80 (80%)
            - Direction: +1 (BUY)
            - Normalized: (0.80 - 0.30) / 0.70 = 0.714
            - Multiplier: 1.0 + (0.714 × 1.5) = 2.07
            - Position: +1 × 0.15 × 2.07 = +0.311 (31.1% of equity)
            - Notional: 0.311 × $10M = $3,110,000
            - If price = $70, Units = 44,428
        
        Scenario 2: Low Conviction SELL
            - Conviction: 0.35 (35%)
            - Direction: -1 (SELL/SHORT)
            - Normalized: (0.35 - 0.30) / 0.70 = 0.071
            - Multiplier: 1.0 + (0.071 × 1.5) = 1.11
            - Position: -1 × 0.15 × 1.11 = -0.167 (16.7% short)
            - Notional: -$1,670,000
        
        Scenario 3: Below Threshold
            - Conviction: 0.25 (25%)
            - Result: No trade (filtered out)
g
            
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            - trades_df: All trade records (opens and closes)
            - equity_curve_df: Daily equity values
        """
        if signals_df is None:
            print("No signals available for backtesting")
            return None
        
        # Initialize portfolio
        equity = self.initial_capital
        position = 0  # Current position in units
        position_notional = 0  # Current position value
        entry_price = 0  # Track entry price for unrealized P&L
        current_direction = 0  # Track current position direction: 0=flat, 1=long, -1=short
        trades = []
        equity_curve = []
        
        # Track realized and unrealized P&L
        total_realized_pnl = 0  # Cumulative profit from closed trades
        current_unrealized_pnl = 0  # Mark-to-market on open position
        
        # Track conviction filtering
        low_conviction_count = 0
        high_conviction_count = 0
        
        for idx, (date, row) in enumerate(signals_df.iterrows()):
            current_price = row['Brent_Price']
            direction = row['direction']
            flat_signal = row['flat_signal']
            conviction = abs(row['signal_raw'])

            if not flat_signal:
                # STEP 1: CONVICTION FILTERING (30% minimum threshold)
                # Only trade if we have at least 30% confidence in the signal
                min_conviction = 0.30
                if conviction >= min_conviction:
                    high_conviction_count += 1
                    
                    # STEP 2a: BASE POSITION SIZE
                    # Start with 15% of current equity as base allocation
                    base_size = 0.15  # 15% of equity
                    
                    # STEP 2b: NORMALIZE CONVICTION
                    # Map conviction from [0.30, 1.0] range to [0.0, 1.0] range
                    # Formula: (actual - min) / (max - min)
                    normalized_conviction = (conviction - min_conviction) / (1 - min_conviction)
                    # Examples:
                    #   conviction = 0.30 → normalized = 0.00 (just met threshold)
                    #   conviction = 0.65 → normalized = 0.50 (medium confidence)
                    #   conviction = 1.00 → normalized = 1.00 (maximum confidence)
                    
                    # STEP 2c: CONVICTION MULTIPLIER
                    # Scale position size based on confidence level
                    # Range: 1.0 (at threshold) to 2.5 (at 100% confidence)
                    conviction_multiplier = 1.0 + (normalized_conviction * 1.5)
                    # Examples:
                    #   normalized = 0.00 → multiplier = 1.00 (use base size only)
                    #   normalized = 0.50 → multiplier = 1.75 (75% larger than base)
                    #   normalized = 1.00 → multiplier = 2.50 (150% larger than base)
                    
                    # STEP 2d: CALCULATE POSITION FRACTION
                    # Combine direction (±1) with base size and multiplier
                    f_t = direction * base_size * conviction_multiplier
                    # Examples:
                    #   BUY (dir=+1), conviction=0.80 → f_t ≈ +0.31 (31% long)
                    #   SELL (dir=-1), conviction=0.35 → f_t ≈ -0.17 (17% short)
                    
                    # STEP 2e: CAP POSITION SIZE
                    # Limit maximum position to ±60% of equity for risk management
                    f_t = np.clip(f_t, -0.60, 0.60)
                    
                else:
                    # FILTERED: Conviction below 30% threshold
                    # Don't trade - signal is too uncertain
                    low_conviction_count += 1
                    f_t = 0
                
                # STEP 3: CALCULATE NOTIONAL AND UNITS
                # Convert position fraction to dollar amount and number of units
                target_notional = f_t * equity  # Dollar value of position
                target_units = target_notional / current_price  # Number of contracts/units
                target_direction = np.sign(target_units)  # +1 for long, -1 for short, 0 for flat
                # Example: f_t = 0.31, equity = $10M, price = $70
                #   → notional = $3.1M, units = 44,286
                
                # Check if DIRECTION CHANGED or SAME DIRECTION
                if target_direction != current_direction and target_direction != 0:
                    # SCENARIO 1: DIRECTION REVERSAL - Close ALL existing, Open NEW opposite
                    if position != 0:
                        # CLOSE ALL existing positions (REALIZE P&L)
                        pnl = position * current_price - position_notional
                        equity += pnl
                        total_realized_pnl += pnl
                        
                        trades.append({
                            'date': date,
                            'action': 'close',
                            'units': position,
                            'price': current_price,
                            'pnl': pnl,
                            'realized_pnl': total_realized_pnl
                        })
                    
                        # Reset for new position
                        current_unrealized_pnl = 0
                    
                    # OPEN NEW position in opposite direction
                    position = target_units
                    position_notional = position * current_price
                    entry_price = current_price
                    current_direction = target_direction
                    
                    if abs(position) > 0.01:
                        trades.append({
                            'date': date,
                            'action': 'open',
                            'units': position,
                            'price': current_price,
                            'notional': position_notional,
                            'entry_price': entry_price
                        })
                
                elif target_direction == current_direction and abs(target_units - position) > 0.01:
                    # SCENARIO 2: SAME DIRECTION - ADD ON (pyramid/scale in)
                    # Calculate units to add (or reduce if conviction decreased)
                    units_to_add = target_units - position
                    notional_to_add = units_to_add * current_price
                    
                    # Update position (ADD ON, don't close)
                    position = target_units
                    position_notional += notional_to_add  # Add to existing notional
                    # Don't change entry_price - weighted average is tracked in position_notional
                    
                    if abs(units_to_add) > 0.01:
                        trades.append({
                            'date': date,
                            'action': 'add_on' if units_to_add > 0 else 'reduce',
                            'units': units_to_add,  # Delta (can be negative)
                            'price': current_price,
                            'notional': notional_to_add,
                            'total_position': position  # Total position after add-on
                        })
                else:
                    # Position unchanged - hold
                    pass
            else:
                # Flat signal (conviction < 1%) - DO NOTHING, hold existing position
                # Don't close, don't open, just maintain current state
                pass
            
            # Calculate unrealized P&L for open positions
            if position != 0:
                current_unrealized_pnl = position * current_price - position_notional
            else:
                current_unrealized_pnl = 0
            
            # Update equity curve
            current_equity = equity + current_unrealized_pnl
            equity_curve.append({
                'date': date,
                'equity': current_equity,
                'position': position,
                'price': current_price,
                'realized_pnl': total_realized_pnl,
                'unrealized_pnl': current_unrealized_pnl
            })
        
        # DON'T force close final position - leave it open with unrealized P&L
        # The position will show in CSV with Exit_Date = 'OPEN'
        if position != 0:
            final_price = signals_df['Brent_Price'].iloc[-1]
            current_unrealized_pnl = position * final_price - position_notional
            # Don't add to equity - it's unrealized

        
        # Store results
        trades_df = pd.DataFrame(trades)
        equity_curve_df = pd.DataFrame(equity_curve)
        if not equity_curve_df.empty:
            equity_curve_df.set_index('date', inplace=True)
        
        # Calculate final equity including unrealized P&L
        final_equity = equity + current_unrealized_pnl
        
        # Save trades to CSV - one row per OPEN/ADD_ON with exit date
        if not trades_df.empty:
            trade_log = []
            trade_num = 0
            pending_trades = []  # Store open/add-on trades waiting for exit
            current_position_units = 0
            current_position_notional = 0
            
            for idx, row in trades_df.iterrows():
                action = row['action']
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                price = row['price']
                
                # Get conviction from signals_df at this date
                trade_date = row['date']
                conviction = abs(signals_df.loc[trade_date, 'signal_raw']) if trade_date in signals_df.index else 0.0
                
                if action == 'open':
                    trade_num += 1
                    direction = "LONG" if row['units'] > 0 else "SHORT"
                    units = abs(row['units'])
                    current_position_units = row['units']
                    current_position_notional = row['notional']
                    
                    # Store this trade, waiting for exit
                    pending_trades.append({
                        'Trade_#': trade_num,
                        'Trade_Type': 'NEW',
                        'Direction': direction,
                        'Entry_Date': date_str,
                        'Entry_Price': price,
                        'Units': units,
                        'Entry_Notional': abs(row['notional']),
                        'Conviction': conviction
                    })
                    
                elif action in ['add_on', 'reduce']:
                    trade_num += 1
                    direction = "LONG" if current_position_units > 0 else "SHORT"
                    units = abs(row['units'])
                    
                    # Update position tracking
                    current_position_units = row.get('total_position', current_position_units)
                    current_position_notional += row['notional']
                    
                    # Store this trade, waiting for exit
                    pending_trades.append({
                        'Trade_#': trade_num,
                        'Trade_Type': 'ADD_ON' if action == 'add_on' else 'REDUCE',
                        'Direction': direction,
                        'Entry_Date': date_str,
                        'Entry_Price': price,
                        'Units': units,
                        'Entry_Notional': abs(row['notional']),
                        'Conviction': conviction
                    })
                    
                elif action == 'close':
                    exit_date = date_str
                    exit_price = price
                    total_pnl = row.get('pnl', 0)
                    
                    # Allocate P&L proportionally to all pending trades
                    if len(pending_trades) > 0:
                        total_notional = sum(t['Entry_Notional'] for t in pending_trades)
                        
                        for trade in pending_trades:
                            # Proportional P&L allocation
                            if total_notional > 0:
                                pnl_share = total_pnl * (trade['Entry_Notional'] / total_notional)
                                pnl_pct = (pnl_share / trade['Entry_Notional'] * 100) if trade['Entry_Notional'] > 0 else 0
                            else:
                                pnl_share = 0
                                pnl_pct = 0
                            
                            # Calculate unrealized at entry
                            unrealized_at_entry = 0  # At entry, unrealized is 0
                            
                            trade_log.append({
                                'Trade_#': trade['Trade_#'],
                                'Trade_Type': trade['Trade_Type'],
                                'Direction': trade['Direction'],
                                'Entry_Date': trade['Entry_Date'],
                                'Entry_Price': trade['Entry_Price'],
                                'Exit_Date': exit_date,
                                'Exit_Price': exit_price,
                                'Units': trade['Units'],
                                'Entry_Notional': trade['Entry_Notional'],
                                'Conviction': trade['Conviction'],
                                'Realized_PnL': pnl_share,
                                'Unrealized_PnL': 0,  # Position closed, all P&L is realized
                                'PnL_Pct': pnl_pct
                            })
                    
                    # Clear pending trades and reset position
                    pending_trades = []
                    current_position_units = 0
                    current_position_notional = 0
            
            # Handle any unclosed trades at end
            if len(pending_trades) > 0:
                exit_date = 'OPEN'
                exit_price = signals_df['Brent_Price'].iloc[-1]
                
                for trade in pending_trades:
                    # Calculate unrealized P&L for open positions
                    direction_multiplier = 1 if trade['Direction'] == 'LONG' else -1
                    unrealized_pnl = direction_multiplier * trade['Units'] * (exit_price - trade['Entry_Price'])
                    unrealized_pct = (unrealized_pnl / trade['Entry_Notional'] * 100) if trade['Entry_Notional'] > 0 else 0
                    
                    trade_log.append({
                        'Trade_#': trade['Trade_#'],
                        'Trade_Type': trade['Trade_Type'],
                        'Direction': trade['Direction'],
                        'Entry_Date': trade['Entry_Date'],
                        'Entry_Price': trade['Entry_Price'],
                        'Exit_Date': exit_date,
                        'Exit_Price': exit_price,
                        'Units': trade['Units'],
                        'Entry_Notional': trade['Entry_Notional'],
                        'Conviction': trade['Conviction'],
                        'Realized_PnL': 0,  # Still open, not realized yet
                        'Unrealized_PnL': unrealized_pnl,  # Mark-to-market P&L
                        'PnL_Pct': unrealized_pct
                    })
            
            trade_log_df = pd.DataFrame(trade_log)
            csv_filename = f'trades_{period_name}_period.csv'
            trade_log_df.to_csv(csv_filename, index=False)
            print(f"✓ Backtest completed: {len(trade_log_df)} trades saved to {csv_filename}")
        
        # Save equity curve to CSV for performance metrics calculation
        if not equity_curve_df.empty:
            equity_csv_filename = f'equity_{period_name}_period.csv'
            equity_curve_df.to_csv(equity_csv_filename)
        
        return trades_df, equity_curve_df
    

    
    def calculate_metrics(self, signals_df, trades_df, equity_curve_df, period_start, period_end):
        """Calculate performance metrics."""
        if equity_curve_df is None or equity_curve_df.empty:
            return None
        
        # Portfolio metrics
        daily_returns = equity_curve_df['equity'].pct_change().dropna()
        
        # Annualized metrics using formula: (1 + R_total)^(252/n) - 1
        total_return = (equity_curve_df['equity'].iloc[-1] / equity_curve_df['equity'].iloc[0]) - 1
        n_days = len(equity_curve_df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Classification metrics (if we have actual labels)
        # Get labels that match signal indices for the period
        period_labels = self.labels.loc[
            (self.labels.index >= period_start) & 
            (self.labels.index <= period_end)
        ]
        
        if not period_labels.empty and signals_df is not None:
            # Only use predictions where we have labels
            common_idx = period_labels.index.intersection(signals_df.index)
            if len(common_idx) > 0:
                val_labels = period_labels.loc[common_idx]
                val_predictions = signals_df.loc[common_idx, 'p_buy']
                val_pred_binary = (val_predictions >= 0.5).astype(int)
            else:
                val_labels = pd.Series()
                val_pred_binary = pd.Series()
        else:
            val_labels = pd.Series()
            val_pred_binary = pd.Series()
        
        
        metrics = {
      
            'portfolio': {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'initial_capital': self.initial_capital,
                'final_equity': equity_curve_df['equity'].iloc[-1],
                'total_return': (equity_curve_df['equity'].iloc[-1] / self.initial_capital - 1)
            },
            'trades': trades_df
        }
        
        return metrics
    
    def print_results(self, metrics, period_name='validation'):
        """
        Print evaluation results in table format.
        
        Trading Statistics Logic:
        - Each row in CSV = 1 trade (NEW, ADD_ON, or REDUCE action)
        - Total P&L = Realized_PnL + Unrealized_PnL for each trade
        - Winning trade: Total P&L > 0
        - Losing trade: Total P&L < 0
        - Average Win: Mean of Total_PnL for winning trades only
        - Average Loss: Mean of Total_PnL for losing trades only
        
        Annualized Return: Calculated using 252 trading days per year
        
        Parameters:
        -----------
        metrics : dict
            Metrics dictionary from calculate_metrics
        period_name : str
            'training' or 'validation' to identify CSV file
        """
        if metrics is None:
            print("No metrics available")
            return
        
        print("\n" + "="*80)
        print(" "*22 + "VALIDATION PERFORMANCE SUMMARY")
        print("="*80)
        
        # SECTION 1: TRADING STATISTICS (from CSV - each row is a trade)
        csv_filename = f'trades_{period_name}_period.csv'
        import os
        
        if os.path.exists(csv_filename):
            try:
                # Read CSV to get ALL trades (each row = 1 trade)
                csv_trades = pd.read_csv(csv_filename)
                
                print("TRADING STATISTICS")
                print("-" * 80)
                
                # Calculate Total P&L for each trade (Realized + Unrealized)
                csv_trades['Total_PnL'] = csv_trades['Realized_PnL'] + csv_trades['Unrealized_PnL']
                
                # Determine winning/losing based on Total P&L
                winning_trades = csv_trades[csv_trades['Total_PnL'] > 0]
                losing_trades = csv_trades[csv_trades['Total_PnL'] < 0]
                total_trades = len(csv_trades)
                
                trade_data = {
                    'Metric': [
                        'Total Trades',
                        'Winning Trades',
                        'Losing Trades', 
                        'Win Rate'
                    ],
                    'Value': [
                        f"{total_trades}",
                        f"{len(winning_trades)}",
                        f"{len(losing_trades)}",
                        f"{(len(winning_trades)/total_trades*100):.1f}%" if total_trades > 0 else "0.0%"
                    ]
                }
                trade_df = pd.DataFrame(trade_data)
                print(trade_df.to_string(index=False))
            except Exception as e:
                print(f"Error reading CSV: {e}")
                print("Trading statistics unavailable")
        else:
            print(f"CSV file not found: {csv_filename}")
            print("Trading statistics unavailable")
        
        # SECTION 2: PROFIT AND LOSS (from CSV)
        if os.path.exists(csv_filename):
            try:
                csv_trades = pd.read_csv(csv_filename)
                
                print("PROFIT AND LOSS")
                print("-" * 80)
                
                # Calculate Total P&L for each trade (Realized + Unrealized)
                csv_trades['Total_PnL'] = csv_trades['Realized_PnL'] + csv_trades['Unrealized_PnL']
                
                # Separate winning and losing trades based on Total P&L
                winning_trades_csv = csv_trades[csv_trades['Total_PnL'] > 0]
                losing_trades_csv = csv_trades[csv_trades['Total_PnL'] < 0]
                
                # Total realized and unrealized P&L
                total_realized = csv_trades['Realized_PnL'].sum()
                total_unrealized = csv_trades['Unrealized_PnL'].sum()
                total_pnl = total_realized + total_unrealized
                
                pnl_data = {
                    'Metric': [
                        'Realized P&L',
                        'Unrealized P&L',
                        'Total P&L',
                        'Total Return',
                        'Average Win',
                        'Average Loss',
                        'Largest Win',
                        'Largest Loss',
                        'Profit Factor'
                    ],
                    'Value': [
                        f"${total_realized:,.2f}",
                        f"${total_unrealized:,.2f}",
                        f"${total_pnl:,.2f}",
                        f"{(metrics['portfolio']['total_return'] * 100):.2f}%",
                        f"${winning_trades_csv['Total_PnL'].mean():,.2f}" if len(winning_trades_csv) > 0 else "$0.00",
                        f"${losing_trades_csv['Total_PnL'].mean():,.2f}" if len(losing_trades_csv) > 0 else "$0.00",
                        f"${winning_trades_csv['Total_PnL'].max():,.2f}" if len(winning_trades_csv) > 0 else "$0.00",
                        f"${losing_trades_csv['Total_PnL'].min():,.2f}" if len(losing_trades_csv) > 0 else "$0.00",
                        f"{abs(winning_trades_csv['Total_PnL'].sum() / losing_trades_csv['Total_PnL'].sum()):.2f}" if len(losing_trades_csv) > 0 and losing_trades_csv['Total_PnL'].sum() != 0 else "N/A"
                    ]
                }
                pnl_df = pd.DataFrame(pnl_data)
                print(pnl_df.to_string(index=False))
            except Exception as e:
                print(f"Error calculating P&L: {e}")
        
        # SECTION 3: PERFORMANCE METRICS
        print("PERFORMANCE METRICS")
        print("-" * 80)
        
        metrics_data = {
            'Metric': [
                'Annualized Return',
                'Volatility', 
                'Max Drawdown',
                'Calmar Ratio',
                'Sharpe Ratio'
            ],
            'Value': [
                f"{metrics['portfolio']['annual_return']*100:.2f}%",
                f"{metrics['portfolio']['annual_volatility']*100:.2f}%",
                f"{metrics['portfolio']['max_drawdown']*100:.2f}%",
                f"{metrics['portfolio']['calmar_ratio']:.4f}",
                f"{metrics['portfolio']['sharpe_ratio']:.4f}"
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)
        print(metrics_df.to_string(index=False))
        
        print("\n" + "="*80)
    
    def plot_period_results(self, signals_df, trades_df, equity_curve_df, 
                           period_start, period_end, period_name):
        """
        Create visualization plots for a specific period.
        
        The VALIDATION PERFORMANCE SUMMARY in the plot reads from CSV to ensure
        consistency with terminal output:
        - Each row in CSV = 1 trade (NEW, ADD_ON, REDUCE)
        - Total P&L = Realized_PnL + Unrealized_PnL
        - Winning trade: Total P&L > 0, Losing trade: Total P&L < 0
        - Average Win/Loss: Mean of Total P&L for respective trades only
        - Annualized Return: Uses 252 trading days
        """
        
        # Check if we have actual labels for this period
        period_labels = self.labels.loc[
            (self.labels.index >= period_start) & 
            (self.labels.index <= period_end)
        ]
        has_labels = not period_labels.empty
        
        # Use 2x2 layout for training (with labels), 1x3 layout for validation (no labels)
        if has_labels:
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            ax1 = axes[0, 0]
            ax2 = axes[0, 1]
            ax3 = axes[1, 0]
            ax4 = axes[1, 1]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            ax2 = axes[0]
            ax3 = axes[1]
            ax4 = axes[2]
        
        # Get data for the period
        period_mask = (self.data.index >= period_start) & (self.data.index <= period_end)
        period_data = self.data[period_mask]
        
        # 1. ACTUAL BUY/SELL LABELS (only for training period)
        if has_labels:
            # Plot price
            ax1.plot(period_data.index, period_data['Brent_Price'], 
                    label='Brent Price', linewidth=2, color='blue', zorder=1)
            
            buy_labels = period_labels[period_labels == 1]
            sell_labels = period_labels[period_labels == 0]
            
            # Plot actual labels as large circles
            for idx, date in enumerate(buy_labels.index):
                if date in period_data.index:
                    ax1.scatter(date, period_data.loc[date, 'Brent_Price'], 
                              color='lightgreen', marker='o', s=200, alpha=0.6, 
                              edgecolors='darkgreen', linewidths=2, 
                              label='Actual BUY Label' if idx == 0 else '', zorder=2)
            
            for idx, date in enumerate(sell_labels.index):
                if date in period_data.index:
                    ax1.scatter(date, period_data.loc[date, 'Brent_Price'], 
                              color='lightcoral', marker='o', s=200, alpha=0.6,
                              edgecolors='darkred', linewidths=2, 
                              label='Actual SELL Label' if idx == 0 else '', zorder=2)
            
            ax1.set_title(f'{period_name.upper()}: Actual BUY/SELL Labels', 
                         fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=10)
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 2. PREDICTED BUY/SELL SIGNALS
        
        # Plot price
        ax2.plot(period_data.index, period_data['Brent_Price'], 
                label='Brent Price', linewidth=2, color='blue', zorder=1)
        
        # Add PREDICTED SIGNALS (model predictions)
        if signals_df is not None:
            # Separate high conviction (traded) and low conviction (filtered)
            min_conviction = 0.30
            signals_df_copy = signals_df.copy()
            signals_df_copy['conviction'] = signals_df_copy['signal_raw'].abs()
            
            # High conviction signals (traded)
            high_conviction = signals_df_copy[signals_df_copy['conviction'] >= min_conviction]
            buy_high = high_conviction[high_conviction['direction'] == 1]
            sell_high = high_conviction[high_conviction['direction'] == -1]
            
            # Low conviction signals (filtered out)
            low_conviction = signals_df_copy[signals_df_copy['conviction'] < min_conviction]
            buy_low = low_conviction[low_conviction['direction'] == 1]
            sell_low = low_conviction[low_conviction['direction'] == -1]
            
            # Plot HIGH conviction (traded) - solid filled markers
            if not buy_high.empty:
                ax2.scatter(buy_high.index, buy_high['Brent_Price'], 
                           color='darkgreen', marker='^', s=120, 
                           label='BUY (Traded)', alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)
            if not sell_high.empty:
                ax2.scatter(sell_high.index, sell_high['Brent_Price'], 
                           color='darkred', marker='v', s=120, 
                           label='SELL (Traded)', alpha=0.9, edgecolors='black', linewidths=0.5, zorder=3)
            
            # Plot LOW conviction (filtered) - empty/outline markers with X
            if not buy_low.empty:
                ax2.scatter(buy_low.index, buy_low['Brent_Price'], 
                           color='none', marker='^', s=120, 
                           label='BUY (Low Conv.)', alpha=0.6, edgecolors='green', linewidths=2, zorder=2)
                ax2.scatter(buy_low.index, buy_low['Brent_Price'], 
                           color='gray', marker='x', s=80, alpha=0.8, linewidths=2, zorder=2)
            if not sell_low.empty:
                ax2.scatter(sell_low.index, sell_low['Brent_Price'], 
                           color='none', marker='v', s=120, 
                           label='SELL (Low Conv.)', alpha=0.6, edgecolors='red', linewidths=2, zorder=2)
                ax2.scatter(sell_low.index, sell_low['Brent_Price'], 
                           color='gray', marker='x', s=80, alpha=0.8, linewidths=2, zorder=2)
        
        ax2.set_title(f'{period_name.upper()}: Predicted BUY/SELL Signals\n(Solid = Traded, Hollow+X = Low Conviction)', 
                     fontsize=11, fontweight='bold')
        ax2.set_ylabel('Price (USD)', fontsize=10)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. Cumulative Daily Profit/Loss (read from equity CSV)
        equity_csv_filename = f'equity_{period_name}_period.csv'
        import os
        if os.path.exists(equity_csv_filename):
            # Read equity from CSV
            equity_df = pd.read_csv(equity_csv_filename, index_col=0, parse_dates=True)
            
            # Calculate cumulative daily P&L
            # Daily P&L = change in equity from previous day
            daily_pnl = equity_df['equity'].diff()
            # First day P&L is equity - initial_capital
            daily_pnl.iloc[0] = equity_df['equity'].iloc[0] - self.initial_capital
            # Cumulative P&L = sum of all daily P&L
            cumulative_pnl = daily_pnl.cumsum()
            
            # Plot cumulative P&L
            ax3.plot(cumulative_pnl.index, cumulative_pnl.values, 
                    linewidth=2.5, color='darkblue', label='Cumulative P&L')
            ax3.axhline(y=0, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Break-even')
            ax3.fill_between(cumulative_pnl.index, 0, 
                            cumulative_pnl.values, 
                            where=(cumulative_pnl.values >= 0), 
                            color='green', alpha=0.2, label='Profit Region')
            ax3.fill_between(cumulative_pnl.index, 0, 
                            cumulative_pnl.values, 
                            where=(cumulative_pnl.values < 0), 
                            color='red', alpha=0.2, label='Loss Region')
            ax3.set_title(f'{period_name.upper()}: Cumulative Daily P&L', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Cumulative P&L (USD)', fontsize=10)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
        elif equity_curve_df is not None and not equity_curve_df.empty:
            # Fallback to passed equity_curve_df if CSV doesn't exist
            daily_pnl = equity_curve_df['equity'].diff()
            daily_pnl.iloc[0] = equity_curve_df['equity'].iloc[0] - self.initial_capital
            cumulative_pnl = daily_pnl.cumsum()
            
            ax3.plot(cumulative_pnl.index, cumulative_pnl.values, 
                    linewidth=2.5, color='darkblue', label='Cumulative P&L')
            ax3.axhline(y=0, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Break-even')
            ax3.fill_between(cumulative_pnl.index, 0, cumulative_pnl.values, 
                            where=(cumulative_pnl.values >= 0), 
                            color='green', alpha=0.2, label='Profit Region')
            ax3.fill_between(cumulative_pnl.index, 0, cumulative_pnl.values, 
                            where=(cumulative_pnl.values < 0), 
                            color='red', alpha=0.2, label='Loss Region')
            ax3.set_title(f'{period_name.upper()}: Cumulative Daily P&L', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Cumulative P&L (USD)', fontsize=10)
            ax3.legend(loc='best', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 4. Performance metrics summary (read from CSV - same logic as terminal output)
        ax4.axis('off')
        
        # Read from CSV to ensure consistency with terminal output
        csv_filename = f'trades_{period_name}_period.csv'
        equity_csv_filename = f'equity_{period_name}_period.csv'
        import os
        
        if os.path.exists(csv_filename):
            try:
                # Read trades CSV - each row = 1 trade
                csv_trades = pd.read_csv(csv_filename)
                
                # Calculate Total P&L for each trade (Realized + Unrealized)
                csv_trades['Total_PnL'] = csv_trades['Realized_PnL'] + csv_trades['Unrealized_PnL']
                
                # Determine winning/losing based on Total P&L
                winning_trades = csv_trades[csv_trades['Total_PnL'] > 0]
                losing_trades = csv_trades[csv_trades['Total_PnL'] < 0]
                total_trades = len(csv_trades)
                
                # Calculate metrics from CSV
                total_realized = csv_trades['Realized_PnL'].sum()
                total_unrealized = csv_trades['Unrealized_PnL'].sum()
                total_return = ((total_realized + total_unrealized) / self.initial_capital) * 100
                
                # Average win/loss based on Total P&L
                avg_win = winning_trades['Total_PnL'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['Total_PnL'].mean() if len(losing_trades) > 0 else 0
                largest_win = winning_trades['Total_PnL'].max() if len(winning_trades) > 0 else 0
                largest_loss = losing_trades['Total_PnL'].min() if len(losing_trades) > 0 else 0
                
                # Calculate performance metrics from equity CSV
                if os.path.exists(equity_csv_filename):
                    equity_df = pd.read_csv(equity_csv_filename, index_col=0, parse_dates=True)
                    daily_returns = equity_df['equity'].pct_change().dropna()
                    
                    # Annualized return using formula: (1 + R_total)^(252/n) - 1
                    total_return_pct = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
                    n_days = len(equity_df)
                    annual_return = (1 + total_return_pct) ** (252 / n_days) - 1 if n_days > 0 else 0
                    annual_vol = daily_returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    # Max drawdown
                    cumulative = equity_df['equity']
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_dd = drawdown.min()
                    calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
                else:
                    annual_return = annual_vol = sharpe_ratio = max_dd = calmar_ratio = 0
                
                # Calculate total P&L
                total_pnl = total_realized + total_unrealized
                
                metrics_text = f"""
        VALIDATION PERFORMANCE SUMMARY
        ═══════════════════════════════════════════
        
        TRADING STATISTICS
        ─────────────────────────────────────────
          Total Trades      {total_trades}
          Winning Trades    {len(winning_trades)}
          Losing Trades     {len(losing_trades)}
          Win Rate          {len(winning_trades)/total_trades*100:.1f}%
        
        PROFIT AND LOSS
        ─────────────────────────────────────────
          Realized P&L      ${total_realized:,.2f}
          Unrealized P&L    ${total_unrealized:,.2f}
          Total P&L         ${total_pnl:,.2f}
          Total Return      {total_return:.2f}%
          Average Win       ${avg_win:,.2f}
          Average Loss      ${avg_loss:,.2f}
          Largest Win       ${largest_win:,.2f}
          Largest Loss      ${largest_loss:,.2f}
        
        PERFORMANCE METRICS
        ─────────────────────────────────────────
          Annualized Return {annual_return*100:.2f}%
          Volatility        {annual_vol*100:.2f}%
          Max Drawdown      {max_dd*100:.2f}%
          Calmar Ratio      {calmar_ratio:.4f}
          Sharpe Ratio      {sharpe_ratio:.4f}
        
        Period: {period_start} to {period_end}
                """
                
                ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            except Exception as e:
                # Fallback if CSV read fails
                ax4.text(0.5, 0.5, f"Error loading metrics\n{str(e)}", 
                        transform=ax4.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'trading_results_{period_name}_period.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot saved: {plot_filename}")
    

    
    def run_complete_system(self):
        """
        Run the complete trading system end-to-end.
        
        Workflow:
        ---------
        1. Extract Data: Download Brent & WTI from Yahoo Finance (2020-2025)
        2. Create Labels: Generate BUY/SELL labels based on 5-day forward returns
        3. Create Features: Build 50+ technical indicators
        4. Select Features: RFE to select top 15 most predictive features
        5. Train Model: XGBoost with time series cross-validation
        6. Training Period Backtest (July 2020 - July 2025):
           - Generate signals, backtest strategy
           - Calculate metrics, save trades to CSV
           - Plot actual labels vs predicted signals
        7. Validation Period Test (August 2025):
           - Generate signals, backtest strategy
           - Calculate metrics, save trades to CSV
           - Plot predicted signals (no labels available)
        8. Generate summary reports and visualizations
        
        Outputs:
        --------
        CSV Files:
        - trades_training_period.csv: All trades with realized/unrealized P&L
        - trades_validation_period.csv: All trades with realized/unrealized P&L
        - equity_training_period.csv: Daily equity curve for training period
        - equity_validation_period.csv: Daily equity curve for validation period
        
        Visualization Files:
        - trading_results_training_period.png: Training visualizations
        - trading_results_validation_period.png: Validation visualizations
        - feature_importance_rfe.png: Feature importance from RFE
        """
        print("\n" + "="*80)
        print(" "*20 + "BRENT CRUDE OIL TRADING SYSTEM")
        print("="*80 + "\n")
        
        # Extract data from Yahoo Finance
        self.extract_data()
        
        # Create forward-looking labels (BUY/SELL)
        self.create_labels()
        
        # Create comprehensive feature set
        self.create_features()
        
        # Select features
        self.select_features()
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Train model
        self.train_model()
        
        print("\n" + "="*80)
        print(" "*15 + "TRAINING PERIOD BACKTEST (2020-2025)")
        print("="*80)
        
        # Generate signals for training period
        train_signals = self.generate_signals(period='training')
        
        # Backtest on training period
        train_trades, train_equity = self.backtest_strategy(train_signals, 'training')
        
        # Calculate metrics for training
        train_metrics = self.calculate_metrics(train_signals, train_trades, train_equity, 
                                               '2020-07-01', '2025-07-31')
        
        # Print results for training
        self.print_results(train_metrics, period_name='training')
        
        # Create plot for training period
        self.plot_period_results(train_signals, train_trades, train_equity,
                                '2020-07-01', '2025-07-31', 'training')
        
        print("\n" + "="*80)
        print(" "*15 + "VALIDATION PERIOD TEST (AUG 2025)")
        print("="*80)
        
        # Generate signals for validation period
        test_signals = self.generate_signals(period='validation')
        
        # Backtest on validation period
        test_trades, test_equity = self.backtest_strategy(test_signals, 'validation')
        
        # Calculate metrics for validation
        test_metrics = self.calculate_metrics(test_signals, test_trades, test_equity,
                                              '2025-08-01', '2025-08-31')
        
        # Print results for validation
        self.print_results(test_metrics, period_name='validation')
        
        # Create plot for validation period
        self.plot_period_results(test_signals, test_trades, test_equity,
                                '2025-08-01', '2025-08-31', 'validation')
        
        print("\n" + "="*80)
        print(" "*25 + "FILES GENERATED")
        print("="*80)
        print("  CSV Files:")
        print("    • trades_training_period.csv")
        print("    • trades_validation_period.csv")
        print("    • equity_training_period.csv")
        print("    • equity_validation_period.csv")
        print("\n  Visualization Files:")
        print("    • trading_results_training_period.png")
        print("    • trading_results_validation_period.png")
        print("    • feature_importance_rfe.png")
        print("="*80)
        print(" "*22 + "EXECUTION COMPLETED ✓")
        print("="*80 + "\n")


def main():
    """Main function to run the trading system."""
    # Initialize and run the system
    system = BrentTradingSystem(initial_capital=10_000_000)
    system.run_complete_system()


if __name__ == "__main__":
    main()

