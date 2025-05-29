import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 LSTM Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Predict stock prices using advanced AI neural networks**")

# Sidebar for inputs
st.sidebar.header("📊 Configuration")

# Ticker input
ticker = st.sidebar.text_input(
    "Stock Ticker Symbol", 
    value="AAPL", 
    help="Enter any valid stock ticker (e.g., AAPL, GOOGL, TSLA)"
).upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=1825),  # 5 years ago
        help="Start date for historical data"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="End date for historical data"
    )

# Model parameters
st.sidebar.subheader("🔧 Model Parameters")
sequence_length = st.sidebar.slider("Sequence Length (days)", 30, 120, 60, help="Number of previous days to use for prediction")
epochs = st.sidebar.slider("Training Epochs", 20, 100, 50, help="Number of training iterations")
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

# Prediction horizon
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 5, help="Number of future days to predict")

# Functions
@st.cache_data
def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.download(start=start_date, end=end_date)
        
        if data.empty:
            return None, None
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        return data, company_name
    except Exception as e:
        return None, None

def prepare_data(data, sequence_length):
    """Prepare data for LSTM training"""
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test, scaler, scaled_data

def build_lstm_model(sequence_length):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_metrics(actual, predicted):
    """Calculate prediction metrics"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def predict_future_prices(model, last_sequence, scaler, days):
    """Predict future prices"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next day
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Convert back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Main app logic
if st.sidebar.button("🚀 Run Prediction", type="primary"):
    if ticker:
        # Download data
        with st.spinner(f"📈 Downloading data for {ticker}..."):
            data, company_name = download_stock_data(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            st.success(f"✅ Successfully downloaded {len(data)} days of data for {company_name}")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'][-1]:.2f}")
            with col2:
                daily_change = data['Close'][-1] - data['Close'][-2]
                st.metric("Daily Change", f"${daily_change:.2f}", f"{(daily_change/data['Close'][-2]*100):+.2f}%")
            with col3:
                st.metric("52W High", f"${data['Close'].max():.2f}")
            with col4:
                st.metric("52W Low", f"${data['Close'].min():.2f}")
            
            # Prepare data
            with st.spinner("🔧 Preparing data for training..."):
                X_train, X_test, y_train, y_test, scaler, scaled_data = prepare_data(data, sequence_length)
            
            st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
            
            # Build and train model
            with st.spinner("🧠 Building and training LSTM model..."):
                model = build_lstm_model(sequence_length)
                
                # Add progress bar for training
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback to update progress
                class StreamlitCallback:
                    def __init__(self, progress_bar, status_text, total_epochs):
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.total_epochs = total_epochs
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / self.total_epochs
                        self.progress_bar.progress(progress)
                        self.status_text.text(f"Training epoch {epoch + 1}/{self.total_epochs} - Loss: {logs.get('loss', 0):.4f}")
                
                # Train model
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    callbacks=[early_stopping]
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training completed!")
            
            # Make predictions
            with st.spinner("🔮 Making predictions..."):
                train_predictions = model.predict(X_train, verbose=0)
                test_predictions = model.predict(X_test, verbose=0)
                
                # Convert back to original scale
                train_predictions = scaler.inverse_transform(train_predictions)
                test_predictions = scaler.inverse_transform(test_predictions)
                y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                train_metrics = calculate_metrics(y_train_actual, train_predictions)
                test_metrics = calculate_metrics(y_test_actual, test_predictions)
                
                # Predict future prices
                last_sequence = scaled_data[-sequence_length:]
                future_predictions = predict_future_prices(model, last_sequence, scaler, prediction_days)
            
            # Display results
            st.header("📊 Prediction Results")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Training Metrics")
                st.metric("MAPE", f"{train_metrics['MAPE']:.2f}%")
                st.metric("RMSE", f"${train_metrics['RMSE']:.2f}")
            
            with col2:
                st.subheader("🎯 Testing Metrics")
                st.metric("MAPE", f"{test_metrics['MAPE']:.2f}%")
                st.metric("RMSE", f"${test_metrics['RMSE']:.2f}")
            
            # Accuracy interpretation
            if test_metrics['MAPE'] < 5:
                st.markdown('<div class="success-box">🎉 <strong>Excellent accuracy!</strong> MAPE < 5% indicates very reliable predictions.</div>', unsafe_allow_html=True)
            elif test_metrics['MAPE'] < 10:
                st.markdown('<div class="success-box">✅ <strong>Good accuracy!</strong> MAPE < 10% indicates reliable predictions.</div>', unsafe_allow_html=True)
            elif test_metrics['MAPE'] < 20:
                st.markdown('<div class="warning-box">⚠️ <strong>Fair accuracy.</strong> Use predictions with caution.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">❌ <strong>Poor accuracy.</strong> Consider using more data or different parameters.</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.header("📈 Visualizations")
            
            # Create price chart with predictions
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='black', width=1)
            ))
            
            # Training predictions
            train_dates = data.index[sequence_length:sequence_length+len(train_predictions)]
            fig.add_trace(go.Scatter(
                x=train_dates,
                y=train_predictions.flatten(),
                mode='lines',
                name='Training Predictions',
                line=dict(color='blue', width=1, dash='dot'),
                opacity=0.7
            ))
            
            # Test predictions
            test_dates = data.index[sequence_length+len(train_predictions):sequence_length+len(train_predictions)+len(test_predictions)]
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_predictions.flatten(),
                mode='lines',
                name='Test Predictions',
                line=dict(color='red', width=2)
            ))
            
            # Future predictions
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days, freq='D')
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f'{company_name} ({ticker}) - LSTM Price Predictions',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                width=1000,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Future predictions table
            st.header("🔮 Future Price Predictions")
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${price:.2f}" for price in future_predictions],
                'Change from Current': [f"{((price - data['Close'][-1]) / data['Close'][-1] * 100):+.2f}%" for price in future_predictions]
            })
            
            st.dataframe(future_df, use_container_width=True)
            
            # Download results
            st.header("💾 Download Results")
            
            # Prepare download data
            results_df = pd.DataFrame({
                'Date': data.index,
                'Actual_Price': data['Close'].values
            })
            
            # Add predictions where available
            all_predictions = np.full(len(data), np.nan)
            all_predictions[sequence_length:sequence_length+len(train_predictions)] = train_predictions.flatten()
            all_predictions[sequence_length+len(train_predictions):sequence_length+len(train_predictions)+len(test_predictions)] = test_predictions.flatten()
            results_df['Predicted_Price'] = all_predictions
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"{ticker}_lstm_predictions.csv",
                mime="text/csv"
            )
            
            # Model summary
            with st.expander("🔍 View Model Architecture"):
                st.text(model.summary())
            
            # Training history
            with st.expander("📊 View Training History"):
                fig_history = go.Figure()
                fig_history.add_trace(go.Scatter(
                    y=history.history['loss'],
                    mode='lines',
                    name='Training Loss'
                ))
                fig_history.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Validation Loss'
                ))
                fig_history.update_layout(
                    title='Model Training History',
                    xaxis_title='Epoch',
                    yaxis_title='Loss'
                )
                st.plotly_chart(fig_history)
            
        else:
            st.error(f"❌ Could not download data for ticker '{ticker}'. Please check if the ticker symbol is valid.")
    else:
        st.warning("⚠️ Please enter a ticker symbol.")

# Information section
st.header("ℹ️ How It Works")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 LSTM Neural Networks")
    st.write("""
    LSTM (Long Short-Term Memory) networks are a type of recurrent neural network 
    capable of learning long-term dependencies. They're particularly well-suited for 
    time series prediction because they can:
    
    - Remember important patterns from the past
    - Forget irrelevant information
    - Make predictions based on sequential data
    """)

with col2:
    st.subheader("📊 Model Features")
    st.write("""
    Our LSTM model uses:
    
    - **2 LSTM layers** with 50 neurons each
    - **Dropout layers** to prevent overfitting  
    - **Dense layers** for final prediction
    - **60-day sequences** as default input
    - **Min-Max scaling** for data normalization
    """)

# Disclaimer
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer:** This tool is for educational purposes only. Stock market predictions are inherently uncertain, 
and this model should not be used as the sole basis for investment decisions. Always conduct your own research 
and consider consulting with financial professionals before making investment choices. Past performance does not 
guarantee future results.
""")

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit and TensorFlow**")
