from macro_analysis_module import get_us_macro_data, evaluate_macro_data

import streamlit as st
import ccxt
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
import logging
import sys
import time
import concurrent.futures
import requests

# Thiết lập logging
logging.basicConfig(
    filename='error.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Danh sách khung thời gian
TIMEFRAMES = ['15m', '30m', '1h', '2h', '4h', '12h', '1d', '1w']

# Khởi tạo sàn Binance và danh sách 100 cặp giao dịch phổ biến nhất
def initialize_exchange_and_symbols():
    global exchange
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        valid_symbols = [
            symbol for symbol in markets.keys()
            if symbol.endswith('/USDT') and markets[symbol].get('active', False)
        ]
        sorted_symbols = sorted(
            valid_symbols,
            key=lambda x: tickers.get(x, {}).get('quoteVolume', 0),
            reverse=True
        )
        if 'SOL/USDT' not in sorted_symbols and 'SOL/USDT' in markets:
            sorted_symbols.append('SOL/USDT')
        logging.info(f"Khởi tạo sàn Binance và tải {len(sorted_symbols[:100])} cặp giao dịch phổ biến nhất")
        return sorted_symbols[:100]
    except Exception as e:
        logging.error(f"Lỗi khởi tạo sàn Binance: {str(e)}")
        st.error(f"Lỗi kết nối với Binance: {str(e)}")
        return []

exchange = None
TRADING_PAIRS = initialize_exchange_and_symbols()
LEVERAGE_OPTIONS = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]

# Hàm giả lập phân tích cảm xúc thị trường từ mạng xã hội
def fetch_social_sentiment(symbol):
    try:
        base_currency = symbol.split('/')[0]
        sentiment_score = np.random.uniform(0, 1)
        if sentiment_score >= 0.7:
            sentiment = "Tích cực (Bullish)"
        elif sentiment_score <= 0.3:
            sentiment = "Tiêu cực (Bearish)"
        else:
            sentiment = "Trung lập (Neutral)"
        logging.info(f"Phân tích cảm xúc cho {symbol}: {sentiment} (Score: {sentiment_score:.2f})")
        return {
            'sentiment': sentiment,
            'sentiment_score': round(sentiment_score, 2)
        }
    except Exception as e:
        logging.error(f"Lỗi phân tích cảm xúc cho {symbol}: {str(e)}")
        return {
            'sentiment': "Không xác định",
            'sentiment_score': 0.0
        }

# Hàm kiểm tra kết nối API
def check_api_connection():
    try:
        exchange.fetch_ticker('BTC/USDT')
        return True
    except Exception as e:
        logging.error(f"Lỗi kiểm tra kết nối API: {str(e)}")
        return False

# Hàm kiểm tra trạng thái cặp giao dịch
def is_symbol_active(symbol):
    try:
        market = exchange.markets.get(symbol)
        return market and market.get('active', False)
    except Exception as e:
        logging.error(f"Cặp {symbol} không khả dụng hoặc đã bị delist: {str(e)}")
        return False

# Hàm lấy dữ liệu giá từ Binance với cache
@st.cache_data(ttl=300)
def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=100):
    global exchange
    try:
        if exchange is None or not check_api_connection():
            exchange = ccxt.binance()
            logging.info("Khởi tạo lại sàn Binance trong fetch_ohlcv")
        if symbol not in TRADING_PAIRS or not is_symbol_active(symbol):
            logging.error(f"Cặp giao dịch {symbol} không hợp lệ hoặc đã bị delist")
            return None
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if len(df) < 20 or df['high'].nunique() == 1 or df['low'].nunique() == 1:
            logging.warning(f"Dữ liệu không đủ hoặc không biến động cho {symbol}")
            return None
        logging.info(f"Lấy dữ liệu thành công cho {symbol} ({timeframe})")
        return df
    except Exception as e:
        logging.error(f"Lỗi lấy dữ liệu cho {symbol}: {str(e)}")
        return None

# Hàm tính toán các chỉ báo kỹ thuật với cache
@st.cache_data
def calculate_indicators(df, rsi_period=14, wma_period=45, ema_period=9):
    try:
        df = df.copy()
        if len(df) >= 2:
            df_obv = df.iloc[:-1].copy()
        else:
            raise ValueError("Dữ liệu không đủ để loại bỏ nến hiện tại (cần ít nhất 2 hàng)")
        
        # WMA RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
        df['ema9_rsi'] = ta.trend.EMAIndicator(df['rsi'], window=ema_period).ema_indicator()
        df['wma45_rsi'] = ta.trend.WMAIndicator(df['rsi'], window=wma_period).wma()
        
        # Các chỉ báo khác
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['di_plus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx_pos()
        df['di_minus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx_neg()
        df_obv['obv'] = ta.volume.OnBalanceVolumeIndicator(df_obv['close'], df_obv['volume']).on_balance_volume()
        df['obv'] = pd.concat([df_obv['obv'], pd.Series([np.nan])], ignore_index=True)[:len(df)]
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['ema12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['volume_ma19'] = df['volume'].rolling(window=19).mean()
        logging.info("Tính toán chỉ báo thành công, bao gồm WMA RSI")
        return df
    except Exception as e:
        logging.error(f"Lỗi tính toán chỉ báo: {str(e)}")
        st.error(f"Lỗi tính toán chỉ báo: {str(e)}")
        return None

# Hàm xác định mức hỗ trợ và kháng cự
def find_support_resistance(df, window=20):
    try:
        support = df['low'].rolling(window=window).min().iloc[-1]
        resistance = df['high'].rolling(window=window).max().iloc[-1]
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df else (resistance - support) / 2
        if abs(resistance - support) < atr * 0.5:
            support = min(support, current_price - atr)
            resistance = max(resistance, current_price + atr)
        logging.info(f"Hỗ trợ: {support}, Kháng cự: {resistance}")
        return support, resistance
    except Exception as e:
        logging.error(f"Lỗi tính toán hỗ trợ/kháng cự: {str(e)}")
        return None, None

# Hàm tính toán lãi, lỗ, call margin, và đòn bẩy đề xuất
def calculate_trading_metrics(current_price, stop_loss, take_profit, investment, leverage, recommendation):
    try:
        if current_price <= 0 or investment <= 0 or leverage <= 0:
            raise ValueError("Giá hiện tại, số tiền đầu tư hoặc đòn bẩy không hợp lệ")
        if stop_loss is None or take_profit is None:
            stop_loss = current_price * 0.99 if 'MUA' in recommendation else current_price * 1.01
            take_profit = current_price * 1.02 if 'MUA' in recommendation else current_price * 0.98
            logging.warning(f"Stop loss hoặc take profit là None, sử dụng giá trị mặc định: SL={stop_loss}, TP={take_profit}")

        quantity = (investment * leverage) / current_price

        if 'MUA' in recommendation:
            if stop_loss >= current_price:
                stop_loss = current_price * 0.99
                logging.warning(f"Stop loss không hợp lệ, điều chỉnh thành {stop_loss}")
            if take_profit <= current_price:
                take_profit = current_price * 1.02
                logging.warning(f"Take profit không hợp lệ, điều chỉnh thành {take_profit}")
            profit = (take_profit - current_price) * quantity
            loss = (current_price - stop_loss) * quantity
        elif 'BÁN' in recommendation:
            if stop_loss <= current_price:
                stop_loss = current_price * 1.01
                logging.warning(f"Stop loss không hợp lệ, điều chỉnh thành {stop_loss}")
            if take_profit >= current_price:
                take_profit = current_price * 0.98
                logging.warning(f"Take profit không hợp lệ, điều chỉnh thành {take_profit}")
            profit = (current_price - take_profit) * quantity
            loss = (stop_loss - current_price) * quantity
        else:
            profit = (take_profit - current_price) * quantity
            loss = (current_price - stop_loss) * quantity

        maintenance_margin_rate = 0.1
        max_loss = investment * (1 - maintenance_margin_rate)
        if 'MUA' in recommendation:
            margin_call_price = current_price - (max_loss / (quantity / leverage))
        elif 'BÁN' in recommendation:
            margin_call_price = current_price + (max_loss / (quantity / leverage))
        else:
            margin_call_price = current_price

        if 'MUA' in recommendation and stop_loss < current_price:
            max_leverage = (investment * (1 - maintenance_margin_rate)) / (abs(current_price - stop_loss) * (investment / current_price))
        elif 'BÁN' in recommendation and stop_loss > current_price:
            max_leverage = (investment * (1 - maintenance_margin_rate)) / (abs(stop_loss - current_price) * (investment / current_price))
        else:
            max_leverage = leverage
        max_leverage = min(max(1.0, max_leverage), 100)

        rr_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0.0

        return {
            'profit': round(profit, 2),
            'loss': round(loss, 2),
            'margin_call_price': round(max(margin_call_price, 0), 2),
            'max_leverage': round(max_leverage, 2),
            'rr_ratio': round(rr_ratio, 2)
        }
    except Exception as e:
        logging.error(f"Lỗi tính toán chỉ số giao dịch: {str(e)}")
        return {
            'profit': 0.0,
            'loss': 0.0,
            'margin_call_price': round(current_price, 2),
            'max_leverage': 1.0,
            'rr_ratio': 0.0
        }

# Hàm đưa ra khuyến nghị dựa trên trọng số
@st.cache_data
def generate_recommendation(symbol, timeframe, weights, rsi_period=14, wma_period=45, ema_period=9, rsi_upper=70, rsi_lower=30):
    try:
        tf_index = TIMEFRAMES.index(timeframe)
        timeframes_to_analyze = [(timeframe, weights['current'])]
        if tf_index > 0:
            timeframes_to_analyze.append((TIMEFRAMES[tf_index - 1], weights['smaller']))
        if tf_index < len(TIMEFRAMES) - 1:
            timeframes_to_analyze.append((TIMEFRAMES[tf_index + 1], weights['larger']))
        if len(timeframes_to_analyze) == 2:
            current_tf, current_weight = timeframes_to_analyze[0]
            other_tf, other_weight = timeframes_to_analyze[1]
            total = current_weight + other_weight
            if total > 0:
                timeframes_to_analyze = [(current_tf, current_weight / total), (other_tf, other_weight / total)]
        
        results = []
        combined_score = 0.0
        combined_signals = []
        
        for tf, weight in timeframes_to_analyze:
            df = fetch_ohlcv(symbol=symbol, timeframe=tf, limit=100)
            if df is None or df.empty:
                logging.warning(f"Không thể lấy dữ liệu cho {symbol} ({tf})")
                continue
            df = calculate_indicators(df, rsi_period, wma_period, ema_period)
            if df is None:
                logging.warning(f"Không thể tính toán chỉ báo cho {symbol} ({tf})")
                continue
            
            if len(df) < 3:
                logging.warning(f"Dữ liệu không đủ để phân tích {symbol} ({tf})")
                continue
            
            last_row = df.iloc[-2]
            prev_row = df.iloc[-3]
            signals = []
            total_score = 0.0
            
            # WMA RSI
            rsi_value = round(last_row['rsi'], 2)
            ema9_rsi = round(last_row['ema9_rsi'], 2)
            wma45_rsi = round(last_row['wma45_rsi'], 2)
            rsi_cross_above_ema = last_row['rsi'] > last_row['ema9_rsi'] and prev_row['rsi'] <= prev_row['ema9_rsi']
            rsi_cross_below_ema = last_row['rsi'] < last_row['ema9_rsi'] and prev_row['rsi'] >= prev_row['ema9_rsi']
            rsi_cross_above_wma = last_row['rsi'] > last_row['wma45_rsi'] and prev_row['rsi'] <= prev_row['wma45_rsi']
            rsi_cross_below_wma = last_row['rsi'] < last_row['wma45_rsi'] and prev_row['rsi'] >= prev_row['wma45_rsi']
            wma_decreasing = last_row['wma45_rsi'] < prev_row['wma45_rsi']
            
            if rsi_cross_above_wma and rsi_value < rsi_lower:
                signals.append(('MUA_RẤT_MẠNH', f'WMA RSI Cắt Lên WMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI cắt lên WMA45 và gần ngưỡng quá bán, tín hiệu tăng rất mạnh.', 0.5))
                total_score += 0.5
            elif rsi_cross_above_ema and rsi_value < rsi_lower + 10:
                signals.append(('MUA_MẠNH', f'WMA RSI Cắt Lên EMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI cắt lên EMA9 và gần ngưỡng quá bán, tín hiệu tăng mạnh.', 0.3))
                total_score += 0.3
            elif last_row['rsi'] > last_row['ema9_rsi'] and rsi_value < rsi_lower + 10:
                signals.append(('MUA_TRUNG_BÌNH', f'WMA RSI Trên EMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI trên EMA9 và gần ngưỡng quá bán, tín hiệu tăng nhẹ.', 0.15))
                total_score += 0.15
            elif rsi_cross_below_wma and rsi_value > rsi_upper and wma_decreasing:
                signals.append(('BÁN_RẤT_MẠNH', f'WMA RSI Cắt Xuống WMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI cắt xuống WMA45, WMA giảm, và gần ngưỡng quá mua, tín hiệu giảm rất mạnh.', -0.5))
                total_score -= 0.5
            elif rsi_cross_below_ema and rsi_value > rsi_upper - 10:
                signals.append(('BÁN_MẠNH', f'WMA RSI Cắt Xuống EMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI cắt xuống EMA9 và gần ngưỡng quá mua, tín hiệu giảm mạnh.', -0.3))
                total_score -= 0.3
            elif last_row['rsi'] < last_row['ema9_rsi'] and rsi_value > rsi_upper - 10:
                signals.append(('BÁN_TRUNG_BÌNH', f'WMA RSI Dưới EMA (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): RSI dưới EMA9 và gần ngưỡng quá mua, tín hiệu giảm nhẹ.', -0.15))
                total_score -= 0.15
            else:
                signals.append(('GIỮ', f'WMA RSI Trung Lập (RSI = {rsi_value}, EMA9 = {ema9_rsi}, WMA45 = {wma45_rsi}, {tf}): Không có tín hiệu rõ ràng.', 0.0))
            
            # MACD
            macd_value = round(last_row['macd'], 4)
            macd_signal = round(last_row['macd_signal'], 4)
            macd_histogram = round(last_row['macd_histogram'], 4)
            hist_diff = macd_histogram - round(prev_row['macd_histogram'], 4)
            if last_row['macd'] > last_row['macd_signal'] and last_row['macd_histogram'] > 0 and hist_diff > 0.5 * abs(prev_row['macd_histogram']):
                signals.append(('MUA_RẤT_MẠNH', f'MACD Tăng Mạnh (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng tăng rất mạnh, giá có thể tiếp tục tăng.', 0.4))
                total_score += 0.4
            elif last_row['macd'] > last_row['macd_signal'] and last_row['macd_histogram'] > 0:
                signals.append(('MUA_MẠNH', f'MACD Tăng (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng tăng, giá có khả năng tiếp tục tăng.', 0.25))
                total_score += 0.25
            elif last_row['macd'] > last_row['macd_signal']:
                signals.append(('MUA_TRUNG_BÌNH', f'MACD Tăng Yếu (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng tăng nhẹ, giá có thể tăng.', 0.1))
                total_score += 0.1
            elif last_row['macd'] < last_row['macd_signal'] and last_row['macd_histogram'] < 0 and hist_diff < -0.5 * abs(prev_row['macd_histogram']):
                signals.append(('BÁN_RẤT_MẠNH', f'MACD Giảm Mạnh (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng giảm rất mạnh, giá có thể tiếp tục giảm.', -0.4))
                total_score -= 0.4
            elif last_row['macd'] < last_row['macd_signal'] and last_row['macd_histogram'] < 0:
                signals.append(('BÁN_MẠNH', f'MACD Giảm (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng giảm, giá có khả năng tiếp tục giảm.', -0.25))
                total_score -= 0.25
            elif last_row['macd'] < last_row['macd_signal']:
                signals.append(('BÁN_TRUNG_BÌNH', f'MACD Giảm Yếu (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Động lượng giảm nhẹ, giá có thể giảm.', -0.1))
                total_score -= 0.1
            else:
                signals.append(('GIỮ', f'MACD Trung Lập (MACD = {macd_value}, Signal = {macd_signal}, Histogram = {macd_histogram}, {tf}): Không có tín hiệu rõ ràng.', 0.0))
            
            # Bollinger Bands
            close_price = round(last_row['close'], 2)
            bb_upper = round(last_row['bb_upper'], 2)
            bb_lower = round(last_row['bb_lower'], 2)
            if last_row['close'] <= last_row['bb_lower'] and last_row['rsi'] < rsi_lower:
                signals.append(('MUA_RẤT_MẠNH', f'Dải Bollinger Dưới + Quá Bán (Giá = {close_price}, BB Dưới = {bb_lower}, RSI = {rsi_value}, {tf}): Giá chạm dải dưới và RSI thấp, khả năng đảo chiều tăng mạnh.', 0.35))
                total_score += 0.35
            elif last_row['close'] <= last_row['bb_lower']:
                signals.append(('MUA_MẠNH', f'Dải Bollinger Dưới (Giá = {close_price}, BB Dưới = {bb_lower}, {tf}): Giá chạm dải dưới, khả năng đảo chiều tăng.', 0.2))
                total_score += 0.2
            elif last_row['close'] <= last_row['bb_lower'] * 1.05:
                signals.append(('MUA_TRUNG_BÌNH', f'Gần Dải Bollinger Dưới (Giá = {close_price}, BB Dưới = {bb_lower}, {tf}): Giá gần dải dưới, có thể tăng nhẹ.', 0.1))
                total_score += 0.1
            elif last_row['close'] >= last_row['bb_upper'] and last_row['rsi'] > rsi_upper:
                signals.append(('BÁN_RẤT_MẠNH', f'Dải Bollinger Trên + Quá Mua (Giá = {close_price}, BB Trên = {bb_upper}, RSI = {rsi_value}, {tf}): Giá chạm dải trên và RSI cao, khả năng đảo chiều giảm mạnh.', -0.35))
                total_score -= 0.35
            elif last_row['close'] >= last_row['bb_upper']:
                signals.append(('BÁN_MẠNH', f'Dải Bollinger Trên (Giá = {close_price}, BB Trên = {bb_upper}, {tf}): Giá chạm dải trên, khả năng đảo chiều giảm.', -0.2))
                total_score -= 0.2
            elif last_row['close'] >= last_row['bb_upper'] * 0.95:
                signals.append(('BÁN_TRUNG_BÌNH', f'Gần Dải Bollinger Trên (Giá = {close_price}, BB Trên = {bb_upper}, {tf}): Giá gần dải trên, có thể giảm nhẹ.', -0.1))
                total_score += 0.1
            else:
                signals.append(('GIỮ', f'Dải Bollinger Trung Lập (Giá = {close_price}, BB Trên = {bb_upper}, BB Dưới = {bb_lower}, {tf}): Giá ở giữa dải, không có tín hiệu rõ ràng.', 0.0))
            
            # ADX
            adx_value = round(last_row['adx'], 2)
            di_plus = round(last_row['di_plus'], 2)
            di_minus = round(last_row['di_minus'], 2)
            if last_row['adx'] > 40 and last_row['di_plus'] > last_row['di_minus']:
                signals.append(('MUA_RẤT_MẠNH', f'ADX Xu Hướng Tăng Mạnh (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng tăng rất mạnh, giá có thể tiếp tục tăng.', 0.3))
                total_score += 0.3
            elif last_row['adx'] > 25 and last_row['di_plus'] > last_row['di_minus']:
                signals.append(('MUA_MẠNH', f'ADX Xu Hướng Tăng (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng tăng, giá có khả năng tiếp tục tăng.', 0.2))
                total_score += 0.2
            elif last_row['adx'] > 20 and last_row['di_plus'] > last_row['di_minus']:
                signals.append(('MUA_TRUNG_BÌNH', f'ADX Xu Hướng Tăng Yếu (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng tăng nhẹ, giá có thể tăng.', 0.1))
                total_score += 0.1
            elif last_row['adx'] > 40 and last_row['di_minus'] > last_row['di_plus']:
                signals.append(('BÁN_RẤT_MẠNH', f'ADX Xu Hướng Giảm Mạnh (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng giảm rất mạnh, giá có thể tiếp tục giảm.', -0.3))
                total_score -= 0.3
            elif last_row['adx'] > 25 and last_row['di_minus'] > last_row['di_plus']:
                signals.append(('BÁN_MẠNH', f'ADX Xu Hướng Giảm (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng giảm, giá có khả năng tiếp tục giảm.', -0.2))
                total_score -= 0.2
            elif last_row['adx'] > 20 and last_row['di_minus'] > last_row['di_plus']:
                signals.append(('BÁN_TRUNG_BÌNH', f'ADX Xu Hướng Giảm Yếu (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Xu hướng giảm nhẹ, giá có thể giảm.', -0.1))
                total_score -= 0.1
            else:
                signals.append(('GIỮ', f'ADX Trung Lập (ADX = {adx_value}, +DI = {di_plus}, -DI = {di_minus}, {tf}): Không có xu hướng rõ ràng.', 0.0))
            
            # OBV
            obv_value = round(last_row['obv'], 2)
            obv_diff = last_row['obv'] - prev_row['obv']
            obv_change = obv_diff / abs(prev_row['obv']) if prev_row['obv'] != 0 else 0
            price_up = last_row['close'] > prev_row['close']
            price_down = last_row['close'] < prev_row['close']
            macd_positive = last_row['macd'] > last_row['macd_signal']
            ema_positive = last_row['close'] > last_row['ema12']
            if obv_change > 0.2 and price_up and (macd_positive or ema_positive):
                signals.append(('MUA_RẤT_MẠNH', f'OBV Tăng Mạnh (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng mua mạnh, giá có thể tiếp tục tăng.', 0.3))
                total_score += 0.3
            elif obv_change > 0.1 and price_up:
                signals.append(('MUA_MẠNH', f'OBV Tăng (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng mua tăng, giá có khả năng tiếp tục tăng.', 0.2))
                total_score += 0.2
            elif obv_change > 0.05 and price_up:
                signals.append(('MUA_TRUNG_BÌNH', f'OBV Tăng Yếu (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng mua tăng nhẹ, giá có thể tăng.', 0.1))
                total_score += 0.1
            elif obv_change < -0.2 and price_down and (not macd_positive or not ema_positive):
                signals.append(('BÁN_RẤT_MẠNH', f'OBV Giảm Mạnh (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng bán mạnh, giá có thể tiếp tục giảm.', -0.3))
                total_score -= 0.3
            elif obv_change < -0.1 and price_down:
                signals.append(('BÁN_MẠNH', f'OBV Giảm (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng bán tăng, giá có khả năng tiếp tục giảm.', -0.2))
                total_score -= 0.2
            elif obv_change < -0.05 and price_down:
                signals.append(('BÁN_TRUNG_BÌNH', f'OBV Giảm Yếu (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng bán tăng nhẹ, giá có thể giảm.', -0.1))
                total_score -= 0.1
            else:
                signals.append(('GIỮ', f'OBV Trung Lập (OBV = {obv_value}, Thay đổi = {round(obv_change*100, 2)}%, {tf}): Khối lượng không có tín hiệu rõ ràng.', 0.0))
            
            # EMA
            ema12 = round(last_row['ema12'], 2)
            ema26 = round(last_row['ema26'], 2)
            ema12_cross_above = last_row['ema12'] > last_row['ema26'] and prev_row['ema12'] <= prev_row['ema26']
            ema12_cross_below = last_row['ema12'] < last_row['ema26'] and prev_row['ema12'] >= prev_row['ema26']
            if last_row['close'] > last_row['ema12'] and last_row['ema12'] > last_row['ema26']:
                signals.append(('MUA_RẤT_MẠNH', f'EMA Giá Trên EMA12 và EMA26 (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): Xu hướng tăng mạnh, giá có khả năng tiếp tục tăng.', 0.35))
                total_score += 0.35
            elif last_row['close'] > last_row['ema12'] and ema12_cross_above:
                signals.append(('MUA_MẠNH', f'EMA Giao Cắt Vàng (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): EMA12 cắt lên EMA26, giá có khả năng tăng.', 0.2))
                total_score += 0.2
            elif last_row['close'] > last_row['ema12']:
                signals.append(('MUA_TRUNG_BÌNH', f'EMA Giá Trên EMA12 (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): Giá đang trên xu hướng tăng nhẹ.', 0.1))
                total_score += 0.1
            elif last_row['close'] < last_row['ema12'] and last_row['ema12'] < last_row['ema26']:
                signals.append(('BÁN_RẤT_MẠNH', f'EMA Giá Dưới EMA12 và EMA26 (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): Xu hướng giảm mạnh, giá có khả năng tiếp tục giảm.', -0.35))
                total_score -= 0.35
            elif last_row['close'] < last_row['ema12'] and ema12_cross_below:
                signals.append(('BÁN_MẠNH', f'EMA Giao Cắt Tử Thần (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): EMA12 cắt xuống EMA26, giá có khả năng giảm.', -0.2))
                total_score -= 0.2
            elif last_row['close'] < last_row['ema12']:
                signals.append(('BÁN_TRUNG_BÌNH', f'EMA Giá Dưới EMA12 (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): Giá đang trên xu hướng giảm nhẹ.', -0.1))
                total_score += 0.1
            else:
                signals.append(('GIỮ', f'EMA Trung Lập (Giá = {close_price}, EMA12 = {ema12}, EMA26 = {ema26}, {tf}): Giá nằm giữa EMA12 và EMA26, không có tín hiệu rõ ràng.', 0.0))
            
            combined_score += total_score * weight
            combined_signals.extend([(action, f"{reason} (Trọng số: {weight})", score * weight) for action, reason, score in signals])
            results.append({
                'timeframe': tf,
                'total_score': total_score,
                'signals': signals,
                'df': df
            })
        
        if not results:
            logging.error(f"Không có dữ liệu hợp lệ cho {symbol} trên bất kỳ khung thời gian nào")
            return None
        
        current_result = next((r for r in results if r['timeframe'] == timeframe), results[0])
        df = current_result['df']
        current_price = df['close'].iloc[-2]
        atr = df['atr'].iloc[-2]
        support, resistance = find_support_resistance(df.iloc[:-1])
        if support is None or resistance is None:
            logging.error("Lỗi tính toán hỗ trợ/kháng cự")
            return None
        
        min_spread = atr * 0.5
        if combined_score >= 1.5:
            recommendation = 'MUA_RẤT_MẠNH'
            stop_loss = min(current_price - 1.5 * atr, support * 0.99)
            take_profit = max(current_price + 3 * atr, resistance * 0.99)
            if abs(take_profit - stop_loss) < min_spread:
                take_profit = stop_loss + min_spread
        elif combined_score >= 0.9:
            recommendation = 'MUA_MẠNH'
            stop_loss = min(current_price - 1.5 * atr, support * 0.99)
            take_profit = max(current_price + 3 * atr, resistance * 0.99)
            if abs(take_profit - stop_loss) < min_spread:
                take_profit = stop_loss + min_spread
        elif combined_score >= 0.3:
            recommendation = 'MUA_TRUNG_BÌNH'
            stop_loss = min(current_price - 1.5 * atr, support * 0.99)
            take_profit = max(current_price + 3 * atr, resistance * 0.99)
            if abs(take_profit - stop_loss) < min_spread:
                take_profit = stop_loss + min_spread
        elif combined_score > -0.3:
            recommendation = 'GIỮ'
            stop_loss = current_price - 1.5 * atr
            take_profit = current_price + 3 * atr
            if abs(take_profit - stop_loss) < min_spread:
                take_profit = stop_loss + min_spread
        elif combined_score > -0.9:
            recommendation = 'BÁN_TRUNG_BÌNH'
            stop_loss = max(current_price + 1.5 * atr, resistance * 1.01)
            take_profit = min(current_price - 3 * atr, support * 1.01)
            if abs(stop_loss - take_profit) < min_spread:
                take_profit = stop_loss - min_spread
        elif combined_score > -1.5:
            recommendation = 'BÁN_MẠNH'
            stop_loss = max(current_price + 1.5 * atr, resistance * 1.01)
            take_profit = min(current_price - 3 * atr, support * 1.01)
            if abs(stop_loss - take_profit) < min_spread:
                take_profit = stop_loss - min_spread
        else:
            recommendation = 'BÁN_RẤT_MẠNH'
            stop_loss = max(current_price + 1.5 * atr, resistance * 1.01)
            take_profit = min(current_price - 3 * atr, support * 1.01)
            if abs(stop_loss - take_profit) < min_spread:
                take_profit = stop_loss - min_spread
        
        logging.info(f"Khuyến nghị cho {symbol}: {recommendation}, Tổng trọng số: {combined_score}, Khung: {timeframe}")
        return {
            'recommendation': recommendation,
            'total_score': round(combined_score, 2),
            'signals': combined_signals,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'current_price': round(current_price, 2),
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'df': df,
            'results': results
        }
    except Exception as e:
        logging.error(f"Lỗi tạo khuyến nghị cho {symbol}: {str(e)}")
        return None

# Hàm vẽ đồ thị giá với Dải Bollinger
def plot_price_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Nến'
        ))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='Dải Bollinger Trên', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_middle'], name='Dải Bollinger Giữa', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='Dải Bollinger Dưới', line=dict(color='green')))
        fig.update_layout(
            title='Biểu Đồ Giá với Dải Bollinger',
            xaxis_title='Thời Gian',
            yaxis_title='Giá (USDT)',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ giá: {str(e)}")
        return None

# Hàm vẽ đồ thị WMA RSI
def plot_rsi_chart(df, rsi_upper=70, rsi_lower=30):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema9_rsi'], name='EMA9 RSI', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['wma45_rsi'], name='WMA45 RSI', line=dict(color='red')))
        fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", annotation_text=f"Quá Mua ({rsi_upper})")
        fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", annotation_text=f"Quá Bán ({rsi_lower})")
        fig.update_layout(
            title='Biểu Đồ WMA RSI',
            xaxis_title='Thời Gian',
            yaxis_title='WMA RSI',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ WMA RSI: {str(e)}")
        return None

# Hàm vẽ đồ thị MACD
def plot_macd_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal Line', line=dict(color='orange')))
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_histogram'], name='Histogram', marker_color='gray'))
        fig.update_layout(
            title='Biểu Đồ MACD',
            xaxis_title='Thời Gian',
            yaxis_title='MACD',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ MACD: {str(e)}")
        return None

# Hàm vẽ đồ thị ADX
def plot_adx_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], name='ADX', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['di_plus'], name='+DI', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['di_minus'], name='-DI', line=dict(color='red')))
        fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Mức Mạnh (25)")
        fig.update_layout(
            title='Biểu Đồ Chỉ Số Định Hướng Trung Bình (ADX)',
            xaxis_title='Thời Gian',
            yaxis_title='ADX',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ ADX: {str(e)}")
        return None

# Hàm vẽ đồ thị OBV
def plot_obv_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['obv'], name='OBV', line=dict(color='purple')))
        fig.update_layout(
            title='Biểu Đồ Khối Lượng Cân Bằng (OBV)',
            xaxis_title='Thời Gian',
            yaxis_title='OBV',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ OBV: {str(e)}")
        return None

# Hàm vẽ đồ thị Volume với MA19
def plot_volume_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Khối Lượng', marker_color='gray'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volume_ma19'], name='MA19 Khối Lượng', line=dict(color='orange')))
        fig.update_layout(
            title='Biểu Đồ Khối Lượng Giao Dịch với MA19',
            xaxis_title='Thời Gian',
            yaxis_title='Khối Lượng',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ Volume: {str(e)}")
        return None

# Hàm vẽ đồ thị EMA
def plot_ema_chart(df):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Giá Đóng Cửa', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema12'], name='EMA12', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema26'], name='EMA26', line=dict(color='green')))
        fig.update_layout(
            title='Biểu Đồ Giá với EMA12 và EMA26',
            xaxis_title='Thời Gian',
            yaxis_title='Giá (USDT)',
            xaxis_rangeslider_visible=False
        )
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ EMA: {str(e)}")
        return None

# Hàm xử lý một cặp giao dịch cho tính năng rà soát
def process_pair(symbol, timeframe, limit, weights, rsi_period=14, wma_period=45, ema_period=9, rsi_upper=70, rsi_lower=30):
    try:
        result = generate_recommendation(symbol, timeframe, weights, rsi_period, wma_period, ema_period, rsi_upper, rsi_lower)
        if result is not None:
            metrics = calculate_trading_metrics(
                result['current_price'], result['stop_loss'], result['take_profit'],
                1000, 1, result['recommendation']
            )
            sentiment = fetch_social_sentiment(symbol)
            if abs(result['support'] - result['resistance']) < result['current_price'] * 0.01:
                logging.warning(f"Cặp {symbol} có support/resistance quá gần nhau")
                return None
            return {
                'symbol': symbol,
                'recommendation': result['recommendation'],
                'total_score': result['total_score'],
                'current_price': result['current_price'],
                'support': result['support'],
                'resistance': result['resistance'],
                'stop_loss': result['stop_loss'],
                'take_profit': result['take_profit'],
                'rr_ratio': metrics['rr_ratio'],
                'sentiment': sentiment['sentiment'],
                'sentiment_score': sentiment['sentiment_score'],
                'signals': result['signals']
            }
        return None
    except Exception as e:
        logging.error(f"Lỗi xử lý cặp {symbol}: {str(e)}")
        return None

# Hàm rà soát các cặp giao dịch với xử lý song song
@st.cache_data(ttl=300)
def scan_trading_pairs(timeframe='1h', limit=100, weights={'current': 0.5, 'larger': 0.3, 'smaller': 0.2}, rsi_period=14, wma_period=45, ema_period=9, rsi_upper=70, rsi_lower=30):
    try:
        global TRADING_PAIRS
        TRADING_PAIRS = initialize_exchange_and_symbols()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(process_pair, symbol, timeframe, limit, weights, rsi_period, wma_period, ema_period, rsi_upper, rsi_lower): symbol for symbol in TRADING_PAIRS}
            for future in concurrent.futures.as_completed(future_to_symbol):
                result = future.result()
                if result is not None:
                    results.append(result)
        sorted_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
        top_25 = sorted_results[:25]
        bottom_25 = sorted_results[-25:] if len(sorted_results) >= 25 else sorted_results
        final_results = top_25 + bottom_25
        final_results = sorted(final_results, key=lambda x: x['total_score'], reverse=True)
        logging.info(f"Rà soát hoàn tất: {len(final_results)} cặp được xử lý (tối đa 25 mua + 25 bán)")
        if len(final_results) < 50:
            logging.warning(f"Chỉ tìm thấy {len(final_results)} cặp hợp lệ, ít hơn 50 cặp mong muốn")
        return final_results
    except Exception as e:
        logging.error(f"Lỗi rà soát các cặp giao dịch: {str(e)}")
        st.error(f"Lỗi rà soát: {str(e)}")
        return []

# Giao diện Streamlit
def main():
    try:
        st.set_page_config(page_title="Phân Tích Kỹ Thuật Tiền Điện Tử", layout="wide")
        st.title("Phân Tích Kỹ Thuật Tiền Điện Tử")
        logging.info("Ứng dụng Streamlit khởi động")
        
        # Khởi tạo session_state cho các tham số
        if 'symbol' not in st.session_state:
            st.session_state['symbol'] = 'BTC/USDT'
        if 'timeframe' not in st.session_state:
            st.session_state['timeframe'] = '1h'
        if 'investment' not in st.session_state:
            st.session_state['investment'] = 1000.0
        if 'leverage' not in st.session_state:
            st.session_state['leverage'] = 1
        if 'last_update' not in st.session_state:
            st.session_state['last_update'] = None
        if 'weight_current' not in st.session_state:
            st.session_state['weight_current'] = 0.5
        if 'weight_larger' not in st.session_state:
            st.session_state['weight_larger'] = 0.3
        if 'weight_smaller' not in st.session_state:
            st.session_state['weight_smaller'] = 0.2
        if 'rsi_period' not in st.session_state:
            st.session_state['rsi_period'] = 14
        if 'wma_period' not in st.session_state:
            st.session_state['wma_period'] = 45
        if 'ema_period' not in st.session_state:
            st.session_state['ema_period'] = 9
        if 'rsi_upper' not in st.session_state:
            st.session_state['rsi_upper'] = 70.0
        if 'rsi_lower' not in st.session_state:
            st.session_state['rsi_lower'] = 30.0
        
        # Giao diện sidebar
        st.sidebar.header("Cài Đặt")
        st.session_state['symbol'] = st.sidebar.selectbox(
            "Chọn Cặp Giao Dịch", TRADING_PAIRS, index=TRADING_PAIRS.index(st.session_state['symbol']) if st.session_state['symbol'] in TRADING_PAIRS else 0
        )
        st.session_state['timeframe'] = st.sidebar.selectbox(
            "Chọn Khung Thời Gian", TIMEFRAMES, index=TIMEFRAMES.index(st.session_state['timeframe']) if st.session_state['timeframe'] in TIMEFRAMES else 0
        )
        st.session_state['investment'] = st.sidebar.number_input(
            "Số Tiền Đầu Tư (USD)", min_value=10.0, value=st.session_state['investment'], step=10.0
        )
        st.session_state['leverage'] = st.sidebar.selectbox(
            "Chọn Tỷ Lệ Đòn Bẩy", LEVERAGE_OPTIONS, index=LEVERAGE_OPTIONS.index(st.session_state['leverage']) if st.session_state['leverage'] in LEVERAGE_OPTIONS else 0
        )
        st.sidebar.subheader("Trọng Số Phân Tích Đa Khung")
        st.session_state['weight_current'] = st.sidebar.number_input(
            "Trọng Số Khung Hiện Tại", min_value=0.0, max_value=1.0, value=st.session_state['weight_current'], step=0.1
        )
        st.session_state['weight_larger'] = st.sidebar.number_input(
            "Trọng Số Khung Lớn Hơn", min_value=0.0, max_value=1.0, value=st.session_state['weight_larger'], step=0.1
        )
        st.session_state['weight_smaller'] = st.sidebar.number_input(
            "Trọng Số Khung Nhỏ Hơn", min_value=0.0, max_value=1.0, value=st.session_state['weight_smaller'], step=0.1
        )
        
        # Thêm giao diện cấu hình WMA RSI
        st.sidebar.subheader("Cấu Hình Chỉ Báo WMA RSI")
        st.session_state['rsi_period'] = st.sidebar.number_input(
            "Chu Kỳ RSI", min_value=1, max_value=100, value=st.session_state['rsi_period'], step=1
        )
        st.session_state['wma_period'] = st.sidebar.number_input(
            "Chu Kỳ WMA", min_value=1, max_value=100, value=st.session_state['wma_period'], step=1
        )
        st.session_state['ema_period'] = st.sidebar.number_input(
            "Chu Kỳ EMA", min_value=1, max_value=100, value=st.session_state['ema_period'], step=1
        )
        st.session_state['rsi_upper'] = st.sidebar.number_input(
            "Ngưỡng Trên RSI", min_value=0.0, max_value=100.0, value=st.session_state['rsi_upper'], step=1.0
        )
        st.session_state['rsi_lower'] = st.sidebar.number_input(
            "Ngưỡng Dưới RSI", min_value=0.0, max_value=100.0, value=st.session_state['rsi_lower'], step=1.0
        )
        
        # Kiểm tra tổng trọng số
        total_weight = st.session_state['weight_current'] + st.session_state['weight_larger'] + st.session_state['weight_smaller']
        if abs(total_weight - 1.0) > 0.01:
            st.sidebar.warning(f"Tổng trọng số phải bằng 1. Hiện tại: {total_weight:.2f}. Vui lòng điều chỉnh!")
        
        # Kiểm tra RSI upper và lower
        if st.session_state['rsi_upper'] <= st.session_state['rsi_lower']:
            st.sidebar.warning("Ngưỡng trên RSI phải lớn hơn ngưỡng dưới!")
        
        # Thêm nút Cập Nhật Phân Tích và Rà Soát Cặp Giao Dịch
        update_button = st.sidebar.button("Cập Nhật Phân Tích")
        scan_button = st.sidebar.button("Rà Soát Cặp Giao Dịch")
        
        st.write("Chào mừng! Nhấn 'Cập Nhật Phân Tích' để xem phân tích cho cặp giao dịch hoặc 'Rà Soát Cặp Giao Dịch' để xem 25 cặp có tín hiệu mua mạnh nhất và 25 cặp có tín hiệu bán mạnh nhất.")
        
        weights = {
            'current': st.session_state['weight_current'],
            'larger': st.session_state['weight_larger'],
            'smaller': st.session_state['weight_smaller']
        }
        
        if scan_button:
            st.subheader("Danh Mục 50 Cặp Giao Dịch Phổ Biến Nhất (25 Mua Mạnh Nhất + 25 Bán Mạnh Nhất)")
            with st.spinner("Đang rà soát 100 cặp giao dịch phổ biến nhất..."):
                scan_results = scan_trading_pairs(
                    timeframe=st.session_state['timeframe'],
                    weights=weights,
                    rsi_period=st.session_state['rsi_period'],
                    wma_period=st.session_state['wma_period'],
                    ema_period=st.session_state['ema_period'],
                    rsi_upper=st.session_state['rsi_upper'],
                    rsi_lower=st.session_state['rsi_lower']
                )
                if scan_results:
                    summary_data = []
                    for result in scan_results:
                        summary_data.append({
                            'Cặp Giao Dịch': result['symbol'],
                            'Khuyến Nghị': result['recommendation'],
                            'Tổng Trọng Số': result['total_score'],
                            'Giá Hiện Tại': f"${result['current_price']:.2f}",
                            'Hỗ Trợ': f"${result['support']:.2f}",
                            'Kháng Cự': f"${result['resistance']:.2f}",
                            'Cắt Lỗ': f"${result['stop_loss']:.2f}",
                            'Chốt Lời': f"${result['take_profit']:.2f}",
                            'Tỷ Lệ RR': f"{result['rr_ratio']:.2f}",
                            'Cảm Xúc Thị Trường': result['sentiment'],
                            'Điểm Cảm Xúc': result['sentiment_score']
                        })
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    if len(scan_results) < 50:
                        st.warning(f"Chỉ tìm thấy {len(scan_results)} cặp giao dịch hợp lệ, ít hơn 50 cặp mong muốn. Vui lòng kiểm tra error.log để biết thêm chi tiết.")
                    
                    for result in scan_results:
                        st.write(f"### Chi Tiết Tín Hiệu: {result['symbol']}")
                        signals_df = pd.DataFrame(result['signals'], columns=['Hành Động', 'Lý Do', 'Trọng Số'])
                        st.dataframe(signals_df, use_container_width=True)
                else:
                    st.warning("Không thể rà soát được cặp giao dịch nào. Vui lòng kiểm tra kết nối hoặc file error.log.")
        
        if update_button:
            with st.spinner(f"Đang phân tích {st.session_state['symbol']}..."):
                result = generate_recommendation(
                    st.session_state['symbol'],
                    st.session_state['timeframe'],
                    weights,
                    rsi_period=st.session_state['rsi_period'],
                    wma_period=st.session_state['wma_period'],
                    ema_period=st.session_state['ema_period'],
                    rsi_upper=st.session_state['rsi_upper'],
                    rsi_lower=st.session_state['rsi_lower']
                )
                if result is not None:
                    trading_metrics = calculate_trading_metrics(
                        result['current_price'], result['stop_loss'], result['take_profit'],
                        st.session_state['investment'], st.session_state['leverage'], result['recommendation']
                    )
                    sentiment = fetch_social_sentiment(st.session_state['symbol'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Giá Hiện Tại", f"${result['current_price']:.2f}")
                        st.metric("Hỗ Trợ", f"${result['support']:.2f}")
                        st.metric("Kháng Cự", f"${result['resistance']:.2f}")
                    with col2:
                        st.metric("Khuyến Nghị", result['recommendation'])
                        st.metric("Tổng Trọng Số", result['total_score'])
                        st.metric("Cắt Lỗ (Stop-Loss)", f"${result['stop_loss']:.2f}")
                        st.metric("Chốt Lời (Take-Profit)", f"${result['take_profit']:.2f}")
                    with col3:
                        st.metric("Lãi Dự Kiến (Take-Profit)", f"${trading_metrics['profit']:.2f}")
                        st.metric("Lỗ Dự Kiến (Stop-Loss)", f"-${trading_metrics['loss']:.2f}")
                        st.metric("Giá Call Margin", f"${trading_metrics['margin_call_price']:.2f}")
                        st.metric("Tỷ Lệ RR", f"{trading_metrics['rr_ratio']:.2f}")
                        st.metric("Đòn Bẩy Đề Xuất", f"{trading_metrics['max_leverage']:.2f}x")
                        st.metric("Cảm Xúc Thị Trường", sentiment['sentiment'])
                        st.metric("Điểm Cảm Xúc", sentiment['sentiment_score'])
                    
                    st.subheader("Phân Tích Vĩ Mô (Kinh Tế Mỹ)")
                    macro_data = get_us_macro_data()
                    macro_df, macro_score, macro_summary = evaluate_macro_data(macro_data)
                    st.dataframe(macro_df, use_container_width=True)
                    st.info(f"**Tổng Đánh Giá Vĩ Mô:** {macro_summary}")

                    st.subheader("Tín Hiệu Chỉ Báo (Tổng Hợp)")
                    signals_df = pd.DataFrame(result['signals'], columns=['Hành Động', 'Lý Do', 'Trọng Số'])
                    st.dataframe(signals_df, use_container_width=True)
                    
                    st.subheader("Biểu Đồ Phân Tích (Khung Thời Gian Hiện Tại)")
                    st.write("### Biểu Đồ Giá với Dải Bollinger")
                    fig_price = plot_price_chart(result['df'])
                    if fig_price is not None:
                        st.plotly_chart(fig_price, use_container_width=True)
                    
                    st.write("### Biểu Đồ WMA RSI")
                    fig_rsi = plot_rsi_chart(result['df'], rsi_upper=st.session_state['rsi_upper'], rsi_lower=st.session_state['rsi_lower'])
                    if fig_rsi is not None:
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    st.write("### Biểu Đồ MACD")
                    fig_macd = plot_macd_chart(result['df'])
                    if fig_macd is not None:
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    st.write("### Biểu Đồ ADX")
                    fig_adx = plot_adx_chart(result['df'])
                    if fig_adx is not None:
                        st.plotly_chart(fig_adx, use_container_width=True)
                    
                    st.write("### Biểu Đồ Khối Lượng Cân Bằng (OBV)")
                    fig_obv = plot_obv_chart(result['df'])
                    if fig_obv is not None:
                        st.plotly_chart(fig_obv, use_container_width=True)
                    
                    st.write("### Biểu Đồ Khối Lượng Giao Dịch với MA19")
                    fig_volume = plot_volume_chart(result['df'])
                    if fig_volume is not None:
                        st.plotly_chart(fig_volume, use_container_width=True)
                    
                    st.write("### Biểu Đồ EMA")
                    fig_ema = plot_ema_chart(result['df'])
                    if fig_ema is not None:
                        st.plotly_chart(fig_ema, use_container_width=True)
                else:
                    st.error(f"Không thể tạo khuyến nghị cho {st.session_state['symbol']}")
            
            st.session_state['last_update'] = time.time()
        
    except Exception as e:
        logging.error(f"Lỗi ứng dụng Streamlit: {str(e)}")
        st.error(f"Lỗi ứng dụng: {str(e)}")

if __name__ == "__main__":
    main()