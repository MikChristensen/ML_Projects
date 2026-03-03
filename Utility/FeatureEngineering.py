import pandas as pd
import numpy as np
import quantreo.features_engineering as fe
from Utility.FeaturesLibrary import *
import pandas_ta as ta
from hurst import compute_Hc

class FeatureEngineering:
    def __init__(self):
        pass

    def get_data_resampled(self, df, high_freq='4h'):
        
        df_high_sample_rate = df.resample(high_freq).agg(
            open = ('open', 'first'),
            high = ('high', 'max'),
            low = ('low', 'min'),
            close = ('close', 'last'),
            volume = ('volume', 'sum'),
            high_time=('high', lambda x: x.idxmax() if len(x) > 0 else None),
            low_time=('low', lambda x: x.idxmin() if len(x) > 0 else None)
        )
        return df_high_sample_rate
        

    def get_features(self, df, high_freq='4h' ):
        df = df.copy()
        high_data = self.get_data_resampled(df, high_freq).dropna()
        high_data = self.get_intra_bar_features(df, high_data, high_freq).dropna()
        high_data = self.get_inter_bar_features(high_data).dropna()
        high_data = self.get_over_bar_features(high_data).dropna()

        high_data = high_data.replace([np.inf, -np.inf], np.nan)
        high_data = high_data.dropna()

        return high_data

    def get_intra_bar_features(self,low_data, high_data, high_freq='4h'):
        df_high = high_data.copy()
        df_low = low_data.copy()

        df_high['hurst'] = fe.math.hurst(df=df_high, col="close", window_size=200)
        
        
        # Basic intra-bar features
        df_high['range'] = df_high['high'] - df_high['low']
        df_high['body'] = df_high['close'] - df_high['open']
        df_high['upper_shadow'] = df_high['high'] - df_high[['close', 'open']].max(axis=1)
        df_high['lower_shadow'] = df_high[['close', 'open']].min(axis=1) - df_high['low']
        df_high['close_open_diff'] = df_high['close'] - df_high['open']
        df_high['close_high_low_ratio'] = (df_high['close'] - df_high['low']) / (df_high['high'] - df_high['low'])

        # Calculate the percentage of closing prices within specified ranges for each 4-hour interval
        df_high['0_to_20'] = df_low.resample(high_freq).apply(lambda x: apply_close_percentage_in_range(x, 0.0, 0.2))
        df_high['20_to_40'] = df_low.resample(high_freq).apply(lambda x: apply_close_percentage_in_range(x, 0.2, 0.4))
        df_high['40_to_60'] = df_low.resample(high_freq).apply(lambda x: apply_close_percentage_in_range(x, 0.4, 0.6))
        df_high['60_to_80'] = df_low.resample(high_freq).apply(lambda x: apply_close_percentage_in_range(x, 0.6, 0.8))
        df_high['80_to_100'] = df_low.resample(high_freq).apply(lambda x: apply_close_percentage_in_range(x, 0.8, 1.0))

        # Calculate the slope of the linear regression for each 4-hour interval
        df_high['linear_slope'] = df_low.resample(high_freq).apply(apply_linear_regression_slope)

        df_high['linear_slope_last_25'] = df_low.resample(high_freq).apply(apply_linear_regression_slope_last_25)


        #df_high["hurst"] = fe.math.hurst(df=df_high, col="close", window_size=window)
        df_high["candle_way"], df_high["filling"], df_high["amplitude"] = fe.candle.candle_information(df=df_high, open_col="open", high_col="high",
                                                                                low_col="low", close_col="close")
        


        return df_high

    def get_inter_bar_features(self, data):
        df = data.copy()

        df.ta.wma(append=True)
        df.ta.vhf(append=True)
        df.ta.variance(append=True)
        df.ta.vidya(append=True)
        df.ta.uo(append=True)
        df.ta.ui(append=True)
        df.ta.tsi(append=True)
        df.ta.trix(append=True)
        df.ta.stochrsi(append=True)
        df.ta.stc(append=True)
        df.ta.slope(append=True)
        df.ta.skew(append=True)
        df.ta.rvi(append=True)
        df.ta.qstick(append=True)
        df.ta.psl(append=True)
        df.ta.pgo(append=True)
        df.ta.nvi(append=True)
        df.ta.increasing(append=True)
        df.ta.er(append=True)
        df.ta.efi(append=True)
        df.ta.ebsw(append=True)
        df.ta.dpo(append=True)
        df.ta.cti(append=True)
        df.ta.cfo(append=True)
        df.ta.cci(append=True)
        df.ta.bop(append=True)
        df.ta.bbands(append=True)
        df.ta.aobv(append=True)
        df.ta.aroon(append=True)
        df.ta.amat(append=True)
        df.ta.adosc(append=True)
        
        df['moving_range_volatility'] = self.moving_range_volatility(df = df, window_size = 14, close_col = 'close')
        df['normalized_volume_spread'] = self.normalized_volume_spread(df = df , window_size = 20, volume_col = 'volume', threshold = 1.0)
        df['exponential_moving_momentum'] = self.exponential_moving_momentum(df = df, close_col = 'close', alpha = 0.05)
        df['mean_reversion_momentum'] = self.mean_reversion_momentum(df = df, window_size = 20, close_col = 'close')
        df['mean_reversion_bollinger_band_width'] = self.mean_reversion_bollinger_band_width(df = df, window_size = 20, std_dev = 2, close_col = 'close')
        df['volume_imbalance_ratio'] = self.volume_imbalance_ratio(df = df, window_size = 10, close_col = 'close', volume_col = 'volume')
        df['volume_mean_reversion'] = self.volume_mean_reversion(df = df, window_size = 10, volume_col = 'volume') 
        df['lagged_volume_concentration'] = self.lagged_volume_concentration(df = df, window_size = 10, volume_col = 'volume')
        df['bidask_spread_ratio'] = self.bidask_spread_ratio(df = df, window_size = 20, high_col = 'high', low_col = 'low', close_col = 'close')
        df['trend_acceleration'] = self.trend_acceleration(df = df, close_col = 'close')
        df['liquidity_stress'] = self.liquidity_stress(df = df, high_col = 'high', low_col = 'low')
        df['range_expansion_ratio'] = self.range_expansion_ratio(df = df, window_size = 20, high_col = 'high', low_col = 'low', close_col = 'close')
        df['range_acceleration'] = self.range_acceleration(df = df, high_col = 'high', low_col = 'low', window_size = 20) 
        df['regime_shift_detection'] = self.regime_shift_detection(df = df, high_col = 'high', low_col = 'low', window_size = 20, threshold = 0.05)
        df['regime_shifts'] = self.regime_shifts(df = df, close_col = 'close', threshold = 1.5) 
        df['exponential_moving_volatility_ratio'] = self.exponential_moving_volatility_ratio(df = df, window_size = 20, ratio_window_size = 50, close_col = 'close', open_col = 'open') 
        df['rolling_correlation'] = self.rolling_correlation(df = df, window_size = 20, min_periods = 5, lag = 1, close_col1 = 'close', close_col2 = 'close')
        df['moving_average_convergence_divergence'] = self.moving_average_convergence_divergence(df = df, fast_ma_window = 12, slow_ma_window = 26, signal_ma_window = 9, close_col = 'close')
        df['standardized_price_action_range'] = self.standardized_price_action_range(df = df, atr_length = 14, high_col = 'high', low_col = 'low', close_col = 'close')
        df['realized_volatility_decay'] = self.realized_volatility_decay(df = df, window_size = 30, close_col = 'close')
        df['double_smooth_ewm_momentum'] = self.double_smooth_ewm_momentum(df = df, close_col = 'close', alpha = 0.5)
        df['exponential_moving_average_difference'] = self.exponential_moving_average_difference(df = df, short_window = 10, long_window = 50, close_col = 'close') 
        
        self.create_lag(df_code = df, n_lag=[3, 7, 14, 21], shift_size=1)


        windowses = range(10, 100, 10)

        df['returns'] = df["close"].pct_change(1)
        df["velocity"], df["acceleration"] = fe.math.derivatives(df=df, col="close")
        df["adf_stat"], df["adf_pvalue"] = fe.math.adf_test(df, col="close", window_size=80, lags=10, regression="ct")
        df["arch_stat"], df["arch_pvalue"] = fe.math.arch_test(df, col="returns", window_size=60, lags=10)
        df["savgol"] = fe.transformation.savgol_filter(df=df, col="close", window_size=21, polyorder=2)
        df["spectral_entropy"] = fe.math.spectral_entropy(df=df, col="close", window_size=60)
        df[f"sample_entropy"] = fe.math.sample_entropy(df=df, col="close", window_size=60, order=3)
        df[f"linear_slope_1M"] = fe.trend.linear_slope(df, col='close', window_size=30*6)
        for window in windowses:    
            df[f'return_{window}'] = df["close"].pct_change(window)
            df[f"log_pct_{window}"] = fe.math.log_pct(df=df, col="close", window_size=window)  
            df[f'abs_returns_{window}'] = abs(df["close"].pct_change(window))
            df[f"skew_{window}"] = fe.math.skewness(df=df, col="returns", window_size=window)
            df[f"kurt_{window}"] = fe.math.kurtosis(df=df, col="returns", window_size=window)      
            df[f"permutation_entropy_{window}"] = fe.math.permutation_entropy(df=df, col="close", window_size=window, order=5)
            df[f"petrosian_fd_{window}"] = fe.math.petrosian_fd(df=df, col="close", window_size=window)
            df[f"tail_index_{window}"] = fe.math.tail_index(df=df, col=f'abs_returns_{window}', window_size=window, k_ratio=0.10)
            df[f"sw_stat_{window}"], df["sw_pvalue_{window}"] = fe.math.shapiro_wilk(df, col="returns", window_size=window)
            df[f"fisher_{window}"] = fe.transformation.fisher_transform(df=df, high_col="high", low_col="low", window_size=window)            
            #df[f"sma_{window}"] = fe.trend.sma(df=df, col="close", window_size=window)
            df[f"kama_{window}"] = fe.trend.kama(df=df, col="close", l1=10, l2=2, l3=window)
            df[f"ctc_vol_{window}"] = fe.volatility.close_to_close_volatility(df=df, close_col="close", window_size=window)
            df[f"parkinson_vol_{window}"] = fe.volatility.parkinson_volatility(df=df, high_col="high", low_col="low", window_size=window)
            df[f"rs_vol_{window}"] = fe.volatility.rogers_satchell_volatility(df=df, high_col="high", low_col="low", open_col="open", close_col="close", window_size=window)
            df[f"yz_vol_{window}"] = fe.volatility.yang_zhang_volatility(df=df, low_col="low",high_col="high", open_col="open", close_col="close", window_size=window)

        df = self.derivatives(df,"close")  
        
        df["kama_trend"] = fe.market_regime.kama_market_regime(df, "close", l1_fast=50, l2_fast=2, l3_fast=30,
                                                       l1_slow=200, l2_slow=2, l3_slow=30)
        df = spread(df)
        df = kama_market_regime(df, "close", 30, 100)
        for i in [1,2,5,10,20,50]:
            df = auto_corr(df, "close", n=100, lag=i)
        for i in [1,2,5,10,20,50]:
            df = log_transform(df, "close", i)
        df = candle_information(df)
        df = moving_yang_zhang_estimator(df, 20)
        df.drop(['tail_index_10'], axis=1, inplace=True)
        return df

    def get_over_bar_features(self, data):
        df = data.copy()

        df['linear_slope_1M'] = df["close"].rolling(20*6).apply(linear_regression_slope_market_trend)

        df["kama_trend"] = fe.market_regime.kama_market_regime(df, "close", l1_fast=50, l2_fast=2, l3_fast=30,
                                                        l1_slow=200, l2_slow=2, l3_slow=30)

        return df
    
    def create_lag(self, df_code, n_lag=[3, 7, 14, ], shift_size=1):
        return_features = ['open', 'high', 'low', 'close', 'volume']
        for col in return_features:
            for window in n_lag:
                rolled = df_code[col].shift(shift_size).rolling(window=window)
                lag_mean = rolled.mean()
                lag_max = rolled.max()
                lag_min = rolled.min()
                lag_std = rolled.std()
                df_code['%s_lag_%s_mean' % (col, window)] = lag_mean
                df_code['%s_lag_%s_max' % (col, window)] = lag_max
                df_code['%s_lag_%s_min' % (col, window)] = lag_min

        return df_code.fillna(-1)

    def exponentially_weighted_volatility(self, df: pd.DataFrame, alpha: float = 0.94, close_col: str = 'close') -> pd.Series:
        """
        Compute exponentially weighted volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        alpha : float, optional
            Smoothing factor (default is 0.94).
        close_col : str, optional
            Name of the column containing closing prices (default is 'close').

        Returns
        -------
        pd.Series
            Exponentially weighted volatility series.
        """
        # Initialize ewv with zeros for the first row
        ewv = pd.Series(index=df.index, dtype=np.float64)
        ewv.iloc[0] = 0

        # Compute relative returns
        relative_returns = np.log(df[close_col] / df[close_col].shift(1))

        # Calculate exponentially weighted volatility
        for i in range(1, len(df)):
            # Compute relative return for current row
            rel_return = relative_returns.iloc[i]
            
            # Calculate exponentially weighted volatility using the formula
            if i == 1:
                ewv.iloc[i] = np.sqrt(alpha**2 * rel_return**2)
            else:
                ewv.iloc[i] = np.sqrt(alpha**2 * rel_return**2 + (1 - alpha**2) * ewv.iloc[i-1]**2)

        return ewv

    def moving_range_volatility(self, df: pd.DataFrame, window_size: int = 14, close_col: str = 'close') -> pd.Series:

        """
        Compute the Moving Range Volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the moving window (default is 14).
        close_col : str, optional
            Name of the column containing close prices (default is 'close').

        Returns
        -------
        pd.Series
            Moving Range Volatility series.
        """
        # Compute differences between consecutive close prices
        close_diff = df[close_col].diff()
        
        # For each window of size 'window_size', compute the maximum and minimum of these differences
        max_diff = close_diff.rolling(window=window_size).max()
        min_diff = close_diff.rolling(window=window_size).min()
        
        # Compute the Moving Range Volatility as the average of the ratio (max - min) / max
        mr_volatility = ((max_diff - min_diff) / max_diff).replace([np.inf, -np.inf], np.nan)
        
        return mr_volatility

    def exponentially_decay_volatility(self, df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', decay_rate: float = 0.95) -> pd.Series:
        """
        Compute exponentially decayed volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        high_col : str, optional
            high price column name (default is 'high').
        low_col : str, optional
            low price column name (default is 'low').
        decay_rate : float, optional
            Decay rate (default is 0.95).

        Returns
        -------
        pd.Series
            Exponentially decayed volatility values.
        """
        # Compute daily volatility values
        daily_volatility = df[high_col] - df[low_col]
        
        # Initialize exponentially decayed volatility values
        exponentially_decay_volatility_values = np.zeros(len(df))
        
        # Compute exponentially decayed volatility for each day
        for i in range(len(df)):
            # Compute sum of decayed volatility values for previous days
            decayed_volatility_sum = np.sum([daily_volatility[j] * (decay_rate ** (i - j)) if j < i else 0 for j in range(i+1)])
            exponentially_decay_volatility_values[i] = decayed_volatility_sum
        
        # Return exponentially decayed volatility values as a pandas Series
        return pd.Series(exponentially_decay_volatility_values)

    def normalized_volume_spread(self, df: pd.DataFrame, window_size: int = 20, volume_col: str = 'volume', threshold: float = 1.0) -> pd.Series:
        """
        Calculate the normalized volume spread.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the rolling window (default is 20).
        volume_col : str, optional
            Name of the column containing volume data (default is 'volume').
        threshold : float, optional
            Threshold value for the normalized volume spread (default is 1.0).

        Returns
        -------
        pd.Series
            Normalized volume spread.
        """
        # Calculate mean daily volume
        mean_volume = df[volume_col].rolling(window=window_size).mean()
        
        # Calculate standard deviation of daily volume
        std_volume = df[volume_col].rolling(window=window_size).std()
        
        # Compute volume spread as difference between mean and standard deviation
        volume_spread = mean_volume - std_volume
        
        # Normalize volume spread by mean volume
        normalized_spread = volume_spread / mean_volume
        
        # Apply threshold to normalized volume spread
        normalized_spread = np.where(np.abs(normalized_spread) > threshold, normalized_spread, 0)
        
        return normalized_spread

    def exponential_moving_momentum(self, df: pd.DataFrame, close_col: str = 'close', alpha: float = 0.05) -> pd.Series:
        """
        Compute rate of change of exponential moving average.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        close_col : str, default 'close'
            Name of the column containing closing prices.
        alpha : float, default 0.05
            Smoothing factor for exponential moving average.

        Returns
        -------
        pd.Series
            Rate of change of exponential moving average.
        """
        # Compute exponential moving average of close_col using alpha parameter
        ema = df[close_col].ewm(alpha=alpha, adjust=False).mean()
        
        # Compute rate of change of EMA by subtracting previous EMA from current EMA
        ema_diff = ema - ema.shift(1)
        
        # Divide rate of change by the current EMA to normalize the value
        ema_momentum = ema_diff / ema.shift(1)
        
        return ema_momentum

    def exponential_momentum(self, df: pd.DataFrame, window_size: int = 50, decay_rate: float = 0.2, close_col: str = 'close') -> pd.Series:
        """
        Compute exponential moving average of returns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the window (default is 50).
        decay_rate : float, optional
            Decay rate for the exponential moving average (default is 0.2).
        close_col : str, optional
            Name of the column containing closing prices (default is 'close').

        Returns
        -------
        pd.Series
            Exponential moving average of returns.
        """
        returns = df[close_col].pct_change()
        ema = returns.ewm(span=window_size, adjust=False, alpha=decay_rate).mean()
        return ema

    def mean_reversion_momentum(self, df: pd.DataFrame, window_size: int = 20, close_col: str = 'close') -> pd.Series:
        """
        Compute mean reversion momentum signal relative to the exponential moving average.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the window for calculations (default is 20).
        close_col : str, optional
            Name of the column containing close prices (default is 'close').

        Returns
        -------
        pd.Series
            Mean reversion momentum signal.
        """
        # Compute the exponential moving average of the close price
        ema = df[close_col].ewm(span=window_size, adjust=False).mean()
        
        # Compute the standard deviation of the close price over the last window_size periods
        std_dev = df[close_col].rolling(window=window_size).std()
        
        # Subtract the moving average from the close price and divide by the standard deviation
        momentum = (df[close_col] - ema) / std_dev
        
        # Multiply the result by 1.96 (assuming a normal distribution)
        mean_reversion_signal = momentum * 1.96
        
        return mean_reversion_signal

    def mean_reversion_bollinger_band_width(self, df: pd.DataFrame, window_size: int = 20, std_dev: int = 2, close_col: str = 'close') -> pd.Series:
        """
        Compute Bollinger Band width.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Window size (default is 20).
        std_dev : int, optional
            Standard deviation (default is 2).
        close_col : str, optional
            close column name (default is 'close').

        Returns
        -------
        pd.Series
            Bollinger Band width.
        """
        # Compute rolling mean and standard deviation for 'close_col'
        rolling_mean = df[close_col].rolling(window=window_size).mean()
        rolling_std = df[close_col].rolling(window=window_size).std()
        
        # Calculate the band width as the moving average of the rolling standard deviation divided by the moving average of the rolling mean
        band_width = (rolling_std * std_dev) / rolling_mean
        
        return band_width

    def volume_imbalance_ratio(self, df: pd.DataFrame, window_size: int = 10, close_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
        """
        Compute the volume imbalance ratio.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Not used in this function (default is 10).
        close_col : str, optional
            Not used in this function (default is 'close').
        volume_col : str, optional
            Column name for volume data (default is 'volume').

        Returns
        -------
        pd.Series
            volume imbalance ratio.
        """
        # Compute current and previous volume ratios
        current_volume_ratio = df[volume_col] / df[volume_col].shift(1)
        previous_volume_ratio = 1 / current_volume_ratio

        # Calculate volume imbalance ratio using the formula
        volume_imbalance = (current_volume_ratio - previous_volume_ratio) / (current_volume_ratio + previous_volume_ratio)

        # Return volume imbalance ratio as a pandas Series
        return volume_imbalance

    def volume_mean_reversion(self, df: pd.DataFrame, window_size: int = 10, volume_col: str = 'volume') -> pd.Series:
        """
        Calculate the volume mean reversion strength.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the volume data.
        window_size : int, optional
            Size of the rolling window (default is 10).
        volume_col : str, optional
            Name of the column containing the volume data (default is 'volume').

        Returns
        -------
        pd.Series
            volume mean reversion strength.
        """
        # Calculate average volume over the given window
        avg_volume = df[volume_col].rolling(window=window_size).mean()
        
        # Compute normalized volume
        normalized_volume = df[volume_col] / avg_volume
        
        # Center the data around zero
        centered_volume = normalized_volume - 1
        
        # Sum up the centered volume values over the window
        sum_centered_volume = centered_volume.rolling(window=window_size).sum()
        
        # Divide the sum by the number of periods to obtain the mean reversion strength
        mean_reversion_strength = sum_centered_volume / window_size
        
        # Scale the result by a factor to obtain a final value
        volume_mean_reversion = -mean_reversion_strength
        
        return volume_mean_reversion

    def lagged_volume_concentration(self, df: pd.DataFrame, window_size: int = 10, volume_col: str = 'volume') -> pd.Series:
        """
        Calculate the proportion of trading activity in the lagged period.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Lag window size (default is 10).
        volume_col : str, optional
            volume column name (default is 'volume').

        Returns
        -------
        pd.Series
            Proportion of trading activity in the lagged period.
        """
        # Calculate the average volume over the lagged window
        avg_lag_volume = df[volume_col].rolling(window_size).mean().shift(1)
        
        # Compute the proportion: lagged_volume_concentration
        lagged_volume_concentration = avg_lag_volume / (avg_lag_volume + df[volume_col])
        
        return lagged_volume_concentration

    def bidask_spread_ratio(self, df: pd.DataFrame, window_size: int = 20, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series:
        """
        Calculate the ratio of average bid-ask spread to the absolute value of the average return.

        Parameters
        ----------
        df : pd.DataFrame
        window_size : int, optional
        high_col : str, optional
        low_col : str, optional
        close_col : str, optional

        Returns
        -------
        pd.Series
        """
        # Calculate simple returns
        simple_returns = (df[close_col] - df[close_col].shift(1)) / df[close_col].shift(1)
        
        # Calculate absolute value of simple returns
        abs_simple_returns = np.abs(simple_returns)
        
        # Calculate average bid-ask spread
        avg_bidask_spread = (df[high_col] - df[low_col]) / 2
        
        # Compute ratio of average bid-ask spread to absolute value of simple returns
        ratio = avg_bidask_spread / abs_simple_returns
        
        # Apply moving average with specified window size
        return ratio.rolling(window=window_size).mean()

    def trend_contraction(self, df: pd.DataFrame, window_size: int = 5, threshold: float = 0.5, close_col: str = 'close') -> pd.Series:
        """
        Compute trend contraction signal.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Contraction window size (default is 5).
        threshold : float, optional
            Contraction threshold (default is 0.5).
        close_col : str, optional
            close price column name (default is 'close').

        Returns
        -------
        pd.Series
            Trend contraction signal.
        """
        # Compute close returns over the contraction window
        returns = df[close_col].pct_change(window=window_size)
        
        # Calculate trend contraction
        trend_contraction = returns
        
        # Compare the trend contraction to the threshold and assign signals accordingly
        signal = np.where(trend_contraction < -threshold, -1, np.where(trend_contraction > threshold, 1, 0))
        
        return pd.Series(signal)

    def trend_acceleration(self, df: pd.DataFrame, close_col: str = 'close') -> pd.Series:
        """
        Compute trend acceleration.

        Parameters
        ----------
        df : pd.DataFrame
        close_col : str, optional
            The column name for closing prices.

        Returns
        -------
        pd.Series
            The computed trend acceleration.
        """
        # Compute daily returns
        daily_returns = df[close_col].pct_change()
        
        # Compute acceleration by subtracting previous return from current return
        acceleration = daily_returns - daily_returns.shift(1)
        
        # Compute daily difference of the acceleration
        trend_acceleration = acceleration - acceleration.shift(1)
        
        return trend_acceleration

    def liquidity_stress(self, df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low') -> pd.Series:
        """
        Compute liquidity stress index.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        high_col : str, optional
            high column name (default is 'high').
        low_col : str, optional
            low column name (default is 'low').

        Returns
        -------
        pd.Series
            Liquidity stress index values.
        """
        # Calculate high-low range at current time step
        current_range = df[high_col] - df[low_col]
        
        # Calculate high-low range at previous time step
        previous_range = (df[high_col].shift(1) - df[low_col].shift(1))
        
        # Divide current range by previous range to get liquidity stress ratio
        liquidity_stress_ratio = current_range / previous_range
        
        return liquidity_stress_ratio

    def liquidity_stress_level(self, df: pd.DataFrame, window_size: int = 20, ask_col: str = 'ask', bid_col: str = 'bid', threshold: float = 0.5) -> pd.Series:
        """
        Calculate liquidity stress level based on the rate of change of the bid-ask spread.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Window size for calculating the rate of change (default is 20).
        ask_col : str, optional
            Name of the ask price column (default is 'ask').
        bid_col : str, optional
            Name of the bid price column (default is 'bid').
        threshold : float, optional
            Threshold for determining the liquidity stress level (default is 0.5).

        Returns
        -------
        pd.Series
            Liquidity stress level at each time point.
        """
        # Compute the rate of change of ask and bid prices
        delta_ask = (df[ask_col] - df[ask_col].shift(window_size)) / df[ask_col].shift(window_size)
        delta_bid = (df[bid_col] - df[bid_col].shift(window_size)) / df[bid_col].shift(window_size)

        # Calculate the maximum rate of change
        max_delta = np.maximum(delta_ask, delta_bid)

        # Divide by the average of the ask and bid prices
        avg_price = (df[ask_col] + df[bid_col]) / 2
        liquidity_stress = max_delta / avg_price

        # Compare the result to the threshold to determine the liquidity stress level
        return (liquidity_stress > threshold).astype(int)

    def intraday_seasonality_imbalance(self, df: pd.DataFrame, 
                                    window_size: int = 30, 
                                    lag: int = 30, 
                                    threshold: float = 0.1, 
                                    open_col: str = 'open', 
                                    close_col: str = 'close') -> pd.Series:
        """
        Detects intraday seasonality imbalance signals by analyzing autocorrelation patterns of intraday returns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the moving window (default is 30).
        lag : int, optional
            Lag for the autocorrelation function (default is 30).
        threshold : float, optional
            Threshold for the absolute difference (default is 0.1).
        open_col : str, optional
            Name of the 'open' column (default is 'open').
        close_col : str, optional
            Name of the 'close' column (default is 'close').

        Returns
        -------
        pd.Series
            Binary signal indicating potential seasonality imbalance (1) or not (0).
        """
        # Compute intraday returns
        returns = np.log(df[close_col]) - np.log(df[open_col])
        
        # Compute autocorrelation function at lag t using a moving window
        autocorr_t = returns.rolling(window=window_size).apply(lambda x: x.autocorr(lag=0), raw=False)
        autocorr_t_lag = returns.rolling(window=window_size).apply(lambda x: x.autocorr(lag=lag), raw=False)
        
        # Calculate the absolute difference between C(t) and C(t + lag)
        abs_diff = np.abs(autocorr_t - autocorr_t_lag)
        
        # Compare the absolute difference to the threshold
        signal = np.where(abs_diff > threshold, 1, 0)
        
        return pd.Series(signal)

    def intraday_seasonal_shift(self, df: pd.DataFrame, col: str = 'close', time_interval: int = 60) -> pd.Series:
        """
        Compute the intraday seasonal shift signal.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing time series data.
        col : str, optional
            Column name for the time series data (default is 'close').
        time_interval : int, optional
            Time interval in minutes for resampling (default is 60).

        Returns
        -------
        pd.Series
            Intraday seasonal shift signal.
        """
        # Resample data by time interval
        resampled_df = df[col].resample(f'{time_interval}min').agg(['mean', 'std'])
        
        # Compute price shift by subtracting mean price from actual price
        price_shift = df[col] - resampled_df['mean']
        
        # Standardize price shift by dividing by standard deviation
        standardized_shift = price_shift / resampled_df['std']
        
        # Sum up standardized price shifts across all intervals
        seasonal_shift_signal = standardized_shift.groupby(standardized_shift.index.time).sum()
        
        return seasonal_shift_signal

    def range_expansion_ratio(self, df: pd.DataFrame, window_size: int = 20, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series:
        """
        Compute the range expansion ratio.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the window (default is 20).
        high_col : str, optional
            Name of the column containing high values (default is 'high').
        low_col : str, optional
            Name of the column containing low values (default is 'low').
        close_col : str, optional
            Name of the column containing close values (default is 'close').

        Returns
        -------
        pd.Series
            Range expansion ratio.
        """
        # Compute current and previous price ranges
        current_range = df[high_col] - df[low_col]
        previous_range = current_range.shift(window_size)

        # Compute current and previous high-close and low-close ranges
        current_high_close_range = df[high_col] - df[close_col]
        previous_high_close_range = current_high_close_range.shift(window_size)
        current_low_close_range = df[close_col] - df[low_col]
        previous_low_close_range = current_low_close_range.shift(window_size)

        # Compute differences between current and previous ranges
        range_diff = current_range - previous_range
        high_close_range_diff = current_high_close_range - previous_high_close_range
        low_close_range_diff = current_low_close_range - previous_low_close_range

        # Compute range expansion ratio
        ratio = range_diff / np.abs(high_close_range_diff - low_close_range_diff)

        return ratio

    def range_acceleration(self, df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 20) -> pd.Series:
        """
        Compute the rate of change in the trading range expansion or contraction.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        high_col : str, optional
            Name of the high column (default is 'high').
        low_col : str, optional
            Name of the low column (default is 'low').
        window_size : int, optional
            Window size for computation (default is 20).

        Returns
        -------
        pd.Series
            The rate of change in the trading range expansion or contraction.
        """
        # Calculate the trading range expansion factor
        expansion_factor = (df[high_col] - df[low_col]) / (df[high_col] + df[low_col])
        
        # Compute the rate of change of the range expansion factor over the specified window
        range_acceleration = expansion_factor.diff().rolling(window=window_size).mean()
        
        return range_acceleration

    def order_flow_imbalance(self, df: pd.DataFrame, tau: float = 0.5, ask_col: str = 'ask', bid_col: str = 'bid', trades_col: str = 'trades') -> pd.Series:
        """
        Calculate order flow imbalance ratio.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        tau : float, optional
            Smoothing parameter (default is 0.5).
        ask_col : str, optional
            Name of ask column (default is 'ask').
        bid_col : str, optional
            Name of bid column (default is 'bid').
        trades_col : str, optional
            Name of trades column (default is 'trades').

        Returns
        -------
        pd.Series
            Imbalance ratio.
        """
        # Compute probability of each trade type (ask or bid)
        prob_asks = (df[trades_col] == df[ask_col]).rolling(window=int(2*tau)).mean()
        prob_bids = (df[trades_col] == df[bid_col]).rolling(window=int(2*tau)).mean()
        
        # Compute imbalance ratio
        imbalance_ratio = (prob_asks - prob_bids) / (prob_asks + prob_bids)
        
        return imbalance_ratio

    def order_flow_imbalance(self, df: pd.DataFrame, window_size: int = 60, buy_volume_col: str = 'buy_volume', sell_volume_col: str = 'sell_volume') -> pd.Series:
        """
        Compute order flow imbalance ratio.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Window size (default is 60).
        buy_volume_col : str, optional
            Buy volume column name (default is 'buy_volume').
        sell_volume_col : str, optional
            Sell volume column name (default is 'sell_volume').

        Returns
        -------
        pd.Series
            Imbalance ratio.
        """
        # Compute mean buy and sell volumes over the window period
        mean_buy_volume = df[buy_volume_col].rolling(window=window_size).mean()
        mean_sell_volume = df[sell_volume_col].rolling(window=window_size).mean()
        
        # Calculate the imbalance ratio using the formula
        imbalance_ratio = (mean_buy_volume - mean_sell_volume) / (mean_buy_volume + mean_sell_volume)
        
        return imbalance_ratio

    def regime_shift_detection(self, df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low', window_size: int = 20, threshold: float = 0.05) -> pd.Series:
        """
        Detect regime shifts by analyzing the distribution of returns over time.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        high_col : str, optional
            high price column name (default is 'high').
        low_col : str, optional
            low price column name (default is 'low').
        window_size : int, optional
            Window size for computing standard deviation of returns (default is 20).
        threshold : float, optional
            Threshold for detecting regime shifts (default is 0.05).

        Returns
        -------
        pd.Series
            Binary indicator of whether a regime shift was detected.
        """
        # Compute returns
        returns = (df[high_col] - df[low_col].shift()) / df[low_col].shift()
        
        # Compute standard deviation of returns
        std_dev = returns.rolling(window=window_size).std()
        
        # Compare standard deviation to threshold * standard deviation of previous window
        regime_shift = std_dev > threshold * std_dev.shift(window_size)
        
        # Mark a regime shift
        detected_shift = regime_shift.astype(int)
        
        return detected_shift

    def regime_shifts(self, df: pd.DataFrame, close_col: str = 'close', threshold: float = 1.5) -> pd.Series:
        """
        Detect regime shifts based on changes in volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        close_col : str, optional
            Name of the close price column (default is 'close').
        threshold : float, optional
            Threshold for detecting regime shifts (default is 1.5).

        Returns
        -------
        pd.Series
            Boolean indicators of regime shifts.
        """
        # Compute returns
        returns = df[close_col].pct_change()
        
        # Calculate sample standard deviation of returns
        std_dev = returns.rolling(window=20).std()
        
        # Compute exponentially weighted moving average of returns
        ewma = returns.ewm(span=20, adjust=False).std()
        
        # Compare standard deviation to EWMA standard deviation
        ratio = std_dev / ewma
        
        # Detect regime shifts
        regime_shift = ratio > threshold
        
        return regime_shift

    def exponential_moving_volatility_ratio(self, df: pd.DataFrame, window_size: int = 20, ratio_window_size: int = 50, close_col: str = 'close', open_col: str = 'open') -> pd.Series:
        """
        Compute the ratio of the exponential moving average of absolute returns to the exponential moving average of absolute price changes.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Window size for exponential moving average of absolute returns (default is 20).
        ratio_window_size : int, optional
            Window size for exponential moving average of absolute price changes (default is 50).
        close_col : str, optional
            Name of the close column (default is 'close').
        open_col : str, optional
            Name of the open column (default is 'open').

        Returns
        -------
        pd.Series
            The ratio of the exponential moving average of absolute returns to the exponential moving average of absolute price changes.
        """
        # Compute daily returns
        daily_returns = (df[close_col] / df[close_col].shift(1)) - 1
        
        # Compute daily price changes
        daily_price_changes = np.abs(df[close_col] - df[open_col])
        
        # Compute the exponential moving average of absolute returns
        ema_returns = daily_returns.ewm(span=window_size, adjust=False).mean().abs()
        
        # Compute the exponential moving average of absolute price changes
        ema_price_changes = daily_price_changes.ewm(span=ratio_window_size, adjust=False).mean()
        
        # Compute the ratio of the exponential moving averages
        ratio = ema_returns / ema_price_changes
        
        return ratio

    def exponential_moving_average_volatility(self, df: pd.DataFrame, window_size: int = 20, close_col: str = 'close', smoothing_factor: float = 0.02) -> pd.Series:
        """
        Compute exponential moving average volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the moving window (default is 20).
        close_col : str, optional
            Name of the column containing close prices (default is 'close').
        smoothing_factor : float, optional
            Smoothing factor for the exponential moving average (default is 0.02).

        Returns
        -------
        pd.Series
            Exponential moving average volatility.
        """
        # Calculate returns of the close prices
        returns = df[close_col].diff()
        
        # Square each return to get squared returns
        squared_returns = returns ** 2
        
        # Calculate the moving sum of the squared returns over the window size
        moving_sum = squared_returns.ewm(span=window_size, adjust=False).mean()
        
        # Divide the moving sum by the window size and multiply by the square root
        emav = np.sqrt(moving_sum) * (1 - smoothing_factor)
        
        return emav

    def correlation_diff(self, df: pd.DataFrame, window: int = 20, close_col: str = 'close', benchmark_col: str = 'benchmark') -> pd.Series:
        """
        Compute difference in correlation between asset and benchmark returns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window : int, optional
            Window size (default is 20).
        close_col : str, optional
            close column name (default is 'close').
        benchmark_col : str, optional
            Benchmark column name (default is 'benchmark').

        Returns
        -------
        pd.Series
            Difference in correlation between asset and benchmark returns.
        """
        # Compute returns for the asset and the benchmarks
        asset_returns = df[close_col].pct_change()
        benchmark_returns = df[benchmark_col].pct_change()

        # Compute the correlation matrix between the asset and the benchmarks for the current time window
        corr_current = asset_returns.rolling(window=window).corr(benchmark_returns)

        # Compute the correlation matrix between the asset and the benchmarks for the previous time window
        corr_previous = asset_returns.rolling(window=window).corr(benchmark_returns).shift(window)

        # Calculate the difference in correlation between the current and previous time windows
        correlation_diff = corr_current - corr_previous

        return correlation_diff

    def rolling_correlation(self, df: pd.DataFrame, window_size: int = 20, min_periods: int = 5, lag: int = 1, close_col1: str = 'close', close_col2: str = 'close') -> pd.Series:
        """
        Compute rolling correlation between two time series at a given lag.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the rolling window (default is 20).
        min_periods : int, optional
            Minimum number of observations in window required to have a value (default is 5).
        lag : int, optional
            Lag between the two time series (default is 1).
        close_col1 : str, optional
            Name of the first close price column (default is 'close').
        close_col2 : str, optional
            Name of the second close price column (default is 'close').

        Returns
        -------
        pd.Series
            Rolling correlation coefficient between the two input time series at a given lag.
        """
        # Shift the second time series by the specified lag
        shifted_close_col2 = df[close_col2].shift(lag)
        
        # Compute the rolling correlation
        correlation = df[close_col1].rolling(window=window_size, min_periods=min_periods).corr(shifted_close_col2)
        
        return correlation

    def moving_average_convergence_divergence(self, 
        df: pd.DataFrame, 
        fast_ma_window: int = 12, 
        slow_ma_window: int = 26, 
        signal_ma_window: int = 9, 
        close_col: str = 'close'
    ) -> pd.Series:
        """
        Compute the Moving Average Convergence Divergence (MACD) line.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the close price column.
        fast_ma_window : int, optional
            Window size for the fast moving average (default is 12).
        slow_ma_window : int, optional
            Window size for the slow moving average (default is 26).
        signal_ma_window : int, optional
            Window size for the signal line (default is 9).
        close_col : str, optional
            Name of the close price column (default is 'close').

        Returns
        -------
        pd.Series
            The MACD line.
        """
        # Compute fast moving average
        fast_ma = df[close_col].ewm(span=fast_ma_window, adjust=False).mean()
        
        # Compute slow moving average
        slow_ma = df[close_col].ewm(span=slow_ma_window, adjust=False).mean()
        
        # Compute MACD line
        macd_line = fast_ma - slow_ma
        
        # Return MACD line
        return macd_line

    def standardized_price_action_range(self, df: pd.DataFrame, atr_length: int = 14, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series:
        """
        Compute the standardized price action range.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing price data.
        atr_length : int, optional
            Length of the average true range calculation (default is 14).
        high_col : str, optional
            Column name for high prices (default is 'high').
        low_col : str, optional
            Column name for low prices (default is 'low').
        close_col : str, optional
            Column name for close prices (default is 'close').

        Returns
        -------
        pd.Series
            Standardized price action range.
        """
        # Compute true range
        df['hl'] = df[high_col] - df[low_col]
        df['hc'] = np.abs(df[high_col] - df[close_col].shift(1))
        df['lc'] = np.abs(df[low_col] - df[close_col].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        
        # Calculate average true range
        df['atr'] = df['tr'].rolling(window=atr_length).mean()
        
        # Compute range of price action
        price_action_range = df[high_col] - df[low_col]
        
        # Standardize range by dividing by average true range
        standardized_range = price_action_range / df['atr']
        
        return standardized_range

    def normalized_rolling_volatility(self, df: pd.DataFrame, window_size: int = 20, normalization_window: int = 200, close_col: str = 'close') -> pd.Series:
        """
        Compute normalized rolling volatility.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Size of the rolling window (default is 20).
        normalization_window : int, optional
            Normalization window size (default is 200).
        close_col : str, optional
            Column name for close prices (default is 'close').

        Returns
        -------
        pd.Series
            Normalized rolling volatility.
        """
        # Compute close price returns
        returns = df[close_col].pct_change()
        
        # Create a rolling window of size 'window_size' to compute the sum of squared returns
        rolling_squared_returns = (returns ** 2).rolling(window=window_size)
        
        # Calculate the sum of squared returns over the rolling window
        sum_squared_returns = rolling_squared_returns.sum()
        
        # Calculate the mean of squared returns over the rolling window
        mean_squared_returns = rolling_squared_returns.mean()
        
        # Subtract the mean from the sum of squared returns and take the square root
        volatility = np.sqrt((1 / (window_size - 1)) * (sum_squared_returns - window_size * mean_squared_returns))
        
        # Divide the result by the normalization window to obtain the normalized volatility value
        normalized_volatility = (1 / normalization_window) * volatility
        
        return normalized_volatility

    def realized_volatility_decay(self, df: pd.DataFrame, window_size: int = 30, close_col: str = 'close') -> pd.Series:
        """
        Compute the decay rate of realized volatility over the specified window.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        window_size : int, optional
            Window length (default is 30).
        close_col : str, optional
            close price column name (default is 'close').

        Returns
        -------
        pd.Series
            Decay rate of realized volatility.
        """
        # Compute daily realized volatility
        realized_volatility = df[close_col].pct_change().rolling(window=window_size).std()
        
        # Calculate decay rate
        decay_rate = 1 - (realized_volatility / realized_volatility.shift(1)) ** (1 / (window_size / window_size))
        
        # Return the result
        return decay_rate

    def double_smooth_ewm_momentum(self, df: pd.DataFrame, close_col: str = 'close', alpha: float = 0.5) -> pd.Series:
        """
        Calculate the double-smoothed exponentially weighted momentum of a given price series.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        close_col : str, optional
            Name of the close price column (default is 'close').
        alpha : float, optional
            Smoothing factor (default is 0.5).

        Returns
        -------
        pd.Series
            The double-smoothed exponentially weighted momentum of the given price series.
        """
        # Compute exponentially weighted moving average of the close price
        ewm_close = df[close_col].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate the difference between the ewma of the current close and the ewma of the previous close
        diff_ewm_close = ewm_close - ewm_close.shift(1)
        
        # Apply exponential weighting again to the differences
        double_smooth_ewm = diff_ewm_close.ewm(alpha=alpha, adjust=False).mean()
        
        return double_smooth_ewm

    def exponential_moving_average_difference(self, df: pd.DataFrame, short_window: int = 10, long_window: int = 50, close_col: str = 'close') -> pd.Series:
        """
        Compute the difference between two exponential moving averages with different time periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        short_window : int, optional
            Short window size (default is 10).
        long_window : int, optional
            Long window size (default is 50).
        close_col : str, optional
            close price column name (default is 'close').

        Returns
        -------
        pd.Series
            Difference between short and long window exponential moving averages.
        """
        # Calculate alpha for short and long window
        alpha_short = 2 / (short_window + 1)
        alpha_long = 2 / (long_window + 1)

        # Compute exponential moving averages
        ema_short = df[close_col].ewm(alpha=alpha_short, adjust=False).mean()
        ema_long = df[close_col].ewm(alpha=alpha_long, adjust=False).mean()

        # Compute EMAD_diff as the difference between EMA_long and EMA_short
        emad_diff = ema_long - ema_short

        return emad_diff

    def exponentially_weighted_volatility(self, df: pd.DataFrame, close_col: str = 'close', alpha: float = 0.9) -> pd.Series:
        """
        Compute exponentially weighted volatility of a time series.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        close_col : str, optional
            Name of the close price column (default is 'close').
        alpha : float, optional
            Smoothing factor (default is 0.9).

        Returns
        -------
        pd.Series
            Exponentially weighted volatility of the time series.
        """
        # Compute daily returns
        returns = df[close_col].pct_change()
        
        # Initialize volatility
        volatility = np.zeros(len(returns))
        volatility[0] = np.nan  # first value will be NaN
        
        # Calculate new volatility for each day
        for i in range(1, len(returns)):
            if i == 1:
                volatility[i] = returns[i]**2
            else:
                volatility[i] = volatility[i-1] * (1-alpha) + alpha * returns[i]**2
        
        # Return exponentially weighted volatility as a pandas Series
        return pd.Series(volatility, index=df.index)



    def derivatives(self, df,col):
        """
        Calculates the first and second derivatives of a given column in a DataFrame 
        and adds them as new columns 'velocity' and 'acceleration'.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the column for which derivatives are to be calculated.
            
        col : str
            The column name for which the first and second derivatives are to be calculated.

        Returns:
        --------
        df_copy : pandas.DataFrame
            A new DataFrame with 'velocity' and 'acceleration' columns added.

        """
        
        df_copy = df.copy()

        df_copy["velocity"] = df_copy[col].diff().fillna(0)
        df_copy["acceleration"] = df_copy["velocity"].diff().fillna(0)
        
        return df_copy