import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


def plot_selected(df, columns, start_index, end_index):
    plot_data(df.ix[start_index:end_index, columns])


def symbol_to_path(symbol, base_dir='data'):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols: # add SPY for reference, if absent
        symbols.inters(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", 
                              parse_dates=True, usecols=['Date', 'Adj Close'], 
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp) 
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

    return df


def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe."""
    return df / df.ix[0,:]


def plot_data(df, title="Stock prices", xlabel='Date', ylabel='Price'):
    """Plot stock prices"""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def test_daily_return():
    start_date = '2012-01-04'
    end_date = '2012-12-31'
    dates = pd.date_range(start_date, end_date)

    symbols = ['SPY', 'IBM']

    df = get_data(symbols, dates)
    plot_data(df)

    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title='Daily returns', ylabel='Daily returns')


def test_run():
    start_date = '2012-01-04'
    end_date = '2012-12-31'
    dates = pd.date_range(start_date, end_date)

    #Read in more stocks
    symbols = ['SPY'] #, 'GOOG', 'IBM', 'GLD']

    df = get_data(symbols, dates)
    #df = normalize_data(df)
    #plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')

    rm_SPY = get_rolling_mean(df['SPY'], window=20)

    rstd_SPY = get_rolling_std(df['SPY'], window=20)

    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    ax = df['SPY'].plot(title='Bollinger Bands', label='SPY')
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    #plot_data(df)


if __name__ == '__main__':
    test_daily_return()
