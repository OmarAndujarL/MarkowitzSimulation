# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:13:34 2022

@author: omara
"""

import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import plotly.io as pio 

pio.renderers.default = 'browser'


def line_fig(df,stock_list, title):
    fig = px.line(x=df['Date'], y=df['value'],
                  color=df['variable'], title=title)
    fig.update_layout(
                   yaxis_title='',
                   xaxis_title=''
                   )

    return fig


def get_stocks(stock_list, months):
    tickers = yf.Tickers(stock_list)  # returns a named tuple of Ticker objects
    stock_df = tickers.history(period=months)['Close']
    return stock_df


def stock_price_line(stock_df, stock_list):
    s_df = stock_df.reset_index()
    stock_df_long = pd.melt(
        s_df, id_vars=s_df.columns.values[0], value_vars=s_df.columns.values[1:])
    fig = line_fig(stock_df_long, stock_list, 'Closing price over time')
    return fig


def growth_df(stock_df, stock_list):
    growth_df = (stock_df / stock_df.iloc[0] * 100).reset_index()
    growth_df_long = pd.melt(
        growth_df, id_vars=growth_df.columns.values[0], value_vars=growth_df.columns.values[1:])  # "tidy data"
    fig = line_fig(growth_df_long, stock_list,
                   'Growth of investment over total dataset')
    return fig


def Returns(stock_df, stock_list):
    returns_df = (np.log(stock_df) - np.log(stock_df.shift(1))).reset_index()
    mean_returns = returns_df.mean(numeric_only = True)
    returns_cov = returns_df.cov()*252
    returns_corr = returns_df.corr()
    returns_df_long = pd.melt(
        returns_df, id_vars=returns_df.columns.values[0], value_vars=returns_df.columns.values[1:])  # "tidy data"
    fig = line_fig(returns_df_long, stock_list, f'Logarithmic returns over time')
    return returns_df, returns_cov, returns_corr, mean_returns, returns_df_long, fig


def graph_corr(df):
    fig = px.imshow(df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='reds',
                    title='Correlation matrix')
    fig.update_xaxes(side="top")
    return fig


def portfolio_performance(mean_returns, returns_cov, weights):
    '''    Given a returns data frame, with its covariance data frame and its weights,
    it returns the volatility, sharpe index and total return value for that portfolio 
    '''
    ret = np.sum(weights * mean_returns) * 252
    v = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights))) * np.sqrt(252)
    sharpe = (ret - risk_free_rt)/v
    return ret, v, sharpe


def markowitz_simulation(n, number_of_stocks):
    return_p = []
    volatility_p = []
    sharpe_p = []
    weights_p = []
    sharpe_p = []

    print('Simulations progress bar:')

    for i in tqdm(range(n)):
        weights = np.random.random(number_of_stocks)
        weights = weights/np.sum(weights)
        ret, v, sharpe = portfolio_performance(
            mean_returns, returns_cov, weights)
        return_p.append(ret)
        volatility_p.append(v)
        weights_p.append(weights)
        sharpe_p.append(sharpe)

    return_p = np.array(return_p)
    volatility_p = np.array(volatility_p)
    sharpe_p = np.array(sharpe_p)
    weights_p = np.array(weights_p)
    return weights_p, return_p, volatility_p, sharpe_p


def Efficient_Frontier(return_p, returns_cov, mean_returns, stock_list):
    returns = np.linspace(min(return_p), max(return_p), 50)
    volatility_opt = []

    def checkSumToOne(w):
        return np.sum(w)-1

    def minimizeMyVolatility(w):
        w = np.array(w)
        V = np.sqrt(np.dot(w.T, np.dot(returns_cov, w))) * np.sqrt(252)
        return V

    def getReturn(w):
        w = np.array(w)
        R = np.sum(w * mean_returns) * 252
        return R

    w0 = [0.25 for i in range(len(stock_list))]
    bounds = tuple((0, 1) for i in range(len(w0)))

    for r in returns:
        constraints = ({'type': 'eq', 'fun': checkSumToOne},
                       {'type': 'eq', 'fun': lambda w: getReturn(w) - r})
        opt = minimize(minimizeMyVolatility, w0, method='SLSQP',
                       bounds=bounds, constraints=constraints)
        volatility_opt.append(opt['fun'])

    return volatility_opt, returns


def markowitz_graph(port_df, n_simulations):
    # Max Sharpe
    max_s_v = port_df[port_df['Sharpe index'] ==
                      port_df['Sharpe index'].max()]['Volatility']
    max_s_r = port_df[port_df['Sharpe index'] ==
                      port_df['Sharpe index'].max()]['Returns']

    # Min Volatility
    min_vol_v = port_df[port_df['Volatility'] ==
                        port_df['Volatility'].min()]['Volatility']
    min_vol_r = port_df[port_df['Volatility'] ==
                        port_df['Volatility'].min()]['Returns']

    port_df_2 = port_df[(port_df['Volatility'] != port_df['Volatility'].min()) &
                        (port_df['Sharpe index'] != port_df['Sharpe index'].max())]

    fig = px.scatter(port_df_2, x="Volatility", y="Returns",
                     title=f'Volatility vs Returns for {n_simulations} Portfolios',
                     color="Sharpe index",
                     color_continuous_scale='Bluered_r')
    fig.add_trace(go.Scatter(x=max_s_v,
                             y=max_s_r,
                             mode='markers',
                             name='Max Sharpe',
                             marker_color='crimson',
                             marker_size=8,
                             marker_symbol='diamond'))

    fig.add_trace(go.Scatter(x=min_vol_v,
                             y=min_vol_r,
                             mode='markers',
                             name='Min Volatility',
                             marker_color='orange',
                             marker_size=8,
                             marker_symbol='diamond'))
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    return fig


def weight_min_vol(port_df):
    weight_index = port_df.index[port_df['Volatility']
                                 == port_df['Volatility'].min()]
    ws = weights_p[weight_index]
    return ws[0]


def weight_max_sharpe(port_df):
    weight_index = port_df.index[port_df['Sharpe index']
                                 == port_df['Sharpe index'].max()]
    ws = weights_p[weight_index]
    return ws[0]


def center_title(fig):
    fig.update_layout(
            title={
            'x':0.5,
            'xanchor': 'center'
        },
        template="seaborn"
        )
    return fig




np.random.seed(73)
time_period = '72mo'  # Data from the past 72 months
stock_list = ['TSLA','AAPL', 'KO', 'DIS', 'BAC', 'MSFT']
df = get_stocks(stock_list, time_period)
risk_free_rt = 0.0125
n_simulations = 20_000
fig1 = stock_price_line(df, stock_list)
fig2 = growth_df(df, stock_list)
returns_df, returns_cov, returns_corr, mean_returns, returns_df_long, fig3 = Returns(
    df, stock_list)
fig4 = graph_corr(returns_corr)
weights_p, return_p, volatility_p, sharpe_p = markowitz_simulation(
    n_simulations, len(stock_list))
port_df = pd.DataFrame(
    {'Returns': return_p, 'Volatility': volatility_p, 'Sharpe index': sharpe_p})

volatility_opt, returns = Efficient_Frontier(
   return_p, returns_cov, mean_returns, stock_list)


weights_vol = weight_min_vol(port_df)
weights_sharpe = weight_max_sharpe(port_df)
Max_sharpe_df = pd.DataFrame({'Stocks': stock_list, 'Weights': weights_sharpe})
Min_vol_df = pd.DataFrame({'Stocks': stock_list, 'Weights': weights_vol})


Max_sharpe_df["Weights"] = ((Max_sharpe_df["Weights"]*100).round(2).astype(str))+'%'
Min_vol_df["Weights"] = ((Min_vol_df["Weights"]*100).round(2).astype(str))+'%'


fig5 = markowitz_graph(port_df, n_simulations)
fig5.add_trace(go.Scatter(x=volatility_opt,
                          y=returns,
                          mode='lines',
                          name='Efficient frontier',
                          marker_color='darkslateblue'
                          ))


fig5 = center_title(fig5)
fig4 = center_title(fig4)
fig3 = center_title(fig3)
fig2 = center_title(fig2)
fig1 = center_title(fig1)


fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()

print('Maximum Sharpe portfolio with weights')
print(Max_sharpe_df)
print()
print('Minimum volatility portfolio with weights')
print(Min_vol_df)


