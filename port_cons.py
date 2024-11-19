import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from datetime import date, timedelta
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import fsolve
import itertools
from tqdm import tqdm
import plotly.graph_objects as go
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.covariance import LedoitWolf
from matplotlib import colors
import statsmodels.formula.api as smf

TOLERANCE = 1e-11

def apply_style_heatmap(df):
    # styled_df = df.style.background_gradient(cmap='Blues')
    return df.style.background_gradient(cmap='Blues', axis=None)

def apply_2dp_percentage(df):
    return df.format('{:.2%}')

def df_apply_percentage_format(df):
    return df.map('{:.2%}'.format)

def freeze(df):
    table_styles = [
            {'selector': 'thead th',
             'props': [('position', 'sticky'), ('top', '0'), ('background', 'white')]},
            {'selector': 'tbody th',
             'props': [('position', 'sticky'), ('left', '0'), ('background', 'white')]}]
    
    # styled_df = df.style.set_table_styles(table_styles)
    return df.style.set_table_styles(table_styles)

def apply_style_heatmap_ret(df, subset=None):
    cdict = {'red':  ((0.0, 0.8, 0.8), 
                  (0.5, 1.0, 1.0),   
                  (1.0, 0, 0.0)), 

        'green': ((0.0, 0.0, 0.0),   
                  (0.5, 1.0, 1.0), 
                  (1.0, 0.8, 0.8)),

        'blue':  ((0.0, 0.0, 0.0),  
                  (0.5, 1.0, 1.0),   
                  (1.0, 0.0, 0.0)) 
       }

    cmap = colors.LinearSegmentedColormap('GnRd', cdict)

    max_val = abs(df).max().max()
    if subset is None:
        return df.style.background_gradient(cmap=cmap, vmin=-max_val, vmax=max_val).format('{:.2%}')
    else:
        return df.style.background_gradient(cmap=cmap, vmin=-max_val, vmax=max_val, subset=subset).format('{:.2%}')


def date_int_to_str(date_int):
    return datetime.strptime(str(date_int), '%Y%m%d').strftime('%Y-%m-%d')

def convert_index_integer(df):
    df.index = df.index.strftime('%Y%m%d').astype(int)
    return df

def convert_index_datetime(df):
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d').strftime('%Y-%m-%d')
    return df

def cumulative_return(df):
    cumulative_ret = (1+df).cumprod() - 1
    return cumulative_ret

def resample_returns_weekly(df_ret, day='wed'):
    if day=='wed':
        df_ret_resampled = df_ret.resample('W-WED').apply(lambda x: (x + 1).prod() - 1)
    else:
        df_ret_resampled = df_ret.resample('W').apply(lambda x: (x + 1).prod() - 1)
        
    return df_ret_resampled

def resample_returns_monthly(df_ret):
   
    df_ret_resampled = df_ret.resample('M').apply(lambda x: (x + 1).prod() - 1)
    return df_ret_resampled

def covar_ewma(returns, alpha):
    weights = (1-alpha) ** np.arange(len(returns))[::-1]

    normalized = (returns-returns.mean()).fillna(0).to_numpy()
    
    out =  ((weights * normalized.T) @ normalized / weights.sum())   
    out = pd.DataFrame(out)
    out.index = returns.columns.tolist()
    out.columns = returns.columns.tolist()
    return out

def exp_covar(returns, halflife, annualized=261):
    alpha = 1 - math.exp(math.log(0.5) / (halflife* annualized))
    span = (2 / alpha) - 1
    exp_cov_matrix = returns.ewm(span=span).cov(pairwise=True).iloc[-len(returns.columns):]
    covar_ann = exp_cov_matrix * annualized
    covar_ann = pd.DataFrame(covar_ann.values, index=returns.columns, columns=returns.columns)
    return covar_ann

def cov2corr(cov, return_std=False):

    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
    
def max_mult_constraint(weights, alloc, min_mult_allocation):
    combined_allocation = np.dot(weights, alloc)
    return min_mult_allocation - combined_allocation

def min_mult_constraint(weights, alloc, min_mult_allocation):
    combined_allocation = np.dot(weights, alloc)
    return combined_allocation - min_mult_allocation

def min_sum_constraint(weights, min_sum_allocation, etf_names, returns):
    etf_indices = [returns.columns.get_loc(etf) for etf in etf_names]
    sum_allocation = np.sum(weights[etf_indices]) - min_sum_allocation
    return sum_allocation

def max_sum_constraint(weights, max_sum_allocation, etf_names, returns):
    etf_indices = [returns.columns.get_loc(etf) for etf in etf_names]
    sum_allocation =  max_sum_allocation - np.sum(weights[etf_indices]) 
    return sum_allocation

def _allocation_risk(weights, covariances):

    # calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # returns the risk of the weights distribution
    return portfolio_risk


def _risk_budget_objective_error(weights, args):
    covariances = args[0]
    weights = np.matrix(weights)
    portfolio_risk = _allocation_risk(weights, covariances)

    error = portfolio_risk

    return error


def _get_minvol_weights(covariances, initial_weights):
    H = 1/0.25
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},{'type': 'ineq', 'fun': lambda x: x},
                   {'type': 'ineq', 'fun': lambda x: -np.sum(x**2) + 1/H}
                  )

    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances],
                               method='SLSQP',
                               constraints=constraints,
                               tol=1e-11,
                               options={'disp': False})

    weights = optimize_result.x
    return weights


def topX_minvol(minvol_wgt_all, top_x_assets):
    sector_etfs = minvol_wgt_all.columns
    topX_wgt = pd.DataFrame(columns=sector_etfs)
    
    for i in range(len(minvol_wgt_all)):
        top_etfs = minvol_wgt_all.iloc[i].nlargest(top_x_assets).index
        topX_wgt.loc[i, top_etfs] = minvol_wgt_all.iloc[i][top_etfs]
        topX_sum = topX_wgt.loc[i, top_etfs].sum()
        topX_wgt.loc[i, top_etfs] = topX_wgt.loc[i, top_etfs] / topX_sum
    
    topX_wgt = topX_wgt.fillna(0)
    topX_wgt.index = minvol_wgt_all.index
    return topX_wgt


def negative_sharpe_ratio(weights, returns, covar, rf):
    portfolio_return = np.dot(returns.T, weights)
    weights_values = weights.values.flatten() if isinstance(weights, pd.DataFrame) else weights
    portfolio_volatility = np.sqrt(np.dot(weights_values.T, np.dot(covar, weights_values)))
    return -(portfolio_return - rf) / portfolio_volatility


def get_maxSharpe_wgt(ret, covar, rf, constraints):
    TOLERANCE = 1e-11
    initial_weights = [1. / len(ret)] * len(ret)
    bounds = [(0, 1)] * len(ret)
    
    result = minimize(negative_sharpe_ratio, initial_weights, args=(ret, covar, rf), 
                      method='SLSQP', constraints=constraints, bounds=bounds, tol=TOLERANCE)

    optimized_weights = result.x
    print(result.message)
    return optimized_weights


def max_sharpe_rebal_wgt(returns_data, start_date, end_date, constraints_dict, shrink_covar=True, rebal_months=[4,10], annualized=261, halflife=3.5):
    
    ret_data_all = returns_data.loc[:end_date]
    ret_data_filtered = returns_data.loc[start_date:end_date]
    first_date = ret_data_all.iloc[0].name
    tickers = list(ret_data_all.columns)

    weights = pd.DataFrame(0,index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y%m%d')
    reb_flag = pd.DataFrame(0,index=weights.index, columns=['reb_flag'])
    
    # param for exp covar
    alpha = 1 - math.exp(math.log(0.5) / (halflife* annualized))
    span = (2 / alpha) - 1
             
    # rebalance first week of April and October
    for i in range(0, len(ret_data_filtered)):

        # for first datapont or on rebalance dates, get optimized weight
        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i==0:
            
            curr_date = ret_data_filtered.iloc[i].name
            
            # shrink covariance matrix
            if shrink_covar:
                shrink_covar_matrix_model = LedoitWolf().fit(ret_data_all.loc[first_date:curr_date])
                # display(shrink_covar_matrix_model.shrinkage_)
                shrink_covar_matrix = shrink_covar_matrix_model.covariance_
                covar_ann = shrink_covar_matrix * annualized
                
            # exponentially weighted covariance matrix    
            else:
                exp_cov_matrix = ret_data_all.loc[first_date:curr_date].ewm(span=span).cov(pairwise=True).iloc[-len(ret_data_all.columns):]
                covar_ann = exp_cov_matrix * annualized
            
            # mean returns = avg annualized return from first data point up to rebal date
            mean_ret = (ret_data_all.loc[first_date:curr_date].mean() * annualized)

            new_wgt = get_maxSharpe_wgt(mean_ret, covar_ann, rf=0, constraints=constraints_dict)
            weights.iloc[i] = new_wgt
            reb_flag.iloc[i] = True
        # non-rebalance dates: weight will drift according to asset's daily returns
        else:
            weights.iloc[i] = weights.iloc[i-1] * (1+ ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            weights.iloc[i] /= weights_sum
            reb_flag.iloc[i] = False

    weights = weights.rename(columns={c: c+'_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)
    return weights

def min_vol_rebal_wgt(returns_data, start_date, end_date, constraints_dict, shrink_covar=True, rebal_months=[4,10], annualized=261, halflife=3.5):
    
    ret_data_all = returns_data.loc[:end_date]
    ret_data_filtered = returns_data.loc[start_date:end_date]
    first_date = ret_data_all.iloc[0].name
    tickers = list(ret_data_all.columns)

    weights = pd.DataFrame(0,index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y%m%d')
    reb_flag = pd.DataFrame(0,index=weights.index, columns=['reb_flag'])
    
    alpha = 1 - math.exp(math.log(0.5) / (halflife* annualized))
    span = (2 / alpha) - 1
             
    for i in range(0, len(ret_data_filtered)):

        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i==0:
            
            curr_date = ret_data_filtered.iloc[i].name
            
            if shrink_covar:
                shrink_covar_matrix_model = LedoitWolf().fit(ret_data_all.loc[first_date:curr_date])
                shrink_covar_matrix = shrink_covar_matrix_model.covariance_
                covar_ann = shrink_covar_matrix * annualized
                
            else:
                exp_cov_matrix = ret_data_all.loc[first_date:curr_date].ewm(span=span).cov(pairwise=True).iloc[-len(ret_data_all.columns):]
                covar_ann = exp_cov_matrix * annualized
            
            mean_ret = (ret_data_all.loc[first_date:curr_date].mean() * annualized)

            new_wgt = min_vol(ret_data_all, covar_ann, constraints=constraints_dict)
            weights.iloc[i] = new_wgt
            reb_flag.iloc[i] = True
            
        # non-rebalance dates: weight will drift according to asset's daily returns
        else:
            weights.iloc[i] = weights.iloc[i-1] * (1+ ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            weights.iloc[i] /= weights_sum
            reb_flag.iloc[i] = False

    weights = weights.rename(columns={c: c+'_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)
    return weights

def target_vol_func(weights, returns, covariance_matrix, target_volatility):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return (portfolio_volatility - target_volatility) ** 2

def get_tgtvol_wgt(returns, covariance_matrix, target_volatility, constraints):
    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    
    bounds = [(0, 1) for _ in range(num_assets)]
    result = minimize(target_vol_func, initial_weights, args=(returns, covariance_matrix, target_volatility),
                      method='SLSQP', constraints=constraints, bounds=bounds)
    
    optimized_weight = result.x
    print(result.message)
    return optimized_weight

def portfolio_vol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def min_vol(returns, cov_matrix, constraints):
    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
        
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    optimized = minimize(portfolio_vol, initial_weights, args=(cov_matrix,),
                         method='SLSQP', constraints=constraints, bounds=bounds)
    print(optimized.message)
  
    return optimized.x

def equal_weight_rebal_wgt(returns_data, start_date, end_date, rebal_months=[4, 10]):
    ret_data_filtered = returns_data.loc[start_date:end_date]
    weights = pd.DataFrame(0, index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y%m%d')
    reb_flag = pd.DataFrame(0, index=weights.index, columns=['reb_flag'])
    
    num_assets = len(weights.columns)
    equal_weight = 1.0 / num_assets  # Each asset receives an equal share

    for i in range(0, len(ret_data_filtered)):
        # Check if it's the first data point or a rebalance date
        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i == 0:
            weights.iloc[i] = equal_weight  # Set equal weights
            reb_flag.iloc[i] = True
        else:
            # Non-rebalance dates: weight drifts according to asset's daily returns
            weights.iloc[i] = weights.iloc[i-1] * (1 + ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            weights.iloc[i] /= weights_sum  # Normalize to maintain a total weight of 1
            reb_flag.iloc[i] = False

    weights = weights.rename(columns={c: c + '_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)
    return weights

def fixed_weight_rebal_wgt(returns_data, start_date, end_date, fixed_weights, rebal_months=[4, 10]):
    """
    Function to create a fixed-weight portfolio with rebalancing at specified months.
    
    Parameters:
    - returns_data: DataFrame containing asset returns
    - start_date: Start date for the backtest
    - end_date: End date for the backtest
    - fixed_weights: Dictionary of fixed weights, e.g., {'SPY': 0.2, 'EFA': 0.3, ...}
    - rebal_months: List of months for rebalancing
    
    Returns:
    - DataFrame with weights over time
    """
    # Ensure weights sum to 1
    if round(sum(fixed_weights.values()),6) != 1:
        raise ValueError("Fixed weights must sum to 1.")
    
    # Filter data for the specified date range
    ret_data_filtered = returns_data.loc[start_date:end_date]
    weights = pd.DataFrame(0, index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y%m%d')
    reb_flag = pd.DataFrame(0, index=weights.index, columns=['reb_flag'])

    # Loop through the data and assign fixed weights on rebalance dates
    for i in range(0, len(ret_data_filtered)):
        # Check if it's the first data point or a rebalance date
        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i == 0:
            # Set fixed weights for each ticker
            for ticker, wgt in fixed_weights.items():
                weights.at[weights.index[i], ticker] = wgt
            reb_flag.iloc[i] = True
        else:
            # Non-rebalance dates: weight drifts according to asset's daily returns
            weights.iloc[i] = weights.iloc[i-1] * (1 + ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            weights.iloc[i] /= weights_sum  # Normalize to maintain a total weight of 1
            reb_flag.iloc[i] = False

    # Rename columns to include '_wgt' suffix for clarity
    weights = weights.rename(columns={c: c + '_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)
    return weights

def display_rebal_wgt(df_wgt):
    display(df_wgt[df_wgt['reb_flag']==True].pipe(apply_style_heatmap).pipe(apply_2dp_percentage))
    
    
def get_tickers_from_weights(weight_df):
    return [col[:-4] for col in weight_df.columns if col.endswith('_wgt')]


def equity_alloc_constraint(weights, equity_allocations, max_equity_allocation):
    combined_allocation = np.dot(weights, equity_allocations)
    return max_equity_allocation - combined_allocation


########## PERFORMANCE / METRICS ##########

def rename_port_cols(name, portfolio_perf_df):
    portfolio_perf_df = portfolio_perf_df.rename(columns={'port_return': name + '_port_ret',
                                                      'cumulative_port_return':name + '_cumret',
                                                      'port_nav':name +'_NAV'})
    return portfolio_perf_df


def calc_port_perf(ret_data, weight, name_prefix=None, start_date=None, end_date=None):
    
    col_name = get_tickers_from_weights(weight)
    ret_data = ret_data.loc[start_date:end_date][col_name]
    
    ret_data.index = pd.to_datetime(ret_data.index) 
    tickers = ret_data.columns
    ret_data.iloc[0] = 0
    df_bt = pd.concat([ret_data.add_suffix('_return'), weight], axis=1)
    
    df_bt = df_bt.loc[start_date:end_date]
    
    returns_columns = df_bt.filter(like='_return')
    weights_columns = df_bt.filter(like='_wgt')
    weighted_returns = pd.DataFrame(0, index=df_bt.index, columns=ret_data.columns)

    for i in range(1, len(weighted_returns)):
        weighted_returns.iloc[i] = weights_columns.iloc[i-1].values * returns_columns.iloc[i].values
        
    df_bt = pd.concat([df_bt, weighted_returns.add_suffix('_weighted_return')], axis=1)
    df_bt['port_return'] = df_bt.filter(like='_weighted_return').sum(axis=1)
    df_bt['cumulative_port_return'] = (1 + df_bt['port_return']).cumprod() - 1
    df_bt['port_nav'] = 100 * (1 + df_bt['cumulative_port_return'])
    
    if name_prefix is not None:
        df_bt = df_bt.rename(columns={'port_return': name_prefix + '_port_ret',
                                      'cumulative_port_return':name_prefix + '_cumret',
                                      'port_nav':name_prefix +'_NAV'})
    
    return df_bt

def performance_annualized_ret(ret_data, freq):
    if freq == 'daily':
        return ret_data.mean() * 261
    elif freq == 'weekly':
        return ret_data.mean() * 52
    elif freq == 'monthly':
        return ret_data.mean() * 12
    
def performance_annualized_ret_via_cum(ret_data, freq):
    cumulative_ret = cumulative_return(ret_data).iloc[-1]
    num_rows = len(ret_data)
    if freq == 'daily':
        annualized = 261
    elif freq == 'weekly':
        annualized = 52
    elif freq == 'monthly':
        annualized = 12
        
    ann_ret = (1 + cumulative_ret) ** (annualized/num_rows) - 1
    return ann_ret
    

def performance_annualized_vol(ret_data, freq):
    if freq == 'daily':
        return ret_data.std() * np.sqrt(261)
    elif freq == 'weekly':
        return ret_data.std() * np.sqrt(52)
    elif freq == 'monthly':
        return ret_data.std() * np.sqrt(12)
    
def performance_sharpe(ret_data, freq, rf=0):
    return (performance_annualized_ret_via_cum(ret_data, freq) - 0) / performance_annualized_vol(ret_data, freq)

def performance_sortino(ret_data, freq, rf=0):
    if isinstance(ret_data, pd.Series):
        ret_data = ret_data.to_frame()  # Convert Series to DataFrame
    
    if freq == 'daily':
        return (performance_annualized_ret_via_cum(ret_data, freq) - 0) / (ret_data[ret_data < 0].std(axis=0) * np.sqrt(261))
    elif freq == 'weekly':
        return (performance_annualized_ret_via_cum(ret_data, freq) - 0) / (ret_data[ret_data < 0].std(axis=0) * np.sqrt(52))
    elif freq == 'monthly':
        return (performance_annualized_ret_via_cum(ret_data, freq) - 0) / (ret_data[ret_data < 0].std(axis=0) * np.sqrt(12))
    
def performance_mdd(ret_data):
    cum_ret = (ret_data + 1).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    dd = cum_ret / peak - 1
    mdd = dd.min()
    return mdd

def portfolio_turnover(*args, start_date=None, end_date=None, rebal_per_yr=2):

    df_turnover = pd.DataFrame()
    
    for arg in args:
        arg = arg.loc[start_date:end_date]
        arg = arg[arg['reb_flag']==True].iloc[:,1:]      
        rebal_turnover = arg.diff()
        
        turnover_per_row = pd.DataFrame(index=rebal_turnover.index)
        turnover_per_row['sell'] = rebal_turnover[rebal_turnover < 0].sum(axis=1)
        turnover_per_row['buy'] = rebal_turnover[rebal_turnover > 0].sum(axis=1)

        turnover_per_row['Turnover'] = abs(turnover_per_row[['sell', 'buy']].min(axis=1))
        df_turnover = pd.concat([df_turnover, turnover_per_row['Turnover']], axis=1)

    df_turnover = df_turnover.loc[(df_turnover != 0).any(axis=1)]
    df_turnover.index = pd.to_datetime(df_turnover.index).strftime('%Y-%m-%d')
    df_turnover.loc['Avg_turnover'] = df_turnover.mean()
    df_turnover.loc['Ann_turnover'] = df_turnover.loc['Avg_turnover'] * rebal_per_yr
    # df_turnover = df_turnover.applymap('{:.2%}'.format)
    
    ann_turnover_row = df_turnover.loc['Ann_turnover']
    df_turnover = df_turnover.drop(['Avg_turnover', 'Ann_turnover'], axis=0)
    df_turnover = pd.concat([pd.DataFrame(ann_turnover_row).transpose(), df_turnover])

    return df_turnover

def calculate_tracking_error(portfolio_returns, benchmark_returns, freq='daily'):
    portfolio_returns = portfolio_returns.reindex(benchmark_returns.index)
    return_diff = portfolio_returns - benchmark_returns
    if freq=='daily':
        tracking_error = np.std(return_diff) * np.sqrt(261)
    elif freq=='weekly':
        tracking_error = np.std(return_diff) * np.sqrt(52)
    elif freq=='monthly':
        tracking_error = np.std(return_diff) * np.sqrt(12)
    return tracking_error

def performance_summary(portfolio_returns, portfolio_weights, benchmark_rel_perf, benchmark_returns=None, start_date=None, end_date=None, frequency='daily', rebal_per_year=1):
    if frequency=='daily':
        annualized = 261
    if frequency=='weekly':
        annualized = 52
    if frequency=='monthly':
        annualized=12
    # else:
    #     print('input a valid frequency: daily, weekly, monthly')
    #     return None
    
    if type(start_date)==int:
        start_date = datetime.strptime(str(start_date), '%Y%m%d').strftime('%Y-%m-%d')
    if type(end_date)==int:
        end_date = datetime.strptime(str(end_date), '%Y%m%d').strftime('%Y-%m-%d')
    
    portfolio_returns = portfolio_returns.loc[start_date:end_date]
    portfolio_weights = portfolio_weights.loc[start_date:end_date]

    total_period_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    portfolio_names = portfolio_returns.columns
    
    relative_perf_df = calculate_relative_per(portfolio_returns, benchmark_rel_perf)
    
    metrics_dict = {}
    
    for portfolio in portfolio_names:
        cum_ret = ((1 + portfolio_returns[portfolio]).cumprod() - 1)[-1]
        ann_ret = performance_annualized_ret_via_cum(portfolio_returns[portfolio], freq=frequency)
        
        ann_vol = performance_annualized_vol(portfolio_returns[portfolio], freq=frequency)
        sharpe_ratio = performance_sharpe(portfolio_returns[portfolio], freq=frequency)
        sortino_ratio = performance_sortino(portfolio_returns[portfolio], freq=frequency).values[0]
        mdd = performance_mdd(portfolio_returns[portfolio])
        median_rel_perf = relative_perf_df.loc['Median'][portfolio]
        min_rel_perf = relative_perf_df.loc['Min'][portfolio]
        port_corr = portfolio_returns[portfolio].corr(benchmark_rel_perf)
        cum_1y_port = ((1 + portfolio_returns.iloc[annualized * -1:][portfolio]).cumprod() - 1)[-1] if total_period_years >= 1 else np.nan
        cum_3y_port = ((1 + portfolio_returns.iloc[annualized * -3:][portfolio]).cumprod() - 1)[-1] if total_period_years >= 3 else np.nan
        ann_3y_port = ((1+cum_3y_port) ** (annualized/(annualized*3))) - 1 if total_period_years >= 3 else np.nan
        cum_5y_port = ((1 + portfolio_returns.iloc[annualized * -5:][portfolio]).cumprod() - 1)[-1] if total_period_years >= 5 else np.nan
        ann_5y_port = ((1+cum_5y_port) ** (annualized/(annualized*5))) - 1 if total_period_years >= 5 else np.nan
        cum_8y_port = ((1 + portfolio_returns.iloc[annualized * -8:][portfolio]).cumprod() - 1)[-1] if total_period_years >= 8 else np.nan
        ann_8y_port = ((1+cum_8y_port) ** (annualized/(annualized*8))) - 1 if total_period_years >= 8 else np.nan

        ann_to = portfolio_turnover(portfolio_weights[portfolio], rebal_per_yr=rebal_per_year).loc['Ann_turnover'].values[0]
        
        metrics_dict[portfolio] = [cum_ret, ann_ret, cum_1y_port, cum_3y_port, cum_5y_port, cum_8y_port, ann_3y_port, ann_5y_port, ann_8y_port, ann_vol, sharpe_ratio, sortino_ratio, mdd, median_rel_perf, min_rel_perf, port_corr, ann_to]
        
    num_columns = len(portfolio_returns.columns)
    
    if benchmark_returns is not None:
        benchmark_returns = pd.DataFrame(benchmark_returns)
        benchmark_returns = benchmark_returns.loc[start_date:end_date]
        benchmark_names = benchmark_returns.columns

        for idx, (benchmark_col, benchmark_name) in enumerate(zip(benchmark_returns.columns, benchmark_names)):
            benchmark_cum_ret = ((1 + benchmark_returns[benchmark_col]).cumprod() - 1).iloc[-1]
            benchmark_ann_ret = performance_annualized_ret_via_cum(benchmark_returns[benchmark_col], freq=frequency)
            benchmark_ann_vol = performance_annualized_vol(benchmark_returns[benchmark_col], freq=frequency)
            benchmark_sharpe = performance_sharpe(benchmark_returns[benchmark_col], freq=frequency)
            benchmark_sortino = performance_sortino(benchmark_returns[benchmark_col], freq=frequency).values[0]
            benchmark_mdd = performance_mdd(benchmark_returns[benchmark_col])
            bm_median_rel_perf = math.nan
            bm_min_rel_perf = math.nan
            benchmark_corr = math.nan
            benchmark_ann_to = math.nan  
            cum_1y_bm = ((1 + benchmark_returns.iloc[annualized * -1:][benchmark_col]).cumprod() - 1)[-1] if total_period_years >= 1 else np.nan
            cum_3y_bm = ((1 + benchmark_returns.iloc[annualized * -3:][benchmark_col]).cumprod() - 1)[-1] if total_period_years >= 3 else np.nan
            ann_3y_bm = ((1+cum_3y_bm) ** (annualized/(annualized*3))) - 1 if total_period_years >= 3 else np.nan
            cum_5y_bm = ((1 + benchmark_returns.iloc[annualized * -5:][benchmark_col]).cumprod() - 1)[-1] if total_period_years >= 5 else np.nan
            ann_5y_bm = ((1+cum_5y_bm) ** (annualized/(annualized*5))) - 1 if total_period_years >= 5 else np.nan
            cum_8y_bm = ((1 + benchmark_returns.iloc[annualized * -8:][benchmark_col]).cumprod() - 1)[-1] if total_period_years >= 8 else np.nan
            ann_8y_bm = ((1+cum_8y_bm) ** (annualized/(annualized*8))) - 1 if total_period_years >= 8 else np.nan

            metrics_dict[benchmark_name] = [benchmark_cum_ret, benchmark_ann_ret, cum_1y_bm, cum_3y_bm, cum_5y_bm, cum_8y_bm, ann_3y_bm, ann_5y_bm, ann_8y_bm, benchmark_ann_vol, benchmark_sharpe, 
                                            benchmark_sortino, benchmark_mdd, bm_median_rel_perf, bm_min_rel_perf, benchmark_corr, benchmark_ann_to]
        
        num_columns += len(benchmark_returns.columns)
            
    metrics_df = pd.DataFrame(metrics_dict, index=['Cumulative return', 'Annualized return', '1y cumulative return', '3y cumulative return', '5y cumulative return' , '8y cumulative return', '3y ann. return', '5y ann. return', '8y ann. return', 
                                                   'Annualized volatility', 'Sharpe ratio', 'Sortino ratio', 
                                                   'Max drawdown', 'Median relative perf_' + benchmark_rel_perf.name , 'Min relative perf_' + benchmark_rel_perf.name, 'Correlation_'+ benchmark_rel_perf.name,
                                                   'Annualized portfolio turnover'])
    
    cols_2dp_perc = ['Cumulative return', 'Annualized return', '1y cumulative return','3y cumulative return', '5y cumulative return' , '8y cumulative return', '3y ann. return', '5y ann. return', '8y ann. return', 'Annualized volatility', 'Max drawdown', 
                     'Median relative perf_' + benchmark_rel_perf.name , 'Min relative perf_' + benchmark_rel_perf.name, 'Annualized portfolio turnover']
    
    cols_2dp = ['Sharpe ratio', 'Sortino ratio','Correlation_'+ benchmark_rel_perf.name,]
    
    metrics_df.loc[cols_2dp_perc] = metrics_df.loc[cols_2dp_perc].map('{:.2%}'.format)
    metrics_df.loc[cols_2dp] = metrics_df.loc[cols_2dp].map('{:.3}'.format)
    
    
    metrics_df.loc['Start date'] = [start_date] * num_columns
    metrics_df.loc['End date'] = [end_date] * num_columns
    
    column_to_shift = metrics_df.pop(benchmark_rel_perf.name)
    metrics_df.insert(0, benchmark_rel_perf.name, column_to_shift)
    metrics_df = metrics_df.T
    return metrics_df
    

def calculate_relative_per(portfolio_ret, benchmark_ret):
    results_df = pd.DataFrame()
    
    for col in portfolio_ret.columns:
        data = portfolio_ret[col]
        data_concat = pd.concat([data, benchmark_ret], axis=1)
        years = data_concat.index.year.unique()

        for year in years:
            year_data = data_concat[data_concat.index.year == year]
            cumulative_return = ((1 + year_data).cumprod() - 1).iloc[-1]
            relative_perf = cumulative_return[0] - cumulative_return[-1]
            results_df.loc[year, col] = relative_perf
            
    for year in years:
        benchmark_cumulative_return = ((1 + benchmark_ret[benchmark_ret.index.year == year]).cumprod() - 1).iloc[-1]
        results_df.loc[year, 'Benchmark'] = benchmark_cumulative_return

    first_column = results_df.pop(results_df.columns[-1]) 
    results_df.insert(0, first_column.name + '_absolute_ret', first_column)
    results_df = results_df.loc[:, (results_df != 0).any(axis=0)]
    
    results_df.loc['Median'] = results_df.median(axis=0)
    results_df.loc['Min'] = results_df.min(axis=0)
    
    # results_df = results_df.style.format("{:.2%}")
    # results_df = results_df.background_gradient(cmap='RdYlGn', subset=results_df.columns[1:])

    return results_df

def performance_summary_yearly(*args, freq='daily', rf=0):

    results = []
    for data in args:
        years = data.index.year.unique()
        for year in years:
            year_data = data[data.index.year == year]
            cumulative_return = (1 + year_data).cumprod() - 1
            yearly_return = cumulative_return.iloc[-1]
            results.append([year, yearly_return, performance_annualized_vol(year_data, freq), 
                            performance_sharpe(year_data, freq), performance_sortino(year_data, freq).values[0], performance_mdd(year_data)])

    results_df = pd.DataFrame(results, columns=['Year', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'])
    results_df.set_index('Year', inplace=True)

    # results_df['Annual Return'] = results_df['Annual Return'].map('{:.2%}'.format)
    # results_df['Annual Volatility'] = results_df['Annual Volatility'].map('{:.2%}'.format)
    # results_df['Sharpe Ratio'] = results_df['Sharpe Ratio'].map('{:.2f}'.format)
    # results_df['Sortino Ratio'] = results_df['Sortino Ratio'].map('{:.2f}'.format)
    # results_df['Max Drawdown'] = results_df['Max Drawdown'].map('{:.2%}'.format)
    return results_df

def cumulative_ret_attribution(weights, returns, start_date=None, end_date=None):
    weights = weights.loc[start_date:end_date]
    returns = returns.loc[start_date:end_date]
    returns.iloc[0] = 0

    weights = weights.filter(like='_wgt')
    ret_attr = weights.multiply(np.array(returns))
    
    cumulative_ret_attr = ret_attr.expanding().sum()
    cumulative_ret_attr.columns = [col.replace('_wgt', '_ret_attr') for col in cumulative_ret_attr.columns]
    return cumulative_ret_attr


def compare_port_ret_attr(df_all_port_weights, returns, portfolio_names, start_date=None, end_date=None):
    df_all_port_weights = df_all_port_weights.loc[start_date:end_date]
    
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index, format='%Y%m%d')
        
    returns = returns.loc[df_all_port_weights.index[0]:df_all_port_weights.index[-1]]
    returns = returns.loc[start_date:end_date]
    
    df_ret_attr = pd.DataFrame()

    for portfolio in portfolio_names:
        tickers = df_all_port_weights[portfolio].filter(like='_wgt').columns.str.replace('_wgt', '')
    
        returns_filtered = returns[tickers]
        ret_attr = cumulative_ret_attribution(df_all_port_weights[portfolio], returns_filtered)
        ret_attr_sum = ret_attr.iloc[-1]
        df_ret_attr = pd.concat([df_ret_attr, ret_attr_sum], axis=1)
        df_ret_attr.fillna(0, inplace=True)

    df_ret_attr.columns = portfolio_names
    non_zero_rows = df_ret_attr.abs().max(axis=1) > 1e-10
    df_ret_attr = df_ret_attr[non_zero_rows]
    df_ret_attr = df_ret_attr.sort_values(by=df_ret_attr.columns[0], ascending=False)
    return df_ret_attr


def compare_port_alloc(df_all_port_weights, portfolio_names, latest_data=True, start_date=None, end_date=None):
    
    df_all_port_weights = df_all_port_weights.loc[start_date:end_date]
    
    df_wgt = pd.DataFrame()
    
    for portfolio in portfolio_names:
        wgt = df_all_port_weights[portfolio].drop(columns=['reb_flag'])
        if latest_data==True:
            wgt = wgt.iloc[-1]
        else:
            wgt = wgt.mean()
        df_wgt = pd.concat([df_wgt, wgt], axis=1)
        df_wgt.fillna(0, inplace=True)

    df_wgt.columns = portfolio_names
    non_zero_rows = df_wgt.abs().max(axis=1) > 1e-10
    df_wgt = df_wgt[non_zero_rows]
    df_wgt = df_wgt.sort_values(ascending=False, by=df_wgt.columns[0])
    return df_wgt

def exposure_analysis(portfolio_weights, exposure_data, latest_data=True, start_date=None, end_date=None):
    if latest_data==True:
        portfolio_weights = portfolio_weights.loc[start_date:end_date].drop(columns=['reb_flag']).iloc[-1]
    else:
        portfolio_weights = portfolio_weights.loc[start_date:end_date].drop(columns=['reb_flag']).mean()
        
    portfolio_weights = portfolio_weights[portfolio_weights != 0]
    
    portfolio_weights = pd.DataFrame(portfolio_weights).T
    portfolio_weights.columns = [col.replace('_wgt', '') for col in portfolio_weights.columns]
    
    tickers = portfolio_weights.columns
    
    exposure_data_filtered = exposure_data.reindex(columns=tickers)
    exposure_analysis = exposure_data_filtered.multiply(np.array(portfolio_weights), axis='columns').sum(axis=1)
    exposure_analysis = pd.DataFrame(exposure_analysis)
    exposure_analysis.columns = ['Exposure %']
    # exposure_analysis = exposure_analysis.sort_values(ascending=False, by='Exposure %')
    return exposure_analysis

def compare_alloc_diff(df_all_port_weights, benchmark_name, portfolio_names, latest_data=True, start_date=None, end_date=None):
    benchmark_weights = df_all_port_weights[benchmark_name]
    if latest_data==True:
        benchmark_weights = benchmark_weights.loc[start_date:end_date].drop(columns=['reb_flag']).iloc[-1]
    
    else:
        benchmark_weights = benchmark_weights.loc[start_date:end_date].drop(columns=['reb_flag']).mean()
    
    df_wgt_diff = pd.DataFrame()
    
    for portfolio in portfolio_names:
        if latest_data==True:
            portfolio_weights = df_all_port_weights[portfolio].loc[start_date:end_date].drop(columns=['reb_flag']).iloc[-1]
           
        else:
            portfolio_weights = df_all_port_weights[portfolio].loc[start_date:end_date].drop(columns=['reb_flag']).mean()
        concatenated = pd.concat([portfolio_weights, benchmark_weights], axis=1).fillna(0)   
        allocation_diff = concatenated.iloc[:, 0] - concatenated.iloc[:, 1]
        allocation_diff = pd.Series(allocation_diff.values, index=allocation_diff.index, name=portfolio)
        df_wgt_diff = pd.concat([df_wgt_diff, allocation_diff], axis=1)
        
    benchmark_col = pd.Series(benchmark_weights, index=benchmark_weights.index,name=benchmark_name)
    df_wgt_diff = pd.concat([benchmark_col, df_wgt_diff], axis=1)
    df_wgt_diff = df_wgt_diff.fillna(0)
    df_wgt_diff = df_wgt_diff.loc[:, (df_wgt_diff != 0).any(axis=0)]
    df_wgt_diff = df_wgt_diff.style.format("{:.2%}")
    # df_wgt_diff = df_wgt_diff.background_gradient(cmap='RdYlGn', subset=df_wgt_diff.columns[1:])

    return df_wgt_diff

def compare_exposure_diff(exposure_portfolio, exposure_benchmark, portfolio_names,start_date=None, end_date=None):

    df_exp_diff = pd.DataFrame()
    
    for portfolio in portfolio_names:
        portfolio_exp = exposure_portfolio[portfolio]
        concatenated = pd.concat([portfolio_exp, exposure_benchmark], axis=1).fillna(0)   
        exp_diff = concatenated.iloc[:, 0] - concatenated.iloc[:, 1]
        exp_diff = pd.Series(exp_diff.values, index=exp_diff.index, name=portfolio)
        df_exp_diff = pd.concat([df_exp_diff, exp_diff], axis=1)
        
    benchmark_col = pd.Series(exposure_benchmark, index=exposure_benchmark.index,name='Benchmark_' + exposure_benchmark.name )
    df_exp_diff = pd.concat([benchmark_col, df_exp_diff], axis=1)
    df_exp_diff = df_exp_diff.fillna(0)
    df_exp_diff = df_exp_diff.loc[:, (df_exp_diff != 0).any(axis=0)]
    df_exp_diff = df_exp_diff * 100
    df_exp_diff = df_exp_diff.map(lambda x: '{:.2f}'.format(x))
    # df_exp_diff = df_exp_diff.background_gradient(cmap='RdYlGn', subset=df_exp_diff.columns[1:])

    return df_exp_diff


def forward_regression(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True, start_date=None, end_date=None):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded,dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()  # Use idxmin to get the index of the minimum value
            included.append(best_feature)
            changed = True
            # if verbose:
                # print(f'Add {best_feature} with p-value {best_pval}')
        if not changed:
            break
    return included

def factor_regression(portfolio_returns, factors_data, threshold_in=0.05, start_date=None, end_date=None):
    
    if portfolio_returns.index.dtype == 'int64':
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index, format='%Y-%m-%d')
    data = pd.concat([portfolio_returns, factors_data], axis=1).dropna()
    
    try:
        data['excess_ret'] = portfolio_returns - data['RF']
    except KeyError:
        data['excess_ret'] = portfolio_returns
    
    start_index = data.index[0]
    end_index = data.index[-1]
    # display(start_index, end_index)
    # display(factors_data.loc[start_index:end_index], data['excess_ret'])
    
    col_factors = [col for col in factors_data.columns if col not in ['excess_ret', 'RF']]
    # display( data)
    factors = forward_regression(factors_data.loc[start_index:end_index][col_factors], data.loc[start_index:end_index]['excess_ret'], threshold_in=threshold_in)
    
    data.rename(columns={'Mkt-RF': 'Mkt'}, inplace=True)
    formula = 'excess_ret ~ ' + ' + '.join(factors)
    # print(formula)
    five_factor_model = smf.ols(formula=formula, data=data).fit()
    return five_factor_model

def factor_exposure_coefficient(ret_data, factor_data, tickers, start_date=None, end_date=None):
    df_params = pd.DataFrame()
    for ticker in tickers:
        portfolio_regression = factor_regression(ret_data[ticker], factor_data)
        temp = pd.DataFrame([portfolio_regression.params], index=[ticker])
        df_params = pd.concat([df_params, temp])
    df_params = df_params.drop(columns=['Intercept'])
    df_params.fillna(0, inplace=True)
    df_params = df_params.map('{:.2f}'.format)
    return df_params


def performance_summary_constituents(returns, start_date=None, end_date=None, frequency='daily'):
    
    if frequency=='daily':
        annualized = 261
    if frequency=='weekly':
        annualized = 52
    if frequency=='monthly':
        annualized=12
        
    start_date = returns.index[0]
    end_date = returns.index[-1]
    
    # Filter returns within the specified date range
    returns = returns.loc[start_date:end_date]

    total_period_years = (end_date - start_date).days / 365

    metrics_dict = {}
    for name in returns.columns:
        cum_ret = ((1 + returns[name]).cumprod() - 1).iloc[-1]
        ann_ret = performance_annualized_ret_via_cum(returns[name], freq=frequency)
        ann_vol = performance_annualized_vol(returns[name], freq=frequency)
        sharpe_ratio = performance_sharpe(returns[name], freq=frequency)
        sortino_ratio = performance_sortino(returns[name], freq=frequency).values[0]
        mdd = performance_mdd(returns[name])
        cum_1y = ((1 + returns.iloc[-annualized:] [name]).cumprod() - 1).iloc[-1] if total_period_years >= 1 else np.nan
        cum_3y = ((1 + returns.iloc[-annualized * 3:][name]).cumprod() - 1).iloc[-1] if total_period_years >= 3 else np.nan
        ann_3y = ((1 + cum_3y) ** (annualized / (annualized * 3)) - 1) if total_period_years >= 3 else np.nan
        cum_5y = ((1 + returns.iloc[-annualized * 5:][name]).cumprod() - 1).iloc[-1] if total_period_years >= 5 else np.nan
        ann_5y = ((1 + cum_5y) ** (annualized / (annualized * 5)) - 1) if total_period_years >= 5 else np.nan
        cum_8y = ((1 + returns.iloc[-annualized * 8:][name]).cumprod() - 1).iloc[-1] if total_period_years >= 8 else np.nan
        ann_8y = ((1 + cum_8y) ** (annualized / (annualized * 8)) - 1) if total_period_years >= 8 else np.nan

        
        metrics_dict[name] = [cum_ret, ann_ret, cum_1y, cum_3y, cum_5y, cum_8y, ann_3y, ann_5y, ann_8y, ann_vol, sharpe_ratio, sortino_ratio, mdd]

    metrics_df = pd.DataFrame(metrics_dict, index=['Cumulative return', 'Annualized return', '1y cumulative return', '3y cumulative return', '5y cumulative return' , '8y cumulative return', '3y ann. return', '5y ann. return', '8y ann. return', 'Annualized volatility', 'Sharpe ratio', 'Sortino ratio', 'Max drawdown'])

    # Format numerical columns
    cols_2dp_perc = ['Cumulative return', 'Annualized return', '1y cumulative return','3y cumulative return', '5y cumulative return' , '8y cumulative return', '3y ann. return', '5y ann. return', '8y ann. return', 'Annualized volatility', 'Max drawdown']
    
    cols_2dp = ['Sharpe ratio', 'Sortino ratio']
    
    metrics_df.loc[cols_2dp_perc] = metrics_df.loc[cols_2dp_perc].astype(float).map('{:.2%}'.format)
    metrics_df.loc[cols_2dp] = metrics_df.loc[cols_2dp].map('{:.3}'.format)

    num_columns = len(returns.columns)
    metrics_df.loc['Start date'] = [start_date] * num_columns
    metrics_df.loc['End date'] = [end_date] * num_columns

    return metrics_df

def summary_returns_constituents(returns_df, frequency='daily', rf=0):
    returns_df.index = pd.to_datetime(returns_df.index, format='%Y%m%d')
    results = {}
    
    for col in returns_df.columns:
        data = returns_df[col]
        years = data.index.year.unique()
        
        for year in years:
            year_data = data[data.index.year == year]
            cumulative_return = (1 + year_data).cumprod() - 1
            yearly_return = cumulative_return.iloc[-1]
            if year not in results:
                results[year] = {}
            results[year][col] = yearly_return        
    
    yearly_results_df = pd.DataFrame(results).T
    
    cumulative_return = ((1 + returns_df).cumprod() - 1).iloc[-1]
    annualized_return = performance_annualized_ret_via_cum(returns_df, freq=frequency)
    
    summary_results_df = pd.concat([cumulative_return, annualized_return], axis=1).T
    summary_results_df.index = ['Cumulative returns', 'Annualized return']
    
    display(summary_results_df.map('{:.2%}'.format))
    
    return yearly_results_df


def constituents_calendar_year_returns(returns_df, frequency='daily', rf=0):
    returns_df.index = pd.to_datetime(returns_df.index, format='%Y%m%d')
    results = {}
    
    for col in returns_df.columns:
        data = returns_df[col]
        years = data.index.year.unique()
        
        for year in years:
            year_data = data[data.index.year == year]
            cumulative_return = (1 + year_data).cumprod() - 1
            yearly_return = cumulative_return.iloc[-1]
            if year not in results:
                results[year] = {}
            results[year][col] = yearly_return        
    
    yearly_results_df = pd.DataFrame(results).T
    
    return yearly_results_df


def summary_vol_constituents(returns_df, frequency='daily', rf=0):
    returns_df.index = pd.to_datetime(returns_df.index, format='%Y%m%d')
    results = {}
    
    for col in returns_df.columns:
        data = returns_df[col]
        years = data.index.year.unique()
        
        for year in years:
            year_data = data[data.index.year == year]
            yearly_std = performance_annualized_vol(year_data, freq=frequency)
            if year not in results:
                results[year] = {}
            results[year][col] = yearly_std        
    
    results_df = pd.DataFrame(results).T
        
    results_df.loc['Annualized volatility'] = performance_annualized_vol(returns_df, freq=frequency)
    results_df = pd.DataFrame(np.roll(results_df.values, 1, axis=0), index=np.roll(results_df.index, 1), columns=results_df.columns)
    results_df = results_df.map(lambda x: '{:.2%}'.format(x))

    return results_df


def median_credit_quality(df_wgt, credit_qual_breakdown, bonds_list):
    credit_qual_df = exposure_analysis_mixed(df_wgt,credit_qual_breakdown, bonds_list, latest_data=True)
    credit_qual_df['cumsum'] = credit_qual_df['Exposure %'].cumsum()
    median = credit_qual_df.index[credit_qual_df['cumsum'] >= 0.5][0]
    return median

def exposure_dcm(df_all_port_weights, etf_dcm, etf_credit_quality, bonds_additional, portfolio_names):
    mat_dur_df_all = pd.DataFrame()

    for core_name in portfolio_names:
        mat_dur_df = exposure_analysis_mixed(df_all_port_weights[core_name], etf_dcm.loc[['Duration', 'Maturity', 'YTM', 'T12M div yield']], bonds_additional, latest_data=True, start_date=start_date_dt, end_date=end_date_dt)
        mat_dur_df.columns = [core_name]
        mat_dur_df_all = pd.concat([mat_dur_df_all, mat_dur_df], axis=1)

    cred_qual_list = []
    for core_name in portfolio_names:
        cred_qual_list.append(median_credit_quality(df_all_port_weights[core_name], etf_credit_quality, bonds_additional))

    mat_dur_df_all.loc['Avg credit quality'] = cred_qual_list
    mat_dur_df_all = pd.concat([etf_dcm[benchmark_tickers] , mat_dur_df_all], axis=1)
    
    mat_dur_df_all.loc[['YTM', 'T12M div yield']] = mat_dur_df_all.loc[['YTM', 'T12M div yield']].map('{:.2%}'.format)
    mat_dur_df_all.loc[['Duration', 'Maturity']] = mat_dur_df_all.loc[['Duration', 'Maturity']].map('{:.3}'.format)
    
    return mat_dur_df_all
    

def plot_cumulative_returns(*args, start_date=None, end_date=None, title='Cumulative Return', 
                            x_axis_title='Date', y_axis_title='Cumulative Return %', width=1000, height=600, show_data=True):
    '''
    input: returns data
    plots cumulative return over time
    '''
    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    df_combined.iloc[0] = 0
    df_combined = (1 + df_combined).cumprod() - 1
    df_combined = df_combined.fillna(method='ffill')
    
    # Resample to weekly intervals
    # df_combined = df_combined.resample('W-Mon').last()
    
    start_date = str(df_combined.index[0].date())
    end_date = str(df_combined.index[-1].date())
    
    fig = px.line(df_combined, title=title + ' (' + start_date + ' to ' + end_date + ')', labels={'variable': 'Asset'})
    fig.add_hline(y=0, line_width=1, line_dash='dash', line_color='red')
    
    fig.update_layout(xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title,
                      yaxis_tickformat='.2%',
                      width=width,
                      height=height)
    
    fig.show()
    
    if show_data==True:
        df_combined = pd.DataFrame(df_combined.iloc[-1]).map('{:.2%}'.format)
        df_combined.columns = ['Cumulative Return']
        return df_combined

def plot_outperformance_vs_benchmark(portfolio_returns, benchmark_returns, start_date=None, end_date=None, title='Relative Performance vs Benchmark',width=1000, height=600,show_data=True):
    
    # input: returns data. plots relative performance of portfolio vs benchmark
    
    portfolio_returns = portfolio_returns.loc[start_date:end_date]
    benchmark_returns = benchmark_returns.loc[start_date:end_date]
    
    portfolio_returns = (1 + portfolio_returns).cumprod() - 1
    benchmark_returns = (1 + benchmark_returns).cumprod() - 1
    benchmark_name = benchmark_returns.name
    
    outperformance = pd.DataFrame()
    
    for i in portfolio_returns.columns:
        outperformance['diff_' + i + '_' + benchmark_name] = portfolio_returns[i] - benchmark_returns
    
    fig = px.area(outperformance, title=title, labels={'variable': 'Asset'})
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Relative returns %',
        yaxis_tickformat='.2%',
        width=width,
        height=height,
        showlegend=True)


    fig.show()
    
    if show_data == True:
        outperformance_end = pd.DataFrame(outperformance.iloc[-1]).map('{:.2%}'.format)
        outperformance_end.columns = ['Relative Performance']
        return outperformance_end
    
def plot_annual_returns(*args, start_date=None, end_date=None, width=1000, height=600, show_data=True):
    
    # combine series to a single df
    dfs = [df.groupby(df.index.year).sum() for df in args]
    df_combined = pd.concat(dfs, axis=1)
    df_combined = df_combined.loc[start_date:end_date]

    fig = px.bar(df_combined, labels={'x': 'Year', 'y': 'Annual Return'}, title="Annual Return", barmode='group')

    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=df_combined.index.min(),
            dtick=1,
            tickangle=45
        ),
        yaxis_title='Annual Return',
        yaxis_tickformat='.2%',
        width=width,
        height=height
    )
    fig.show()
    if show_data==True:
        return df_combined.map('{:.2%}'.format)
    
def plot_weights_and_turnover(weights, start_date=None, end_date=None, show_data=True, show_turnover=True, show_rebal=False, height=800, width=1200):
    weights = weights.loc[start_date:end_date]
    rebalance_dates = weights[weights['reb_flag']].index
    weights_data = weights.drop(columns=['reb_flag'])
    weights_data = weights_data.loc[:, (weights_data != 0).any(axis=0)]

    fig = go.Figure()

    for column in weights_data.columns:
        fig.add_trace(go.Scatter(x=weights_data.index, y=weights_data[column], mode='lines', stackgroup='one', name=column))

    if show_turnover:
        turnover_data = portfolio_turnover(weights).iloc[1:]
        fig.add_trace(go.Scatter(x=turnover_data.index, y=turnover_data['Turnover'], mode='lines+markers', name='Turnover', yaxis='y2', line=dict(width=2, color='black')))

    if show_rebal==True:
        for date in rebalance_dates:
            fig.add_vline(type='line', x=date, line=dict(color='gray', width=1, dash='dash'), name='Rebalance')
            
    fig.update_layout(title='Historical Weights and Turnover',
                      xaxis_title='Date',
                      yaxis=dict(title='Weights', tickformat='.2%'),
                      yaxis2=dict(title='Turnover', overlaying='y', side='right', range=[0, 0.5], tickformat='.2%'),
                      legend=dict(x=1.1),
                      height=height,
                      width=width)

    fig.show()
    
    if show_data:
        avg_wgt = pd.DataFrame(weights_data.mean(axis=0))
        avg_wgt = avg_wgt.loc[(avg_wgt != 0).any(axis=1)]
        avg_wgt = avg_wgt.sort_values(by=avg_wgt.columns[0], axis=0, ascending=False).map('{:.2%}'.format)
        avg_wgt.columns = ['Average allocation']
        return avg_wgt

    
def plot_drawdowns(*args, start_date=None, end_date=None, title='Drawdown', show_data=True):

    # input: returns data. plots cumulative drawdown and max drawdown line
    
    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]

    num_series = len(args)
    fig = make_subplots(rows=1, cols=1)
    
    df_mdd = pd.DataFrame()
    
    for i, col in enumerate(df_combined.columns):
        data = df_combined[col]
        ret_cum = (1 + data).cumprod()
        dd = 1 - ret_cum.div(ret_cum.cummax())
        mdd = dd.expanding().max()
        df_mdd = pd.concat([df_mdd, mdd], axis=1)

        # Plot Drawdown Area
        drawdown_area = go.Scatter(x=dd.index, y=-dd, fill='tozeroy', mode='none', stackgroup=f'dd{col}', name=f'{col}')
        drawdown_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        fig.add_trace(drawdown_area, row=1, col=1)
        
#         # Plot MDD Line
#         mdd_line = go.Scatter(x=mdd.index, y=-mdd, mode='lines', line=dict(color=drawdown_color), name=f'MDD {col}: {mdd.max():.2%}')
#         fig.add_trace(mdd_line, row=1, col=1)

    fig.update_layout(xaxis=dict(title='Date'), yaxis=dict(title='Drawdown'), yaxis_tickformat='.2%',
                      title_text=title, showlegend=True, height=500,width=1000)
    fig.show()
    
    if show_data==True:
        df_mdd = df_mdd.max(axis=0)
        df_mdd = pd.DataFrame(df_mdd)
        df_mdd.columns = ['Max Drawdown']
        return df_mdd.map('{:.2%}'.format)
    
def plot_returns_distribution_histogram(*args, start_date=None, end_date=None, title='Returns Distribution', width=1000, height=600):
    
    # input: returns data. Plot returns distribution histogram    

    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    df_melted = df_combined.melt(var_name='Asset', value_name='Returns')

    fig = px.histogram(df_melted, x='Returns', color='Asset', title=title, width=width, height=height)
    fig.update_layout(xaxis_tickformat=".2%")
    fig.show()
    
def plot_returns_distribution_boxplot(*args, start_date=None, end_date=None, title='Returns Distribution', width=1000, height=600):
    
    # input: returns data. Plot returns distribution using boxplot

    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    df_melted = df_combined.melt(var_name='Asset', value_name='Returns')

    fig = px.box(df_melted, x='Asset', y='Returns', color='Asset', title=title, width=width, height=height)
    fig.update_layout(yaxis_tickformat=".2%")
    fig.show()

def plot_rolling_volatility(*args, start_date=None, end_date=None, window=261, title='Rolling Volatility', width=1000, height=600):

    # input: returns data. Plot rolling volatility

    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    df_combined = df_combined.fillna(0)
    rolling_volatility = df_combined.rolling(window=window).std() * np.sqrt(window)
    
    fig = px.line(rolling_volatility, x=rolling_volatility.index, y=rolling_volatility.columns,
                      title=title, width=width, height=height, labels={'variable': 'Asset'})
    fig.update_layout(yaxis_tickformat=".2%")
    fig.show()
    
def plot_correlation_heatmap(*args, start_date=None, end_date=None, title='Correlation Heatmap', width=1000, height=600):
    
    # input: returns data. Plot correlation matrix as a heatmap.
    
    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    correlation_matrix = df_combined.corr().round(3)

    fig = px.imshow(correlation_matrix, text_auto=True, title=title, width=width, height=height)
    fig.show()
        
def plot_rolling_correlation(*args, benchmark, start_date=None, end_date=None, window=52, title='Rolling Correlation', width=1000, height=600):
    
    df_combined = pd.concat(args, axis=1)
    df_combined = df_combined.loc[start_date:end_date]
    df_combined = df_combined.dropna()
    benchmark = benchmark.dropna()
    
    fig = go.Figure()

    for col in df_combined.columns:
        df_concat = pd.concat([df_combined[col], benchmark], axis=1)
        df_concat = df_concat.fillna(0)
        rolling_corr = df_concat.iloc[:, 0].rolling(window=window).corr(df_concat.iloc[:, 1])
        fig.add_trace(go.Scatter(x=df_combined.index, y=rolling_corr, mode='lines', name=f'Rolling Corr {col} & ' + benchmark.name))

    fig.update_layout(title=title, xaxis_title='Date',yaxis_title='Correlation', width=width, height=height)

    fig.show()
    
def plot_ret_attribution(weights, returns, start_date=None, end_date=None, title='Cumulative Return Attribution', width=1000, height=600, show_data=True):
    df_ret_attr = cumulative_ret_attribution(weights, returns, start_date=start_date, end_date=end_date)
    df_ret_attr = df_ret_attr.loc[:, (df_ret_attr != 0).any(axis=0)]
    
    fig = go.Figure()
    for ticker in df_ret_attr.columns:
        fig.add_trace(go.Scatter(x=df_ret_attr.index, y=df_ret_attr[ticker], mode='lines', name=ticker))

    fig.update_layout(title=title,
                      xaxis_title='Date',
                      yaxis_title='Return Attribution',
                      yaxis_tickformat='.2%',
                      legend_title='Ticker',
                      width=width,
                      height=height)

    fig.show()
    
    if show_data==True:
        df_ret_attr_total = pd.DataFrame(df_ret_attr.iloc[-1])
        df_ret_attr_total.columns = ['Return attribution']
        df_ret_attr_total = df_ret_attr_total.sort_values(by='Return attribution', axis=0, ascending=False).map('{:.2%}'.format)
        return df_ret_attr_total
    
def plot_pie_chart(series, title=None, width=1000, height=600):

    fig = go.Figure(data=[go.Pie(labels=series.index, values=series, textinfo='label+percent', hole=0.3)])
    fig.update_layout(title=title, width=width, height=height, showlegend=True)
    fig.show()
    
def plot_pie_chart_topX(series, title, topX=7, width=1000, height=600):
    top_x = series.sort_values(ascending=False).head(topX)
    other_sum = series.sum() - top_x.sum()
    
    labels = list(top_x.index) + ['Others']
    values = list(top_x) + [other_sum]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', hole=0.3)])
    fig.update_layout(title=title, width=width, height=height, showlegend=True)
    fig.show()
    
    
def plot_exposure_diff(portfolio_exposure, benchmark_exposure, portfolio_names, start_date=None, end_date=None, width=1500, height=800):
    
    benchmark_name = benchmark_exposure.name
    if type(portfolio_names)is str:
        portfolio_names = [portfolio_names]
    
    fig = go.Figure()

    for portfolio in portfolio_names:
        concatenated = pd.concat([portfolio_exposure[portfolio], benchmark_exposure], axis=1)
        concatenated = concatenated.fillna(0)
        exposure_diff = concatenated.iloc[:, 0] - concatenated.iloc[:, 1]
        exposure_diff = exposure_diff[exposure_diff != 0]
        exposure_diff = exposure_diff[::-1]
        
        fig.add_trace(go.Bar(
            y=exposure_diff.index,  
            x=exposure_diff.values, 
            name=portfolio,
            orientation='h',
            text=[f'{val * 100:.2f}%' for val in exposure_diff.values],
            textposition='outside'
        ))

    fig.update_layout(title="Exposure Difference vs " + benchmark_name, yaxis_title="Exposure", xaxis_title="Weight diff %", xaxis_tickformat='.2%', 
                      bargap=0.3, showlegend=True, width=width, height=height)

    fig.show()

def plot_exposure_diff_topX(portfolio_exposure, benchmark_exposure, portfolio_names, topX=10, start_date=None, end_date=None, width=1500, height=800):
    
    benchmark_name = benchmark_exposure.name
    if isinstance(portfolio_names, str):
        portfolio_names = [portfolio_names]
    
    fig = go.Figure()

    for portfolio in portfolio_names:
        concatenated = pd.concat([portfolio_exposure[portfolio], benchmark_exposure], axis=1)
        concatenated = concatenated.fillna(0)
        
        top_exposures = concatenated.sort_values(by=portfolio, ascending=False).head(topX)
        top_exposures_sum = top_exposures.sum()
        top_exposures.loc['Others'] = 1 - top_exposures_sum
        top_exposures = top_exposures[::-1]
        
        exposure_diff = top_exposures.iloc[:, 0] - top_exposures.iloc[:, 1]
        exposure_diff = exposure_diff[exposure_diff != 0]
        
        fig.add_trace(go.Bar(
            y=exposure_diff.index,  
            x=exposure_diff.values, 
            name=portfolio,
            orientation='h',
            text=[f'{val * 100:.2f}%' for val in exposure_diff.values],
            textposition='outside'
        ))

    fig.update_layout(title="Exposure Difference vs " + benchmark_name, yaxis_title="Exposure", xaxis_title="Weight diff %", xaxis_tickformat='.2%', 
                      bargap=0.3, showlegend=True, width=width, height=height)

    fig.show()

def plot_alloc_diff(df_all_port_weights, benchmark_weights, portfolio_names, start_date=None, end_date=None, width=1500, height=800):
    
    if type(portfolio_names)is str:
        portfolio_names = [portfolio_names]
    
    fig = go.Figure()
    benchmark_weights = benchmark_weights.loc[start_date:end_date].drop(columns=['reb_flag']).mean()

    for portfolio_name in portfolio_names:
        portfolio_weights = df_all_port_weights[portfolio_name].loc[start_date:end_date].drop(columns=['reb_flag']).mean()
        concatenated = pd.concat([portfolio_weights, benchmark_weights], axis=1)
        concatenated = concatenated.fillna(0)
        allocation_diff = concatenated.iloc[:, 0] - concatenated.iloc[:, 1]
        
        allocation_diff = allocation_diff[allocation_diff != 0]
        
        fig.add_trace(go.Bar(
            y=allocation_diff.index,  
            x=allocation_diff.values, 
            name=portfolio_name,
            orientation='h',
            text=[f'{val * 100:.2f}%' for val in allocation_diff.values],
            textposition='outside'
        ))

    fig.update_layout(title="Portfolio Allocation Diff", yaxis_title="Asset", xaxis_title="Weight diff %", xaxis_tickformat='.2%', 
                      bargap=0.3, showlegend=True, width=width, height=height)

    fig.show()
    
def plot_return_risk_scatter_year(returns_df, freq='daily', periods=[1, 5], show_data=True, width=1500, height=800):
    
    if freq == 'daily':
        annualized = 261
    elif freq == 'weekly':
        annualized = 52
    elif freq == 'monthly': 
        annualized = 12
    else:
        print('Enter a valid frequency: daily, weekly, monthly')
        return
    
    for period in periods:
        # calculate annualized volatility and return
        annualized_volatility = performance_annualized_vol(returns_df.iloc[-period * annualized:], freq=freq)
        annualized_return = performance_annualized_ret_via_cum(returns_df.iloc[-period * annualized:], freq=freq)
        
        scatter_plot = go.Scatter(
            x=annualized_volatility,
            y=annualized_return,
            mode='markers+text',
            text=returns_df.columns, 
            textposition='top center',
            marker=dict(size=8, opacity=1),
            textfont=dict(size=10), 
            name=str(period) + ' Year'
        )

        layout = go.Layout(
        title='Return v Risk: ' + str(period) + ' Year', 
        xaxis=dict(title='Annualized Volatility', range=[-1.4 * abs(min(annualized_return)), max(max(annualized_volatility), max(annualized_return)) * 1.2]), 
        yaxis=dict(title='Annualized Return', range=[-1.4 * abs(min(annualized_return)), max(max(annualized_volatility), max(annualized_return)) * 1.2]), 
        xaxis_tickformat='.2%', 
        yaxis_tickformat='.2%', 
        width=width, 
        height=height,
        shapes=[
            dict(
                type='line',
                x0=-2 * abs(min(annualized_return)),
                y0=-2 * abs(min(annualized_return)),
                x1=max(max(annualized_volatility), max(annualized_return)) * 2,
                y1=max(max(annualized_volatility), max(annualized_return)) * 2,
                line=dict(color='red', width=1),
            )
        ]
    )

        fig = go.Figure(data=scatter_plot, layout=layout)
        fig.show()

        if show_data == True:
            results_df = performance_summary_constituents(returns_df.iloc[-period * annualized:], frequency=freq)
            results_df = results_df.loc[['Annualized return', 'Annualized volatility', 'Sharpe ratio']]
            results_df.index = [f"{period}Y_{row}" for row in results_df.index]
            display(results_df.pipe(freeze))

def plot_return_risk_scatter_maxrange(returns_df, freq='daily', show_data=True, width=1500, height=800):
    
    if freq == 'daily':
        annualized = 261
    elif freq == 'weekly':
        annualized = 52
    elif freq == 'monthly': 
        annualized = 12
    else:
        print('Enter a valid frequency: daily, weekly, monthly')
        return
    
    # calculate annualized volatility and return
    annualized_volatility = performance_annualized_vol(returns_df, freq=freq)
    annualized_return = performance_annualized_ret_via_cum(returns_df, freq=freq)
    
    scatter_plot = go.Scatter(
        x=annualized_volatility,
        y=annualized_return,
        mode='markers+text',
        text=returns_df.columns, 
        textposition='top center',
        marker=dict(size=8, opacity=1),
        textfont=dict(size=10), 
    )

    layout = go.Layout(
    title='Return v Risk',
    xaxis=dict(title='Annualized Volatility', range=[-1.4 * abs(min(annualized_return)), max(max(annualized_volatility), max(annualized_return)) * 1.2]), 
    yaxis=dict(title='Annualized Return', range=[-1.4 * abs(min(annualized_return)), max(max(annualized_volatility), max(annualized_return)) * 1.2]), 
    xaxis_tickformat='.2%', 
    yaxis_tickformat='.2%', 
    width=width, 
    height=height,
    shapes=[
        dict(
            type='line',
            x0=-2 * abs(min(annualized_return)),
            y0=-2 * abs(min(annualized_return)),
            x1=max(max(annualized_volatility), max(annualized_return)) * 2,
            y1=max(max(annualized_volatility), max(annualized_return)) * 2,
            line=dict(color='red', width=1),
        )
    ])

    fig = go.Figure(data=scatter_plot, layout=layout)
    fig.show()

    if show_data == True:
        results_df = performance_summary_constituents(returns_df, frequency=freq)
        results_df = results_df.loc[['Annualized return', 'Annualized volatility', 'Sharpe ratio']]
        display(results_df.pipe(freeze))


def monthly_performance_table(portfolio_returns):
       # Resample returns to monthly frequency and calculate returns
       monthly_rets = portfolio_returns.resample('M').apply(lambda x: (x + 1).prod() - 1)
       
       # Pivot the table to have years as rows and months as columns
       monthly_rets_table = monthly_rets.pivot_table(index=monthly_rets.index.year, columns=monthly_rets.index.month, values=monthly_rets.columns[0]) 
       
       # Rename columns to represent months
       monthly_rets_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
       
       return monthly_rets_table
        
def plot_factor_exposure(df_factor_port_exp, df_factor_benchmark_exp, portfolio_name, benchmark_ticker='URTH', show_data=True, width=1000, height=600):

    df_factor_port_exp = df_factor_port_exp.T
    df_factor_benchmark_exp = df_factor_benchmark_exp.T
    
    if 'Constraints' in df_factor_port_exp.index:
        df_factor_port_exp = df_factor_port_exp.drop('Constraints')
        
    
    df_concat = pd.concat([df_factor_benchmark_exp[benchmark_ticker], df_factor_port_exp],axis=1)
    df_concat = df_concat.fillna(0)
    df_concat = df_concat.apply(pd.to_numeric, errors='coerce')
    
    benchmark_col = df_concat.iloc[:, 0]  
    port_col = df_concat.iloc[:, 1:]  
    
    factors = df_concat.index
    
    df_concat = df_concat[[portfolio_name, benchmark_ticker]]
    
    portfolio_trace = go.Bar(
        x=df_concat[portfolio_name].iloc[::-1],
        y=factors[::-1],
        name=portfolio_name,
        orientation='h'
    )
    
    benchmark_trace = go.Scatter(
        x=df_concat[benchmark_ticker].iloc[::-1],
        y=factors[::-1],
        mode='markers',
        name='Benchmark',
        marker=dict(symbol='circle', size=10)
    )
    
    layout = go.Layout(title='Factor Exposure', xaxis=dict(range=[-1,1]), yaxis=dict(title='Factors'), width=width, height=height)
    
    data = [portfolio_trace, benchmark_trace]
    
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    if show_data == True:
        return df_concat
    
    
def plot_linechart_over_time(df, width=1000, height=600):
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    fig.update_layout(xaxis_title='Date', width=width, height=height)

    fig.show()

def plot_scatter(x, y, title=None):
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

    fig.update_layout(
        title=title,
        xaxis_title=x.name,
        yaxis_title=y.name,
        width=800, height=600,
    )

    fig.show()
    

def asset_class_view(df_wgt, bonds_list, equity_list, gold_list):
    reb_flag = df_wgt[['reb_flag']]
    df_wgt = df_wgt.drop(columns='reb_flag')
    df_wgt.columns = df_wgt.columns.str.replace('_wgt', '')
    
    df_wgt_assetclass = pd.DataFrame()
    df_wgt_assetclass['Equity'] = df_wgt[equity_list].sum(axis=1)
    df_wgt_assetclass['Fixed_income'] = df_wgt[bonds_list].sum(axis=1)
    df_wgt_assetclass['Gold'] = df_wgt[gold_list].sum(axis=1)
    
    df_wgt_assetclass = pd.concat([reb_flag, df_wgt_assetclass], axis=1)
    
    return df_wgt_assetclass


def exposure_over_time(df_weight, df_exposure, start_date=None, end_date=None, frequency=None):
    df_weight = df_weight.loc[start_date:end_date]
    
    df_weight.columns = [col.replace('_wgt', '') for col in df_weight.columns]
    
    exposure_analysis = pd.DataFrame()
    
    for index, row in df_weight.iterrows():
        tickers = row.index
        weights = row.values
        
        exposure_data_filtered = df_exposure.reindex(columns=tickers)
        exposure_row = exposure_data_filtered.multiply(weights, axis='columns').sum(axis=1)
        
        exposure_analysis = pd.concat([exposure_analysis, exposure_row.rename(index)], axis=1)
    
    exposure_analysis = exposure_analysis.T
    exposure_analysis = exposure_analysis.loc[:, (exposure_analysis != 0).any(axis=0)]
    
    if frequency == 'monthly':
        exposure_analysis = exposure_analysis.resample('M').mean()
    elif frequency == 'quarterly':
        exposure_analysis = exposure_analysis.resample('Q').mean()
    elif frequency == 'yearly':
        exposure_analysis = exposure_analysis.resample('Y').mean()

    exposure_analysis = exposure_analysis[exposure_analysis.columns[exposure_analysis.iloc[0].argsort()[::-1]]]
    # exposure_analysis = exposure_analysis.applymap(lambda x: '{:.2%}'.format(x))
    return exposure_analysis

def relative_exposure_over_time(df_weight, df_exposure, benchmark_exposure, start_date=None, end_date=None, frequency=None):
    benchmark_exposure = benchmark_exposure.T
    benchmark_exposure = benchmark_exposure.loc[:, (benchmark_exposure != 0).any(axis=0)]
        
    exposure_analysis = exposure_over_time(df_weight, df_exposure, start_date, end_date, frequency)
    
    relative_exposure = exposure_analysis.subtract(benchmark_exposure.squeeze(), axis=1)
    
    relative_exposure = pd.concat([benchmark_exposure, relative_exposure])

    relative_exposure = relative_exposure[relative_exposure.columns[relative_exposure.iloc[0].argsort()[::-1]]]
    # relative_exposure = relative_exposure.applymap(lambda x: '{:.2%}'.format(x))

    return relative_exposure

def plot_exposure_over_time(df_weight, df_exposure, start_date=None, end_date=None, frequency=None, show_rebal=False, height=800, width=1200):
    df_weight = df_weight.loc[start_date:end_date]
    rebalance_dates = df_weight[df_weight['reb_flag']].index
    
    df_exposure_over_time = exposure_over_time(df_weight, df_exposure, start_date, end_date, frequency)

    fig = go.Figure()

    for column in df_exposure_over_time.columns:
        fig.add_trace(go.Scatter(x=df_exposure_over_time.index, y=df_exposure_over_time[column], mode='lines', stackgroup='one', name=column))

    if show_rebal==True:
        for date in rebalance_dates:
            fig.add_vline(type='line', x=date, line=dict(color='gray', width=1, dash='dash'), name='Rebalance')
            
    fig.update_layout(title='Exposure over time', xaxis_title='Date', yaxis=dict(title='Weights', tickformat='.2%'), legend=dict(x=1.1), height=height, width=width)

    fig.show()
