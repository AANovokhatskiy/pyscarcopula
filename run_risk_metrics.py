#from memory_profiler import profile
from pyscarcopula.src.Clayton.ClaytonCopula  import ClaytonCopula
from pyscarcopula.src.Gumbel.GumbelCopula  import GumbelCopula
from pyscarcopula.src.Frank.FrankCopula  import FrankCopula
from pyscarcopula.src.Joe.JoeCopula  import JoeCopula

from pyscarcopula.metrics.risk_metrics import risk_metrics

import pandas as pd
import numpy as np
import os
import sys

import datetime
import traceback

from time import localtime, strftime


'''this file has created to run risk_metrics from console. Scripts writes result to risk_data folder.'''

'''example of start
linux: python3 run_risk_metrics.py
'''

def do_risk_metrics_and_save(copula,
                             data,
                             window_len,
                             gamma,
                             MC_iterations,
                             marginals_params_method,
                             latent_process_type,
                             latent_process_tr,
                             optimize_portfolio,
                             portfolio_weight,
                             seed,
                             M_iterations,
                             pre_calc_latent_process_params
                             ):
    try:
        print(method)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process started {method} {start_time} \n")

        result = risk_metrics(copula,
                              data.values,
                              window_len,
                              gamma,
                              MC_iterations,
                              marginals_params_method = marginals_params_method,
                              latent_process_type = latent_process_type,
                              latent_process_tr = latent_process_tr,
                              optimize_portfolio = optimize_portfolio,
                              portfolio_weight = portfolio_weight,
                              seed = seed,
                              M_iterations = M_iterations,
                              pre_calc_latent_process_params = pre_calc_latent_process_params
                             )


        if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
            os.mkdir(os.path.join(os.getcwd(),'risk_data'))

        copula_name = copula.name.split(' ')[0]
        current_time = strftime("%Y-%m-%d_%H%M%S", localtime())

        for i in gamma:
            for j in MC_iterations:
                df = pd.DataFrame()
                df['var'] = -result[i][j]['var']
                df['cvar'] = -result[i][j]['cvar']
                df.index = data.index
                df = df.shift(1)
                if result[i][j]['weight'].ndim == 1:
                    pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = data.columns)
                else:
                    pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = data.columns, index = data.index).shift(1)

                df.to_csv(f"risk_data/{copula_name}_{method}_{marginals_method}_{latent_process_tr}_{i}_{j}_{current_time}.csv", sep = ';')
                pd_weight_data.to_csv(f"risk_data/weight_data_{copula_name}_{method}_{marginals_method}_{latent_process_tr}_{i}_{j}_{current_time}.csv", sep = ';')

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process ended {method} {end_time}\n")


    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open('error.log', 'a') as f:
            f.write(f"Error at {current_time}: {str(e)}\n")
            traceback.print_exc(file=f)


if __name__ == "__main__":
    # crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep = ';')
    # tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD',  'DOGE-USD', 'LINK-USD', 'MATIC-USD', 'ALGO-USD']
    # crypto_returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:505]
    # crypto_returns = crypto_returns_pd.values

    moex_data = pd.read_csv("data/moex_top.csv", index_col=0)
    tickers = ['AFLT', 'LSRG', 'GAZP', 'NLMK', 'ROSN', 'KMAZ', 'AFKS', 'BSPB', 'MGNT']
    moex_returns_pd = np.log(moex_data[tickers] / moex_data[tickers].shift(1))[1:505]#
    #moex_returns = moex_returns_pd.values

    count_instruments = len(tickers)
    copula = GumbelCopula(count_instruments)


    #############################################################################################
    #gamma = [0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999]
    gamma = [0.95]
    window_len = 250
    MC_iterations = [int(10**3)]

    latent_process_tr = 500
    M_iterations = 15

    optimize_portfolio = False
    portfolio_weight = np.ones(count_instruments) / count_instruments

    #method = 'scar-m-ou'
    #method = 'scar-p-ou'
    #method = 'scar-p-ld'
    method = 'mle'

    marginals_method = 'normal'
    #marginals_method = 'hyperbolic'
    #marginals_method = 'stable'

    seed = None
    pre_calc_latent_process_params = None

    do_risk_metrics_and_save(copula,
                             moex_returns_pd,
                             window_len,
                             gamma,
                             MC_iterations,
                             marginals_method,
                             method,
                             latent_process_tr,
                             optimize_portfolio,
                             portfolio_weight,
                             seed,
                             M_iterations,
                             pre_calc_latent_process_params
                             )



