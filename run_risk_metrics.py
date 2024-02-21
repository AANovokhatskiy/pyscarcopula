#from memory_profiler import profile
from pyscarcopula.src.Gumbel.GumbelCopula  import GumbelCopula
from pyscarcopula.metrics.risk_metrics import risk_metrics

import pandas as pd
import numpy as np
import os
import sys

'''this file has created to run risk_metrics from console. Scripts writes result to risk_data folder.'''

'''example of start
linux: python3 run_risk_metrics.py
'''

if __name__ == "__main__":
    '''Read data and transform returns to pseudo observations'''
    returns_data1 = pd.read_csv(r"data/test_returns_data_for_eis.csv", index_col=0) # Data from MOEX for tickers: AFKS, ROSN, SBER, YNDX. 
                                                                                    # Data starts from 01.01.2020 with 30-minutes interval  
    returns_data2 = pd.read_csv(r"data/dj.csv", sep=';') # Daily data of indexies DowJones и NASDAQ. 
                                                        # Data starts from 01.01.1990

    moex_data = pd.read_csv("data/moex_top.csv", index_col=0)
    tickers = ['AFLT', 'LSRG', 'GAZP', 'NLMK', 'ROSN', 'KMAZ', 'AFKS', 'BSPB', 'MGNT']
    moex_returns_pd = np.log(moex_data[tickers] / moex_data[tickers].shift(1))[1:601]
    moex_returns = moex_returns_pd.values

    count_instruments = len(tickers)
    copula = GumbelCopula(count_instruments)

    '''set params'''
    MC_iterations = [int(10**4), int(10**5), int(10**6)]
    latent_process_tr = 200
    gamma = [0.9, 0.95, 0.97, 0.99]
    window_len = 250
    method = 'MLE'
    cpus = 6

    result = risk_metrics(copula, moex_returns, window_len,
                                                    gamma, MC_iterations,
                                                    marginals_params_method = 'norm',
                                                    latent_process_type = method,
                                                    latent_process_tr = latent_process_tr,
                                                    optimize_portfolio = False,
                                                    portfolio_type = 'I',
                                                    portfolio_weight = None,
                                                    cpu_processes=cpus)
    
    if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
        os.mkdir(os.path.join(os.getcwd(),'risk_data'))
    for i in gamma:
        for j in MC_iterations:
            pd_var = pd.Series(data = -var_data[i][j]['var'], index = moex_returns_pd.index).shift(1)
            pd_cvar = pd.Series(data = -cvar_data[i][j]['cvar'], index = moex_returns_pd.index).shift(1)
            pd_weight_data = pd.DataFrame(weight_data[i][j]['weight'], columns = moex_returns_pd.columns, index = moex_returns_pd.index).shift(1)

            pd_var.to_csv(f"risk_data/pd_var_{copula_method}_{i}_{j}_.csv", sep = ';')
            pd_cvar.to_csv(f"risk_data/pd_cvar_{copula_method}_{i}_{j}.csv", sep = ';')
            pd_weight_data.to_csv(f"risk_data/pd_weight_data_{copula_method}_{i}_{j}.csv", sep = ';')
