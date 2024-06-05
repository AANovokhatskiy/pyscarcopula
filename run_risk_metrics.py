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


'''this file has created to run risk_metrics from console. Scripts writes result to risk_data folder.'''

'''example of start
linux: python3 run_risk_metrics.py
'''

if __name__ == "__main__":
    crypto_prices = pd.read_csv("data/crypto_prices.csv", index_col=0, sep = ';')
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD',  'DOGE-USD', 'LINK-USD', 'MATIC-USD', 'ALGO-USD']
    crypto_returns_pd = np.log(crypto_prices[tickers] / crypto_prices[tickers].shift(1))[1:505]
    crypto_returns = crypto_returns_pd.values

    count_instruments = len(tickers)
    copula = GumbelCopula(count_instruments)

    '''mle result'''
    #gamma = [0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999]
    gamma = [0.95]
    window_len = 250
    MC_iterations = [int(10**7)]
    M_iterations = 15
    optimize_portfolio = True
    if optimize_portfolio == True:
        optimize_portfolio_cut = 'opt'
    else:
        optimize_portfolio_cut = 'noopt'

    #############################################################################################
    latent_process_tr = 500
    method = 'scar-m-ou'
    #method = 'mle'

    marginals_method = 'hyperbolic'
    #marginals_method = 'normal'
    marginals_method_cut = ''

    if marginals_method == 'hyperbolic':
        marginals_method_cut = 'h'
    elif marginals_method == 'normal':
        marginals_method_cut = 'n'

    portfolio_weight = np.ones(count_instruments) / count_instruments
    try:
        print(method)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process started {method} {start_time} \n")

        result = risk_metrics(copula,
                            crypto_returns[0:505],
                            window_len,
                            gamma,
                            MC_iterations,
                            marginals_params_method = marginals_method,
                            latent_process_type = method,
                            latent_process_tr = latent_process_tr,
                            optimize_portfolio = optimize_portfolio,
                            portfolio_weight = portfolio_weight,
                            #seed = 1000,
                            M_iterations = M_iterations,
                            #pre_calc_latent_process_params = latent_process_params
                            )


        if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
            os.mkdir(os.path.join(os.getcwd(),'risk_data'))

        for i in gamma:
            for j in MC_iterations:
                pd_var = pd.DataFrame(data = -result[i][j]['var'], columns = ['var'], index = crypto_returns_pd.index).shift(1)
                pd_cvar = pd.DataFrame(data = -result[i][j]['cvar'], columns = ['cvar'], index = crypto_returns_pd.index).shift(1)
                if result[i][j]['weight'].ndim == 1:
                    pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = crypto_returns_pd.columns)
                else:
                    pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = crypto_returns_pd.columns, index = crypto_returns_pd.index).shift(1)

                pd_var.to_csv(f"risk_data/pd_var_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_cvar.to_csv(f"risk_data/pd_cvar_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_weight_data.to_csv(f"risk_data/pd_weight_data_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process ended {method} {end_time}\n")


    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open('error.log', 'a') as f:
            f.write(f"Error at {current_time}: {str(e)}\n")
            traceback.print_exc(file=f)


    #############################################################################################
    latent_process_tr = 20000
    method = 'scar-p-ou'
    #method = 'mle'

    marginals_method = 'hyperbolic'
    #marginals_method = 'normal'
    marginals_method_cut = ''

    if marginals_method == 'hyperbolic':
        marginals_method_cut = 'h'
    elif marginals_method == 'normal':
        marginals_method_cut = 'n'

    portfolio_weight = np.ones(count_instruments) / count_instruments
    try:
        print(method)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process started {method} {start_time} \n")

        result = risk_metrics(copula,
                            crypto_returns[0:505],
                            window_len,
                            gamma,
                            MC_iterations,
                            marginals_params_method = marginals_method,
                            latent_process_type = method,
                            latent_process_tr = latent_process_tr,
                            optimize_portfolio = optimize_portfolio,
                            portfolio_weight = portfolio_weight,
                            #seed = 1000,
                            M_iterations = M_iterations,
                            #pre_calc_latent_process_params = latent_process_params
                            )


        if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
            os.mkdir(os.path.join(os.getcwd(),'risk_data'))

        for i in gamma:
            for j in MC_iterations:
                pd_var = pd.DataFrame(data = -result[i][j]['var'], columns = ['var'], index = crypto_returns_pd.index).shift(1)
                pd_cvar = pd.DataFrame(data = -result[i][j]['cvar'], columns = ['cvar'], index = crypto_returns_pd.index).shift(1)
                if result[i][j]['weight'].ndim == 1:
                    pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = crypto_returns_pd.columns)
                else:
                    pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = crypto_returns_pd.columns, index = crypto_returns_pd.index).shift(1)

                pd_var.to_csv(f"risk_data/pd_var_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_cvar.to_csv(f"risk_data/pd_cvar_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_weight_data.to_csv(f"risk_data/pd_weight_data_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process ended {method} {end_time}\n")

    except Exception as e:
        with open('error.log', 'a') as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)

    #############################################################################################
    latent_process_tr = 20000
    method = 'scar-p-ld'
    #method = 'mle'

    marginals_method = 'hyperbolic'
    #marginals_method = 'normal'
    marginals_method_cut = ''

    if marginals_method == 'hyperbolic':
        marginals_method_cut = 'h'
    elif marginals_method == 'normal':
        marginals_method_cut = 'n'

    portfolio_weight = np.ones(count_instruments) / count_instruments

    try:
        print(method)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process started {method} {start_time} \n")

        result = risk_metrics(copula,
                            crypto_returns[0:505],
                            window_len,
                            gamma,
                            MC_iterations,
                            marginals_params_method = marginals_method,
                            latent_process_type = method,
                            latent_process_tr = latent_process_tr,
                            optimize_portfolio = optimize_portfolio,
                            portfolio_weight = portfolio_weight,
                            #seed = 1515,
                            M_iterations = M_iterations,
                            #pre_calc_latent_process_params = latent_process_params
                            )


        if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
            os.mkdir(os.path.join(os.getcwd(),'risk_data'))

        for i in gamma:
            for j in MC_iterations:
                pd_var = pd.DataFrame(data = -result[i][j]['var'], columns = ['var'], index = crypto_returns_pd.index).shift(1)
                pd_cvar = pd.DataFrame(data = -result[i][j]['cvar'], columns = ['cvar'], index = crypto_returns_pd.index).shift(1)
                if result[i][j]['weight'].ndim == 1:
                    pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = crypto_returns_pd.columns)
                else:
                    pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = crypto_returns_pd.columns, index = crypto_returns_pd.index).shift(1)

                pd_var.to_csv(f"risk_data/pd_var_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_cvar.to_csv(f"risk_data/pd_cvar_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_weight_data.to_csv(f"risk_data/pd_weight_data_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process ended {method} {end_time}\n")

    except Exception as e:
        with open('error.log', 'a') as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)

    #############################################################################################
    latent_process_tr = 200
    #method = 'scar-m-ou'
    method = 'mle'

    marginals_method = 'hyperbolic'
    #marginals_method = 'normal'
    marginals_method_cut = ''

    if marginals_method == 'hyperbolic':
        marginals_method_cut = 'h'
    elif marginals_method == 'normal':
        marginals_method_cut = 'n'

    portfolio_weight = np.ones(count_instruments) / count_instruments
    try:
        print(method)
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process started {method} {start_time} \n")

        result = risk_metrics(copula,
                            crypto_returns[0:505],
                            window_len,
                            gamma,
                            MC_iterations,
                            marginals_params_method = marginals_method,
                            latent_process_type = method,
                            latent_process_tr = latent_process_tr,
                            optimize_portfolio = optimize_portfolio,
                            portfolio_weight = portfolio_weight,
                            seed = 1000,
                            M_iterations = M_iterations,
                            #pre_calc_latent_process_params = latent_process_params
                            )


        if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
            os.mkdir(os.path.join(os.getcwd(),'risk_data'))

        for i in gamma:
            for j in MC_iterations:
                pd_var = pd.DataFrame(data = -result[i][j]['var'], columns = ['var'], index = crypto_returns_pd.index).shift(1)
                pd_cvar = pd.DataFrame(data = -result[i][j]['cvar'], columns = ['cvar'], index = crypto_returns_pd.index).shift(1)
                if result[i][j]['weight'].ndim == 1:
                    pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = crypto_returns_pd.columns)
                else:
                    pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = crypto_returns_pd.columns, index = crypto_returns_pd.index).shift(1)

                pd_var.to_csv(f"risk_data/pd_var_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_cvar.to_csv(f"risk_data/pd_cvar_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
                pd_weight_data.to_csv(f"risk_data/pd_weight_data_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')

        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('error.log', 'a') as f:
            f.write(f"Process ended {method} {end_time}\n")

    except Exception as e:
        with open('error.log', 'a') as f:
            f.write(f"Error: {str(e)}\n")
            traceback.print_exc(file=f)

    # #############################################################################################
    # latent_process_tr = 200
    # #method = 'scar-m-ou'
    # method = 'mle'
    #
    # #marginals_method = 'hyperbolic'
    # marginals_method = 'normal'
    # marginals_method_cut = ''
    #
    # if marginals_method == 'hyperbolic':
    #     marginals_method_cut = 'h'
    # elif marginals_method == 'normal':
    #     marginals_method_cut = 'n'
    #
    # portfolio_weight = np.ones(count_instruments) / count_instruments
    # try:
    #     print(method)
    #     start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     with open('error.log', 'a') as f:
    #         f.write(f"Process started {method} {start_time} \n")
    #
    #     result = risk_metrics(copula,
    #                         crypto_returns[0:505],
    #                         window_len,
    #                         gamma,
    #                         MC_iterations,
    #                         marginals_params_method = marginals_method,
    #                         latent_process_type = method,
    #                         latent_process_tr = latent_process_tr,
    #                         optimize_portfolio = False,
    #                         portfolio_weight = portfolio_weight,
    #                         seed = 1000,
    #                         M_iterations = M_iterations,
    #                         #pre_calc_latent_process_params = latent_process_params
    #                         )
    #
    #
    #     if not os.path.exists(os.path.join(os.getcwd(),'risk_data')):
    #         os.mkdir(os.path.join(os.getcwd(),'risk_data'))
    #
    #     for i in gamma:
    #         for j in MC_iterations:
    #             pd_var = pd.DataFrame(data = -result[i][j]['var'], columns = ['var'], index = crypto_returns_pd.index).shift(1)
    #             pd_cvar = pd.DataFrame(data = -result[i][j]['cvar'], columns = ['cvar'], index = crypto_returns_pd.index).shift(1)
    #             if result[i][j]['weight'].ndim == 1:
    #                 pd_weight_data = pd.DataFrame([result[i][j]['weight']], columns = crypto_returns_pd.columns)
    #             else:
    #                 pd_weight_data = pd.DataFrame(result[i][j]['weight'], columns = crypto_returns_pd.columns, index = crypto_returns_pd.index).shift(1)
    #
    #             pd_var.to_csv(f"risk_data/pd_var_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
    #             pd_cvar.to_csv(f"risk_data/pd_cvar_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
    #             pd_weight_data.to_csv(f"risk_data/pd_weight_data_{optimize_portfolio_cut}_{method}_{marginals_method_cut}_{i}_{j}.csv", sep = ';')
    #
    #     end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     with open('error.log', 'a') as f:
    #         f.write(f"Process ended {method} {end_time}\n")
    #
    # except Exception as e:
    #     with open('error.log', 'a') as f:
    #         f.write(f"Error: {str(e)}\n")
    #         traceback.print_exc(file=f)
