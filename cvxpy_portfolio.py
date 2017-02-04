from cvxpy import *
import cvxpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahoo_finance import Share
np.set_printoptions(suppress=True)
    
def calculate_portfolio(cvxtype, returns_function, long_only, exp_return, 
                        selected_solver, max_pos_size, ticker_list):
    assert cvxtype in ['minimize_risk','maximize_return']
    # Variables:
    # mu is the vector of expected returns.
    # sigma is the covariance matrix.
    # gamma is a Parameter that trades off risk and return.
    # x is a vector of stock holdings as fractions of total assets.
    gamma = Parameter(sign="positive")
    gamma.value = 1
    returns, stocks, betas = returns_function
        
    cov_mat = returns.cov()
    Sigma = cov_mat.values # np.asarray(cov_mat.values) 
    w = Variable(len(cov_mat))  # #number of stocks for portfolio weights
    risk = quad_form(w, Sigma)  #expected_variance => w.T*C*w =  quad_form(w, C)
    num_stocks = len(cov_mat)
    
    if cvxtype == 'minimize_risk': # Minimize portfolio risk / portfolio variance
        if long_only == True:
            prob = Problem(Minimize(risk), [sum_entries(w) == 1, w > 0 ])  # Long only
        else:
            prob = Problem(Minimize(risk), [sum_entries(w) == 1]) # Long / short 
    
    elif cvxtype == 'maximize_return': # Maximize portfolio return given required level of risk
        #mu  #Expected return for each instrument
        #expected_return = mu*x
        #risk = quad_form(x, sigma)
        #objective = Maximize(expected_return - gamma*risk)
        #p = Problem(objective, [sum_entries(x) == 1])
        #result = p.solve()

        mu = np.array([exp_return]*len(cov_mat)) # mu is the vector of expected returns.
        expected_return = np.reshape(mu,(-1,1)).T * w  # w is a vector of stock holdings as fractions of total assets.   
        objective = Maximize(expected_return - gamma*risk) # Maximize(expected_return - expected_variance)
        if long_only == True:
            constraints = [sum_entries(w) == 1, w > 0]
        else: 
            #constraints=[sum_entries(w) == 1,w <= max_pos_size, w >= -max_pos_size]
            constraints=[sum_entries(w) == 1]
        prob = Problem(objective, constraints)

    prob.solve(solver=selected_solver)
    
    weights = []
    for weight in w.value:
        weights.append(float(weight[0]))
        
    if cvxtype == 'maximize_return':
        optimal_weights = {"Optimal expected return":expected_return.value,
                          "Optimal portfolio weights":np.round(weights,2),
                          "tickers": ticker_list,
                          "Optimal risk": risk.value*100
                          }
        
    elif cvxtype == 'minimize_risk':
        optimal_weights = {"Optimal portfolio weights":np.round(weights,2),
                           "tickers": ticker_list,
                           "Optimal risk": risk.value*100
                          }   
    return optimal_weights


def getReturns(stocks = 'MSFT,AAPL,NFLX,JPM,UVXY,RSX,TBT', period_days = 100, end = '2016-12-09'):
    stocks = stocks.split(",")
    index = 'SPY'
    stocks.append(index)
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(period_days)).strftime("%Y-%m-%d")
        
    i = 0
    w= pd.DataFrame()
    t = []
    for s in stocks:
        z = Share(s)
        px = pd.DataFrame(z.get_historical(start,end))[['Close','Date']]
        px['Close']=px['Close'].astype(float)
        px.index = px['Date']
        del px['Date']
        px.columns = [s]
        t.append(px)
    w = pd.concat(t,axis=1, join='inner')
    w = w.sort_index().pct_change()  #returns => w.cov() covariance matrix of returns
    #calculate betas
    betas = []
    for s in stocks:
        if s != index:
            col = np.column_stack((w[s],w[index]))
            b = np.cov(col)/np.var(w[index])
            betas.append(b)  
    stocks.remove(index)
    del w[index]
    returns = w
    return returns,stocks,np.round(betas,4)


tickers = 'SHLD,AAPL,NFLX,RSX,TBT,EEM,UVXY,GME,AMZN,SVXY'
lookback_days = 300
last_day = "2017-02-02"
ret = getReturns(tickers,lookback_days,last_day)

#get portfolio weights
p = calculate_portfolio(cvxtype='maximize_return',
                        returns_function=ret,
                        long_only=False,
                        exp_return=0.20,
                        selected_solver='SCS',
                        max_pos_size=0.50,
                        ticker_list=tickers)

for key in p:
    print(p.get(key)) 
