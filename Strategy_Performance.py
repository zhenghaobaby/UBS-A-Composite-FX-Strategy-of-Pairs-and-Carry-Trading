# Author : zhenghaobaby
# Time : 2019/10/12 20:07
# File : try.py
# Ide : PyCharm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

"""pair_trading function"""
def trade(S1, S2, indexS1, indexS2, window1, window2):
    MAX_HOLD = 5
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1, center=False).mean()
    ma2 = ratios.rolling(window=window2, center=False).mean()
    std = ratios.rolling(window=window2, center=False).std()
    zscore = (ma1 - ma2) / std
    money = [0] * len(ratios)
    value = [0] * len(ratios)
    countS1 = 0
    countS2 = 0
    sig = [[0] * 10 for _ in range(len(ratios))]  # log the operation signal.
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            if countS1 > -MAX_HOLD:
                countS1 -= 1
                countS2 += ratios[i]
                sig[i][indexS1] = -1
                sig[i][indexS2] = ratios[i]
                money[i]-=S1[i]*TRANS_FEE*1+abs(ratios[i])*S2[i]*TRANS_FEE  # long 1 short 1, 2 transaction fees
        # Buy long if the z-score is < 1
        elif zscore[i] < -1:
            if countS1 < MAX_HOLD:
                countS1 += 1
                countS2 -= ratios[i]
                sig[i][indexS1] = 1
                sig[i][indexS2] = -ratios[i]
                money[i]-=S1[i]*TRANS_FEE*1+abs(ratios[i])*S2[i]*TRANS_FEE # long 1 short 1, 2 transaction fees
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            money[i] += countS1 * S1[i] + countS2 * S2[i]
            money[i] -= abs(countS1 * S1[i]) * TRANS_FEE + abs(countS2 * S2[i]) * TRANS_FEE
            sig[i][indexS1] = -countS1
            sig[i][indexS2] = -countS2
            countS1 = 0
            countS2 = 0
        # when it comes to the end, clear the position.
        if i==len(S1)-1:
            money[i] += countS1 * S1[i] + countS2 * S2[i]
            money[i]-= abs(countS1 * S1[i])*TRANS_FEE+abs(countS2 * S2[i])*TRANS_FEE
            sig[i][indexS1] = -countS1
            sig[i][indexS2] = -countS2
            countS1 = 0
            countS2 = 0

        value[i] = countS1 * S1[i] + countS2 * S2[i]

    money = np.array(money)
    money = np.cumsum(money) # money we earn
    value = np.array(value)
    value = money + value
    value = pd.Series(value, index=ratios.index)
    sig = pd.DataFrame(sig,index = S1.index,columns=namelist)

    return value, sig


"""performance"""
Data_sheet=pd.read_csv("fair_price.csv",index_col=0)
base = Data_sheet['GBP']
N = len(base)
TRANS_FEE=0.0002 # transaction fee

plt.rcParams['figure.figsize'] = (12.8, 9.6)   #define the size of graphs
pairs=[('EUR', 'GBP'), ('GBP', 'JPY'), ('GBP', 'DKK'), ('JPY', 'AUD'), ('JPY', 'NZD'),
       ('JPY', 'CAD'), ('JPY', 'CHF'), ('JPY', 'NOK'), ('JPY', 'SEK'), ('JPY', 'DKK'),
       ('AUD', 'SEK'), ('CAD', 'CHF'), ('CAD', 'NOK'), ('CAD', 'SEK'), ('CHF', 'NOK'), ('CHF', 'SEK'), ('CHF', 'DKK')]

pnl_series=[]
sharpe_series={}
pairs_pnl=[0]*N
namelist = ['EUR','GBP','JPY','AUD','NZD','CAD','CHF','NOK','SEK','DKK']
signal = pd.DataFrame([[0]*10 for _ in range(N)],index=base.index,columns=namelist)

"""polt the whole pairs"""
for i in range(len(pairs)):
    indexS1 = namelist.index(pairs[i][0])
    indexS2 = namelist.index(pairs[i][1])
    pnl,temp = trade(Data_sheet[pairs[i][0]].iloc[:N], Data_sheet[pairs[i][1]].iloc[:N],indexS1,indexS2,5, 60)
    signal += temp
    pairs_pnl=list(map(lambda x,y:x+y,pnl,pairs_pnl))
    pnl.plot(label=pairs[i])
    # plt.plot(pnl,label=pairs[i])
    return_pnl = pnl.diff(1)/pnl
    sharpe_ratio = return_pnl.mean()/return_pnl.std()*np.sqrt(252)
    pnl_series.append(return_pnl)
    sharpe_series[pairs[i]]=sharpe_ratio


"""plot the total pairs strategy"""
pairs_pnl = pd.Series(pairs_pnl,index=base.index)
pairs_pnl.plot(label='pairs')


"""get the t_strategy signal"""
t_strategy = pd.read_csv("t-quantile.csv",index_col=0)
t_strategy_value_signal = t_strategy.cumsum()

t_strategy_money = t_strategy*-Data_sheet
t_strategy_money = t_strategy_money.sum(axis=1)-abs(t_strategy*Data_sheet*TRANS_FEE).sum(axis=1)
t_strategy_money = t_strategy_money.cumsum()

t_strategy_valule = t_strategy_value_signal*Data_sheet
t_strategy_valule = t_strategy_valule.sum(axis=1)

t_strategy_pnl = t_strategy_money+t_strategy_valule
t_strategy_pnl.plot(label='t-strategy')


"""combine two strategy """
money_signal = signal
value_signal = money_signal.cumsum()

combine_money_signal = money_signal+t_strategy #money is when you close the position what you earn from that
combine_value_signal = value_signal+t_strategy.cumsum() #value is that you hold a position, if the price change, the value of your protofilio will change

money_pnl = combine_money_signal*-Data_sheet
money_pnl = money_pnl.sum(axis=1)-abs(combine_money_signal*Data_sheet*TRANS_FEE).sum(axis=1)
money_pnl = money_pnl.cumsum()

value_pnl = combine_value_signal*Data_sheet
value_pnl = value_pnl.sum(axis=1)

Combined_pnl = money_pnl+value_pnl
Combined_pnl.plot(label = 'combined',linewidth='2')


"""sharpe ratio"""
print ("Transaction Fee: %.4f" %TRANS_FEE)
print ("Pairs return: %.4f, Pairs sharpe ratio: %.4f" %(pairs_pnl[-1],np.mean(pairs_pnl)/np.std(pairs_pnl)))
print ("t-strategy return: %.4f, t-strategy sharpe ratio: %.4f" %(t_strategy_pnl[-1],np.mean(t_strategy_pnl)/np.std(t_strategy_pnl)))
print ("Combined return: %.4f, Combined sharpe ratio: %.4f" %(Combined_pnl[-1],np.mean(Combined_pnl)/np.std(Combined_pnl)))

plt.legend()
plt.show()

"""output position,pnl"""
strategy_net_val = pd.concat([pairs_pnl,t_strategy_pnl,Combined_pnl],axis=1)
strategy_net_val.columns = ['pair_trading','t_strategy','combined']
strategy_net_val.to_csv("strategy_net_value.csv")
combine_value_signal.to_csv("position_daily.csv")


