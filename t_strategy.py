#fx trade
#Currency order: EUR, GBP, JPY, AUD, NZD, CAD, CHF, NOK, SEK, USD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import sub

class Portfolio:
    def __init__(self,length):
        self.long=[[10,0,0],[10,0,0]] #currency index, open price, quantity, 10 represents closed position
        self.short=[[10,0,0],[10,0,0]]
        self.long_history=np.zeros((length,10))    #the history of position
        self.short_history=np.zeros((length,10)) 
        self.value=0
        self.value_track=[0]
        self.returns=[]
        self.status="closed"
    def reset_longshort(self):
        self.long=[[10,0,0],[10,0,0]] 
        self.short=[[10,0,0],[10,0,0]]
        self.status="closed"

def get_rate(s,row_n):  #get excess rate
    return np.array([np.log(s[i].iloc[row_n,1]) for i in range(10)])

def gen_portfolio(s, f): #s:spot rate; f:one month forward rate
    p1,p2,p3,p4,p5=[],[],[],[],[]
    length=len(s[0])
    for d in range(length):
        spot_list=[]
        forward_list=[]
        for cur in range(10):
            spot_list.append(s[cur].iat[d,1])
            forward_list.append(f[cur].iat[d,1])
        forward_premium=list(map(sub,np.log(forward_list),np.log(spot_list)))
        forward_premium_sort=np.argsort(forward_premium)
        p5.append([forward_premium_sort[0],forward_premium_sort[1]])
        p4.append([forward_premium_sort[2],forward_premium_sort[3]])
        p3.append([forward_premium_sort[4],forward_premium_sort[5]])
        p2.append([forward_premium_sort[6],forward_premium_sort[7]])
        p1.append([forward_premium_sort[8],forward_premium_sort[9]])
        p=(p1,p2,p3,p4,p5)
    return p

def cal_mv(s,f,D=21):   #calculate mv
    mv=[]
    length=len(s[0])
    m=s[0].iat[0,0].month
    mv_n=0
    for d in range(2,length):
        if s[0].iat[d,0].month!=m:
            mv.append(mv_n)
            mv_n=0
        r=get_rate(s,d)-get_rate(f,d-1)
        r_lag=get_rate(s,d-1)-get_rate(f,d-2)
        mv_n+=sum(r**2+2*r*r_lag)/100
        m=s[0].iat[d,0].month
    return mv

def close_pos(port,d,s,f,p1,p5): #return value brought by closing position
    long_value=s[port.long[0][0]].iat[d,1]*port.long[0][2]+s[port.long[1][0]].iat[d,1]*port.long[1][2]
    short_value=s[port.short[0][0]].iat[d,1]*port.short[0][2]+s[port.short[1][0]].iat[d,1]*port.short[1][2]
    port.long_history[d,port.long[0][0]]-=port.long[0][2]
    port.long_history[d,port.long[1][0]]-=port.long[1][2]
    port.short_history[d,port.short[0][0]]-=port.short[0][2]
    port.short_history[d,port.short[1][0]]-=port.short[1][2]
    port.value=long_value+short_value
    port.value_track.append(port.value)
    if (port.value_track[-2])!=0:
        port.returns.append(port.value_track[-1]/port.value_track[-2]-1)
    port.reset_longshort()
    return port
    
def open_pos(port,d,s,f,p1,p5): #opening position
    port.long=[[p1[d][0],s[p1[d][0]].iat[d,1],1/4/s[p1[d][0]].iat[d,1]],[p1[d][1],s[p1[d][1]].iat[d,1],1/4/s[p1[d][1]].iat[d,1]]]
    port.short=[[p5[d][0],s[p5[d][0]].iat[d,1],-1/4/s[p5[d][0]].iat[d,1]],[p5[d][1],s[p5[d][1]].iat[d,1],-1/4/s[p5[d][1]].iat[d,1]]]
    port.long_history[d,p1[d][0]]+=1/4/s[p1[d][0]].iat[d,1]    #record position
    port.long_history[d,p1[d][1]]+=1/4/s[p1[d][1]].iat[d,1]
    port.short_history[d,p5[d][0]]-=1/4/s[p5[d][0]].iat[d,1]
    port.short_history[d,p5[d][1]]-=1/4/s[p5[d][1]].iat[d,1]
    port.status="open"
    return port
    
def t_strategy(t,s,f,mv,p): #inplement t strategy
    p1,p2,p3,p4,p5=p
    length=len(s[0])
    port=Portfolio(length)
    port=open_pos(port,0,s,f,p1,p5)    
    d=0
    m=0
    while m<11:
        d+=1
        if s[0].iat[d,0].month!=s[0].iat[d-1,0].month:
            m+=1
            if port.status=="open":
                port=close_pos(port,d,s,f,p1,p5)
                port=open_pos(port,d,s,f,p1,p5)
    while d<length-2:
        d+=1
        if s[0].iat[d,0].month!=s[0].iat[d-1,0].month:
            m+=1
            t_indicator_1=port.returns[-1]<np.percentile(port.returns,t)
            t_indicator_2=mv[m-1]>np.percentile(mv[:m-1],50)
            t_indicator=(t_indicator_1 and t_indicator_2)   #critical indicators (t quantile and mv), please refer to the summary paper
            if (t_indicator and (port.status=="open")):
                port=close_pos(port,d,s,f,p1,p5)
            elif ((not t_indicator) and (port.status=="open")):
                port=close_pos(port,d,s,f,p1,p5)
                port=open_pos(port,d,s,f,p1,p5)
            elif ((not t_indicator) and (port.status=="closed")):
                port.value_track.append(0)
                port=open_pos(port,d,s,f,p1,p5)
            elif (t_indicator and (port.status=="closed")):
                port.value_track.append(0)
    if port.status=="open":
        d+=1
        port=close_pos(port,d,s,f,p1,p5)
    return port

def import_pd(cur):     #import spot and forward rate
    s=[]
    f=[]
    imp=pd.read_csv("forward_rate.csv")
    for i in range(10):
        fn=imp[["DATE",cur[i]]].copy(deep=True)
        fn["DATE"]=pd.to_datetime(fn["DATE"])
        f.append(fn)
    imp=pd.read_csv("fair_price.csv")
    for i in range(10):
        sn=imp[["DATE",cur[i]]].copy(deep=True)
        sn["DATE"]=pd.to_datetime(sn["DATE"])
        s.append(sn)
    return s,f

def output_pos(port,s,cur):     #ouput position information
    history_sum=port.long_history+port.short_history
    history_sum=pd.DataFrame(history_sum,index=s[0]["DATE"],columns=cur)
    history_sum.to_csv("t-quantile.csv")   #unquote to output position information 
    return history_sum

def plot_pos(history_sum,s):    #plotting
    s_sum=s[0]
    for i in range(1,10):
        s_sum[s[i].columns.values.tolist()[1]]=s[i].iloc[:,1]
    s_sum=s_sum.set_index(["DATE"])
    total_cf=history_sum*-s_sum
    total_cf_allq=total_cf.sum(axis=1)
    total_cf_allq=total_cf_allq.cumsum()
    plt.plot(total_cf_allq.to_list(),label="monthly evaluated")
    return total_cf_allq
    
def cal_sharpe(history_sum,total_cf_allq,s):        #calculate sharpe ratio and plot daily values
    s_sum=s[0]
    for i in range(1,10):
        s_sum[s[i].columns.values.tolist()[1]]=s[i].iloc[:,1]
    s_sum=s_sum.set_index(["DATE"])
    history_cumsum=history_sum.cumsum()
    mark_to_market=history_cumsum*s_sum
    mark_to_market=mark_to_market.sum(axis=1)
    mark_to_market=mark_to_market+total_cf_allq
    plt.plot(mark_to_market.to_list(),label="daily evaluated")
    mark_to_market=mark_to_market+1
    mark_to_market=mark_to_market.pct_change()[1:]
    return mark_to_market.mean()/mark_to_market.std()*np.sqrt(252)

def main():
    cur=['EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'NOK', 'SEK', 'DKK']
    s,f=import_pd(cur)
    p=gen_portfolio(s, f)
    mv=cal_mv(s,f)
    t=50
    '''for iterating t
    for ti in range(5,6):
        t=10*ti #set threshold
        #for i in range(10):
        #    plt.plot(np.subtract(np.log(f[i].iloc[:,1]),np.log(s[i].iloc[:,1])))
        port=t_strategy(t,s,f,mv,p)
        #plt.plot(port.value_track[1:])
        pnl = np.array(port.value_track)
        pnl = pnl.cumsum()
        #plt.plot(pnl,label=t)
        #sharpe=np.mean(port.value_track)/np.std(port.value_track)*np.sqrt(12)
        #print("sharpe: " + str(t) + " " + str(sharpe))
    #plt.legend()
    #plt.show()
    '''
    port=t_strategy(t,s,f,mv,p)
    pnl = np.array(port.value_track)
    pnl = pnl.cumsum()
    history_sum=output_pos(port,s,cur)
    total_cf_allq=plot_pos(history_sum,s)
    print("sharpe ratio: "+str(cal_sharpe(history_sum,total_cf_allq,s)))
    plt.legend()
    plt.show()
    
main()