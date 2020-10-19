# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:58:25 2020

@author: 徐钦华
"""

import sys
sys.path.append(r'D:\model_train')  
from copy import deepcopy  
from log import logger 
import bs_meathod
import ms_br_ys
import time
import joblib
import pandas as pd
import pickle
import inspect, re
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from scipy.stats import distributions
from sqlalchemy import create_engine
import toad
from toad.metrics import KS,AUC,PSI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold,ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, partial,space_eval
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance
import warnings
warnings.filterwarnings('ignore')
def model_evaluation(df_list,df_list_title,model,features):
    lst=[]
    for i,j in zip(df_list,df_list_title):
        i['prob']=model.predict_proba(i[features])[:,1]
        ks=KS(i['prob'],i['y'])
        total = i['y'].count()
        bad_rate = i['y'].sum()/total
        auc = AUC(i['prob'],i['y'])
        data={'样本类别':j,'样本数':total,'坏样本占比':bad_rate,'AUC':auc,'KS':ks}
        lst.append(data)
    data=pd.DataFrame(lst)
    return data
    
def month_ks(df,month):
    ks_data=[]
    for i,j in df.groupby(month):
        if len(j.y.unique())>1:
            full_AUC = AUC(j['prob'],j['y'])
            full_ks = KS(j['prob'],j['y'])
            ks_data.append([i,len(j),full_AUC,full_ks])
            df = pd.DataFrame(ks_data,columns=['月份','样本数','AUC','KS'])
    return df
    
class ks_plot:
    def __init__(self):
        pass
 
    def ComuTF(self,lst1,lst2):
        #计算TPR和FPR	
        #lst1为真实值,lst2为预测值
        TP = sum([1 if a==b==1 else 0 for a,b in zip(lst1,lst2)])#正例被预测为正例
        FN = sum([1 if a==1 and b==0 else 0 for a,b in zip(lst1,lst2)])#正例被预测为反例
        TPR = TP/(TP+FN) 
        TN = sum([1 if a==b==0 else 0 for a,b in zip(lst1,lst2)])#反例被预测为反例
        FP = sum([1 if a==0 and b==1 else 0 for a,b in zip(lst1,lst2)])#反例被预测为正例
        FPR = FP/(TN+FP)
        return TPR - FPR
 
    def get_ks(self,df,real_data,data):
        #real_data为真实值，data为原数据
   
        d = []
        for i in data:
            pre_data = [1 if line >=i else 0 for line in data]
            d.append(self.ComuTF(real_data,pre_data))
        return max(d)
    
    def GetKS(self,y_test,y_pred_prob):
        '''
        功能: 计算KS值，输出对应分割点和累计分布函数曲线图
        输入值:
        y_pred_prob: 一维数组或series，代表模型得分（一般为预测正类的概率）
        y_test: 真实值，一维数组或series，代表真实的标签（{0,1}或{-1,1}）
        '''
        fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
        ks_value = max(tpr-fpr)
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        sns.set(style="darkgrid", palette="muted", color_codes=True,context='paper')
        sns.set_style({'font.sans-serif':['simhei','Arial']})
#         #画ROC曲线
#         plt.plot([0,1],[0,1],'k--')
#         plt.plot(fpr,tpr)
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.show()
        #画ks曲线
        plt.plot(fpr, label='bad')
        plt.plot(tpr, label='good')
        plt.plot(abs(tpr-fpr), label='diff')
        x = np.argwhere(abs(tpr-fpr) == ks_value)[0, 0]
        plt.plot((x, x), (0, ks_value), label='ks - {:.2f}'.format(ks_value), color='r', marker='o', markerfacecolor='r', markersize=5)
        plt.scatter((x, x), (0, ks_value), color='r')
        plt.legend()
        plt.show()

def ks_chart(df, y_true, y_pre, num=10, good=0, bad=1):
    # 1.将数据从小到大平均分成num组
    df_ks = df.sort_values(y_pre).reset_index(drop=True)
    df_ks['rank'] = np.floor((df_ks.index / len(df_ks) * num) + 1)
    df_ks['set_1'] = 1
    # 2.统计结果
    result_ks = pd.DataFrame()
    result_ks['group_sum'] = df_ks.groupby('rank')['set_1'].sum()
    result_ks['group_min'] = df_ks.groupby('rank')[y_pre].min()
    result_ks['group_max'] = df_ks.groupby('rank')[y_pre].max()
    result_ks['group_mean'] = df_ks.groupby('rank')[y_pre].mean()
    # 3.最后一行添加total汇总数据
    result_ks.loc['total', 'group_sum'] = df_ks['set_1'].sum()
    result_ks.loc['total', 'group_min'] = df_ks[y_pre].min()
    result_ks.loc['total', 'group_max'] = df_ks[y_pre].max()
    result_ks.loc['total', 'group_mean'] = df_ks[y_pre].mean()
    # 4.好用户统计
    result_ks['good_sum'] = df_ks[df_ks[y_true] == good].groupby('rank')['set_1'].sum()
    result_ks.good_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'good_sum'] = result_ks['good_sum'].sum()
    result_ks['good_percent'] = result_ks['good_sum'] / result_ks.loc['total', 'good_sum']
    result_ks['good_percent_cum'] = result_ks['good_sum'].cumsum() / result_ks.loc['total', 'good_sum']
    # 5.坏用户统计
    result_ks['bad_sum'] = df_ks[df_ks[y_true] == bad].groupby('rank')['set_1'].sum()
    result_ks.bad_sum.replace(np.nan, 0, inplace=True)
    result_ks.loc['total', 'bad_sum'] = result_ks['bad_sum'].sum()
    result_ks['bad_percent'] = result_ks['bad_sum'] / result_ks.loc['total', 'bad_sum']
    result_ks['bad_percent_cum'] = result_ks['bad_sum'].cumsum() / result_ks.loc['total', 'bad_sum']
    # 6.计算ks值
    result_ks['diff'] = abs(result_ks['bad_percent_cum'] - result_ks['good_percent_cum'])
    # 7.更新最后一行total的数据
    result_ks.loc['total', 'bad_percent_cum'] = np.nan
    result_ks.loc['total', 'good_percent_cum'] = np.nan
    result_ks.loc['total', 'diff'] = result_ks['diff'].max()

    result_ks = result_ks.reset_index()

    return result_ks
def Rank_qcut(vector, K):
    '''
    解决qcut()为了删除重复值设置duplicates=‘drop'，则易出现于分片个数少于指定个数的问题
    return:分箱label列
    '''
    quantile = np.array([float(i) / K for i in range(K + 1)]) # Quantile: K+1 values
    return vector.rank(pct=True).apply(lambda x: (quantile >= x).argmax())

def ks_lift_chart(y, prob, title, **kwargs):
    '''
    模型分箱表现
    训练集默认分箱bins
    测试及oot传入训练集切分的bins
    '''
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    data = pd.DataFrame(data={'y': y.values, 'prob': prob})
    data.sort_values('prob', ascending=False, inplace=True)     
    if 'bins' not in kwargs:
        bins = 10
        data['prob_n']= Rank_qcut(prob,10)
    else:
        bins = kwargs['bins']
        data['prob_n'] = pd.cut(prob,bins,right=False)
    groupby_bin=data.groupby('prob_n',as_index=True)
    min_max_bin = pd.DataFrame()
    min_max_bin['min_score']=groupby_bin['prob'].min()
    min_max_bin['max_score']=groupby_bin['prob'].max()
    if 'bins' not in kwargs:
        min_max_bin.iloc[0,0]=float('-inf')
        min_max_bin.iloc[-1,1]=float('inf')
        bins = min_max_bin['min_score'].tolist()
        bins[0]=float('-inf')
        bins.append(float('inf'))
    data['y'] = data['y'].map({1: 'bad', 0: 'good'})
    cross_freq = pd.crosstab(index=data['prob_n'], columns=data['y'], values=data['prob'], aggfunc='count')
    cross_freq.sort_index(ascending=False, inplace=True)
    cross_freq['total']=cross_freq['good']+cross_freq['bad']
    cross_freq['bad_rate']=cross_freq['bad']/cross_freq['total']
    cross_freq['%bad']=cross_freq['bad']/cross_freq['bad'].sum()
    cross_freq['%total']=cross_freq['total']/cross_freq['total'].sum()
    cross_freq['cum_good_rate']=cross_freq['good'].cumsum()/cross_freq['good'].sum()
    cross_freq['cum_bad_rate']=cross_freq['bad'].cumsum()/cross_freq['bad'].sum()
    cross_freq['lift']=cross_freq['%bad']/cross_freq['%total']
    cross_freq['ks']=abs(cross_freq['cum_bad_rate']-cross_freq['cum_good_rate'])
    cross_freq.drop(['cum_good_rate','good','%bad','%total','cum_bad_rate'],axis=1,inplace=True)
    cross_freq = pd.merge(min_max_bin,cross_freq,left_index=True,right_index=True)
    plt.plot(cross_freq['bad'].cumsum()/cross_freq['bad'].sum(),cross_freq['lift'],'bo-', linewidth=2)
    plt.axhline(1, color='gray', linestyle='--')
    plt.ylim([0.0,2.5])
    plt.title(title, fontsize=15)
    if 'bins' not in kwargs:
        return cross_freq,bins
    else:
        return cross_freq



def var_lift_plot(y, var, title, bins=5, return_detail=False):
    '''
    入模前单变量的预筛选
    
    变量lift -分箱内坏样本占总坏样本的比例/分箱内样本数占样本的比例
    '''
    
    data = pd.DataFrame(data={'y': y.values, 'var': var})
    data.sort_values('var', ascending=False, inplace=True)
    c = toad.transform.Combiner()
    c.fit(data,y='y',method='step',n_bins=bins,empty_separate=True)
    data = c.transform(data, labels=False)
#    data2 = c.transform(data, labels=True)
#    data2 = data2.rename(columns={'var':'var_n','y':'y_n'})
#    data=pd.concat([data1,data2],axis=1)
    data['y'] = data['y'].map({1: 'bad', 0: 'good'})
    cross_freq = pd.crosstab(index=data['var'], columns=data['y'], values=data['var'], aggfunc='count')
    cross_freq.sort_index(ascending=False, inplace=True)
    cross_freq['total']=cross_freq['good']+cross_freq['bad']
    cross_freq['bad_rate']=cross_freq['bad']/cross_freq['total']
    cross_freq['%bad']=cross_freq['bad']/cross_freq['bad'].sum()
    cross_freq['%total']=cross_freq['total']/cross_freq['total'].sum()
    cross_freq['lift']=cross_freq['%bad']/cross_freq['%total']
    plt.plot(cross_freq.index,cross_freq['lift'],'--o', linewidth=2)
    plt.axhline(1, color='gray', linestyle='--')
    plt.ylim([0,2.5])
    plt.title(title, fontsize=15)
    plt.show()
    if return_detail:
        return cross_freq
    else:
        pass

def var_avg_plot(df_list,var_list,q=10):
    '''
    df_list:按train,test,oot顺序传入
    var_list:需要查看的变量
    观察入模后变量在模型中的稳定性和单调性
    变量在模型分区间的均值表现
    '''
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    df,bins = pd.qcut(df_list[0]['prob'],q=q,retbins=True,duplicates='drop')
    bins[0]=float('-inf')
    bins[-1]=float('inf')
    for i in tqdm(var_list):
        df_lst=[]    
        for k,j in enumerate(df_list):            
            data = pd.DataFrame(data={'prob': j['prob'].values, 'var': j[i]})
            data.sort_values('var', ascending=False, inplace=True)
            data['prob_n'] = pd.cut(data['prob'], bins)
            cross_count = pd.crosstab(index=data['prob_n'], columns='count',values=data['var'], aggfunc='count')
            cross_avg = pd.crosstab(index=data['prob_n'], columns='mean', values=data['var'], aggfunc='mean')
            cross_table = pd.merge(cross_count,cross_avg,on='prob_n')
            
            cross_table['{}'.format('train' if k==0  else('test' if k==1 else 'oot' ))]=cross_table['mean']
            cross_table.drop(['count','mean'],axis=1,inplace=True)
            cross_table.sort_index(ascending=False, inplace=True)
            df_lst.append(cross_table)
        data=pd.concat(df_lst,axis=1)
        data.plot(style='--o',alpha=0.8)
        plt.xlabel('prob_cut')
        plt.ylabel('var_mean')
        plt.legend(loc=1)
        plt.title('变量{}的轶'.format(i), fontsize=15)
        
def var_cut_plot(df_list,var_list,q=10):
    '''
    df_list:按train,test,oot顺序传入
    var_list:需要查看的变量
    观察入模重要变量在训练集、测试集的分布情况
    分布差异大导致模型拟合程度差
    '''
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    for i in tqdm(var_list):
        df_lst=[]    
        df,bins = pd.qcut(df_list[0][i],q=q,retbins=True,duplicates='drop')
        bins[0]=float('-inf')
        bins[-1]=float('inf')
        for k,j in enumerate(df_list):
            j['var_cut'] = pd.cut(j[i],bins)
            cross_count = pd.crosstab(index=j['var_cut'], columns='count',values=j[i], aggfunc='count') 
            cross_count['count'] = cross_count['count']/cross_count['count'].sum()
            cross_count['{}'.format('train' if k==0  else('test' if k==1 else 'oot' ))]=cross_count['count']
            cross_count.drop(['count'],axis=1,inplace=True)
            cross_count.sort_index(ascending=False, inplace=True)
            df_lst.append(cross_count)
        data=pd.concat(df_lst,axis=1)
        data.plot(style='--o',alpha=0.8)
#        fig.set_xticklabels(data.index)
        plt.xlabel('var_cut')
        plt.ylabel('count_percent')
        plt.legend(loc=1)
        plt.title('变量{}的分布'.format(i), fontsize=15)
def var_psi_chart(train_data,test_data,var_list,month,q=10):
    '''
    以训练集为预期分布,跨时间窗监控变量稳定性
    var_list:需要查看的变量
    '''
    data_lst=[]
    for i in tqdm(var_list):
        df_lst=[]
        for j,k in test_data.groupby(month):
            if len(k)<50:
                pass
            else:
                df = pd.DataFrame()
                df['var']=[i]
                df.set_index('var',inplace=True)
                df[j]=PSI(k[i],train_data[i])
                df_lst.append(df)
        data = pd.concat(df_lst,axis=1)
        data_lst.append(data)
    data_all = pd.concat(data_lst,axis=0)
    return data_all


def psi_self(expect_score, acture_score, length=10, method='step', return_detail=False):
    import math
    labels = ['c' + str(i) for i in range(length)]
    if method == 'step':
        expt_out, bins = pd.cut(expect_score, retbins=True, labels=labels, bins=length)
    else:
        expt_out, bins = pd.qcut(expect_score, retbins=True, labels=labels, q=length)
    bins[0] = min(acture_score)-1
    bins[length] = max(acture_score)+1
    acture_out = pd.cut(acture_score, bins=bins, labels=labels)
    re = pd.DataFrame({'expect': expt_out.value_counts(), 'acture': acture_out.value_counts()})
    re['expect%'] = re['expect'] / re['expect'].sum()
    re['acture%'] = re['acture'] / re['acture'].sum()
    re['expect%'] = re['expect%'].map(lambda x: 0.0001 if x == 0 else x)
    re['acture%'] = re['acture%'].map(lambda x: 0.0001 if x == 0 else x)

    re['psi'] = re.apply(lambda x: ((x['acture%'] - x['expect%']) * (math.log((x['acture%'] / x['expect%'])))),
                       axis=1)
    psi = re.psi.sum()
    if return_detail:
        return re, psi
    else:
        return psi
    
    
def model_cv(model, X, y, cv_folds=5, early_stopping_rounds=50, seed=11):
    xgb_param = model.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=1000, nfold=cv_folds,
                    metrics='auc',early_stopping_rounds=50, seed=seed, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(early_stopping_rounds)
       ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)
    return num_round_best

def randomsearchcv(model,train_data,features):
    k_fold=StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
    ratio = np.sum(train_data.y == 0) / float(np.sum(train_data.y == 1))
    dtrain = xgb.DMatrix(train_data[features],train_data.y)
    cv_result = xgb.cv(model.get_xgb_params(),
               dtrain,                   
               num_boost_round=1000,
               nfold=5,
               metrics='auc',
               early_stopping_rounds=50,
               callbacks=[xgb.callback.early_stop(50),
                          xgb.callback.print_evaluation(period=1, show_stdv=True)])
    print('best number of trees={}'.format(cv_result.shape[0]))
    model.set_params(n_estimators=cv_result.shape[0]) 
    param_grid = {
        'max_depth': [ 3, 4, 5, 6, 7,8,9],
        'min_child_weight': [10 * (i + 1) for i in range(30)],
        'gamma': [i for i in range(1, 30, 2)],
        'subsample': [i / 10.0 for i in range(5, 11)],
        'colsample_bytree': [i / 10.0 for i in range(5, 11)],
        'reg_lambda': [i for i in range(1, 30, 2)],
        'reg_alpha': [i for i in range(1, 30, 2)]
        }
    grid_search =RandomizedSearchCV(
            xgb.XGBClassifier(n_jobs=1,objective ='binary:logistic',eval_metric = 'auc',
                              learning_rate=0.1,n_estimators=cv_result.shape[0],
                              booster='gbtree',scale_pos_weight=float(ratio),seed=11),
            param_grid,
            scoring='roc_auc',
            n_iter=300,
            random_state=10,
            cv=k_fold)
    grid_search.fit(train_data[features],train_data.y)
    grid_search.estimator.set_params(**grid_search.best_params_)
    return grid_search.estimator

def bayessearchcv(model,train_data,features):
    ss_fold=ShuffleSplit(n_splits=5, random_state=11, test_size=0.25)
    ratio = np.sum(train_data.y == 0) / float(np.sum(train_data.y == 1))

    num_round = model_cv(model,train_data[features],train_data['y'])
    model.set_params(n_estimators=num_round)
    param_grid = {
        'max_depth': (3, 7),
        'min_child_weight': (1,300),
        'gamma': (1,30),
        'subsample': (0.4,1.0),
        'colsample_bytree':(0.4,1.0),
        'reg_lambda': (1,30),
        'reg_alpha':(1,30)
        }
    bayes_cv_tuner=BayesSearchCV(estimator=xgb.XGBClassifier(n_jobs=3,objective ='binary:logistic',eval_metric = 'auc',
                                                             n_estimators=num_round,scale_pos_weight=float(ratio),learning_rate=0.03,
                                                             booster='gbtree',seed=11),search_spaces=param_grid,scoring='roc_auc',
                                                            cv=ss_fold,n_iter=30,random_state=1)
    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""
        
        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
        
        # Get current parameters and the best parameters    
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
    bayes_cv_tuner.fit(train_data[features], train_data['y'],callback=status_print)
    best_parameters = bayes_cv_tuner.best_params_
    print(best_parameters)
    model.set_params(**best_parameters)
    return model
def get_bayes_param(model,train_data, features,bayes_params=None):
    ratio = np.sum(train_data.y == 0) / float(np.sum(train_data.y == 1))
    ss_fold=ShuffleSplit(n_splits=5, random_state=11, test_size=0.25)
    num_round=model_cv(model,train_data[features],train_data['y'])
    def rfc_cv( max_depth,
               min_child_weight, gamma, colsample_bytree,
               subsample, reg_lambda, reg_alpha):
        estimator = xgb.XGBClassifier(n_estimators=num_round, max_depth=int(max_depth),objective='binary:logistic',booster='gbtree',
                                      min_child_weight=int(min_child_weight), gamma=int(gamma),scale_pos_weight=float(ratio),
                                      learning_rate=0.03, colsample_bytree=colsample_bytree,seed=11,
                                      subsample=subsample, reg_lambda=int(reg_lambda), reg_alpha=int(reg_alpha))
        cval = cross_val_score(estimator, train_data[features], train_data.y, scoring='roc_auc', cv=ss_fold)
        return cval.mean()

    def optimize_rfc(params):
        optimizer = BayesianOptimization(f=rfc_cv, pbounds=params, random_state=11)
        optimizer.maximize(n_iter=20)
        return optimizer
    if bayes_params is None:
        adj_params = {'max_depth': (3, 5),
                      'min_child_weight': (1, 300),
                      'gamma': (0, 20),
                      'colsample_bytree': (0.6, 1),
                      'subsample': (0.6, 1),
                      'reg_lambda': (1, 20),
                      'reg_alpha': (1, 20)
                      }
    else:
        adj_params = bayes_params
    optimizer = optimize_rfc(adj_params)

    score = optimizer.max['target']
    print('best_score: {}'.format(score))
    params = optimizer.max['params']
    for i in [ 'max_depth', 'gamma', 'min_child_weight', 'reg_lambda', 'reg_alpha']:
        params[i] = int(round(params[i]))
    params['n_estimators']=num_round
    return params
def hyperopt_searchcv(model,train_data,features):
    num_round=model_cv(model,train_data[features],train_data['y'])
    def XGB_CV(params):

#     x_train, x_predict, y_train, y_predict
        ratio = np.sum(train_data.y == 0) / float(np.sum(train_data.y == 1))
        ss_fold=ShuffleSplit(n_splits=5, random_state=11, test_size=0.25)
        _model = xgb.XGBClassifier(max_depth=int(params['max_depth']),objective='binary:logistic',booster='gbtree',
                              n_estimators=num_round,
                              learning_rate=model.get_params('learning_rate')['learning_rate'],
                                   colsample_bytree=params['colsample_bytree'],
                                   colsample_bylevel=1,
                                   subsample=params['subsample'],
                                   gamma=params['gamma'],
                                   min_child_weight=params['min_child_weight'],
                                   scale_pos_weight=float(ratio),
                                   reg_alpha=params['reg_alpha'],
                                   reg_lambda=params['reg_lambda'],
                                   seed=11)
        metric = cross_val_score(_model, train_data[features], train_data.y, cv=ss_fold, scoring="neg_mean_squared_error")
        return min(-metric)
    from numpy.random import RandomState
    import hyperopt
    trials_2 = hyperopt.Trials()
    params_space = {"max_depth": hp.quniform("max_depth", 3,6,1),
#         "n_estimators": hp.randint("n_estimators", 300),
#         'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
         'gamma': hp.randint('gamma', 30),
         "subsample": hp.uniform("subsample", 0.4,1),
         "min_child_weight": hp.randint("min_child_weight", 300),
         'colsample_bytree':hp.uniform('colsample_bytree', 0.6,1),
         'reg_lambda': hp.randint('reg_lambda', 30),
        'reg_alpha':hp.randint('reg_alpha', 30)
         }
    
    best = fmin(
        fn=XGB_CV,
        space=params_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials_2,
        rstate=RandomState(123))
    best_params=space_eval(params_space,best)
    best_params['n_estimators']=num_round
    best_params['max_depth']=int(best_params['max_depth'])
    return best_params
def get_xgboost_importances(model,return_df=True):
    from sklearn import preprocessing
    importance_dict = {}
    for import_type in ['weight', 'gain', 'cover']:
        importance_dict['xgBoost-'+import_type] = model.get_booster().get_score(importance_type=import_type)
    
    # MinMax scale all importances
    importance_df = pd.DataFrame(importance_dict).fillna(0)
    importance_df = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(importance_df),
        columns=importance_df.columns,
        index=importance_df.index)
    # Create mean column
    importance_df['mean'] = importance_df.mean(axis=1)
    importance_df.sort_values('mean',ascending=False,inplace=True)

    # Plot the feature importances
    logger.info('plot')
    importance_df.sort_values('mean').plot(kind='barh', figsize=(10, 20))
    if return_df:
        return importance_df

def get_plot_tree(model,num_trees=2):
    from xgboost import plot_tree
    from matplotlib.pylab import rcParams
    
    ##set up the parameters
    rcParams['figure.figsize'] = 20,10
    plot_tree(model,num_trees=num_trees)
    