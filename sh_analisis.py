# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:30:50 2020

@author: 徐钦华
"""

import sys
sys.path.append(r'D:\model_train')  
import os
from copy import deepcopy 
import var_cut  
import bs_meathod
import model_method
import ms_br_ys
import time
import joblib
import pandas as pd
from sklearn_pandas import DataFrameMapper,gen_features
from sklearn.preprocessing import LabelBinarizer,StandardScaler,MinMaxScaler
import pickle
from log import logger
from copy import deepcopy
import numpy as np
from sqlalchemy import create_engine
import toad
from toad.metrics import KS,AUC,PSI
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings('ignore')
                

os.chdir(r'D:\数据分析项目\算话测试')
sh_dt_data = pd.read_csv('算话测试报告-恒普-20200826.part01//算话变量_多头借贷行为特征_8w.csv',engine='python',encoding='utf-8-sig')
label_data =pd.read_excel('恒普测试数据with_label.xlsx',sheet_name='y_label',encoding='utf-8')
data_2 = pd.read_csv('算话测试报告-恒普-20200826.part01//算话变量_高风险借贷行为_8w.csv',engine='python')
data_3 = pd.read_csv('算话测试报告-恒普-20200826.part01//算话变量_高风险借贷行为特征_8w.csv',engine='python')
var_dict =pd.read_excel('算话测试报告-恒普-20200826.part01//算话变量说明.xlsx',sheet_name='1.多头借贷行为特征')

def read_csv_gbk(df_name):
    with open(df_name, 'rb') as f:
        data = f.read().decode('utf-8', 'ignore').replace('?1', '').replace('?', '')
    f.close()
    with open(df_name.replace('.csv', '') + '_gbk.csv' , 'w') as f:
        f.write(data)
    f.close()
    df = pd.read_csv(df_name.replace('.csv', '') + '_gbk.csv', encoding='gbk' )
    return df
data_4 = read_csv_gbk('算话测试报告-恒普-20200826.part01//算话变量_个人资料置信度_8w.csv')
#data_4 = pd.read_csv(r'C:\Users\徐钦华\Desktop\数据分析项目\算话测试\算话测试报告-恒普-20200826.part01\算话变量_个人资料置信度_8w.csv',
#                     engine='python')
data_5 = pd.read_csv('算话测试报告-恒普-20200826.part01//算话变量_团伙风险识别_8w.csv',engine='python')
label_data=label_data[label_data['数据编号'].notna()]

mapper_2 = DataFrameMapper([('pdls041',LabelBinarizer())],default=None,df_out=True)
data_2=mapper_2.fit_transform(data_2.copy())
mapper_3=DataFrameMapper([('z_risk_rate',LabelBinarizer())],default=None,df_out=True)
data_3 = mapper_3.fit_transform(data_3.copy())
feature_def_4 = gen_features(columns=[[m] for m in ['xx'+str(i) for i in range(598,619)]+['xx1247']+['xx'+str(i) for i in range(2531,2538)]],classes=[MinMaxScaler,StandardScaler])
mapper_4=DataFrameMapper(feature_def_4,default=None,input_df=True,df_out=True)
data_4=np.round(mapper_4.fit_transform(data_4.copy()),2)
feature_def_5=gen_features(columns=[['xx1174'],['xx1175'],['xx1176'],['xx1183'],['xx1184'],['xx1185'],['xx631'],['xx632'],['xx633'],['xx2495'],
                                  ['xx2492'],['xx2538']],classes=[MinMaxScaler,StandardScaler])
mapper_5 = DataFrameMapper(feature_def_5,default=None,input_df=True,df_out=True)
data_5 = np.round(mapper_5.fit_transform(data_5.copy()),2)
for i in ['R01','R02','R07']:
    data_3[i]=0
    data_3[i][(data_3['z_risk_reason'].notna())&(data_3['z_risk_reason'].str.contains(i))] = 1
for i in['R03','R04','R05','R06']:
    data_3['R_risk']=0
    data_3['R_risk'][(data_3['z_risk_reason'].notna())&(data_3['z_risk_reason'].str.contains(i))] = 1
for j in ['S100','S200','S210','S220','S221','S222','S223','S224','S225']:
    data_3[j]=0
    data_3[j][(data_3['z_business_source'].notna())&(data_3['z_business_source'].str.contains(j))]=1

for i in [sh_dt_data,data_2,data_3,data_4,data_5]:
    drop_var = [m for m in i.columns.tolist() if i[m].dtypes=='object' and m!='app_num']
    i.drop(drop_var+['app_date'],axis=1,inplace=True)
data = pd.merge(sh_dt_data,data_2,on='app_num')
data = pd.merge(data,data_3,on='app_num')
data = pd.merge(data,data_4,on='app_num')
data = pd.merge(data,data_5,on='app_num')
data = pd.merge(data,label_data,left_on='app_num',right_on='数据编号',how='left')

br_data_ch = bs_meathod.mysql_data_out(['洋钱罐fk_derck','凑数','凑数'],['three', 'label_y'])
br_data_ch = bs_meathod.label_y_check(br_data_ch)
br_data_ch = bs_meathod.event_name_check(br_data_ch)


data_zhongbang = data[data['event_name']=='众邦开发3W']
data_ms = data[data['event_name']=='民生易贷']
data_xfxj = data[data['event_name']=='幸福消金1']
data_yqg = data[data['event_name']=='洋钱罐']
data_yqg = data_yqg.drop('y',axis=1)
data_yqg = pd.merge(data_yqg,br_data_ch,on=['phonenum_md5','id_card_md5','app_date'],how='left')
data_yqg=data_yqg[data_yqg['y'].notna()]

data_in=deepcopy(data_zhongbang)
object_var = [n for n in data_in.columns.tolist() if data_in[n].dtypes=='object']
data_in.drop(object_var,axis=1,inplace=True)

data_in=data_in[data_in['y']!=99]
data_in['app_date']=data_in['app_date'].map(lambda x:str(x)[:6])
features = data_in.columns.tolist()

select_features = toad.detect(data_in[features])
drop_col = select_features[select_features['unique']==1].index.tolist()
model_data = data_in[features].drop(drop_col,axis=1)
model_data.info()

model_data_2,drop_var = toad.select(model_data,model_data['y'], empty=0.8, iv=0.02, corr=0.8, return_drop=True,
                                     exclude=['y', 'app_date'])
iv_data = bs_meathod.mutil_feature_iv(model_data_2.drop('app_date',axis=1),'y',method='dt')
var_dict['变量名']=var_dict['变量名'].str.replace('XX','xx')
iv_data_full = pd.merge(iv_data,var_dict,left_on='col_name',right_on='变量名',how='left')
#iv_data_full.to_csv(r'D:\model_train\iv_var.csv',encoding='utf-8-sig')

#iv_var = toad.quality(model_data.drop('app_date',axis=1),model_data['y'],iv_only=True)
features = model_data_2.columns.tolist()
features.remove('app_date')
features.remove('y')
before_data = model_data[~model_data['app_date'].isin(['202001','202002','202003','202004'])]
oot_data = model_data[model_data['app_date'].isin(['202001','202002','202003','202004'])]
#before_data = model_data[(model_data['app_date'] < '202001') | (model_data['app_date'] > '202002')]
#last_data = model_data[(model_data['app_date'] >= '202001')&(model_data['app_date'] <= '202002')]

#选择最优seed,尽量使拆分数据集的特征分布一致
best_seed = bs_meathod.select_seed(before_data,features)
print(best_seed)

#拆分训练集、验证集和测试集

training, testing = bs_meathod.split_month_self(before_data,'y','app_date',frac=0.8,random_seed=25)
train, valid = bs_meathod.split_month_self(training,'y','app_date',frac=0.8)

#before_data.drop('app_date', axis=1, inplace=True)
#oot_data.drop('app_date', axis=1, inplace=True)
#model_data.drop('app_date', axis=1, inplace=True)

#x_train, y_train = train[features], train.y
#x_valid, y_valid = valid[features], valid.y

#dtrain = xgb.DMatrix(x_train,label=y_train)
ratio = np.sum(training.y == 0) / float(np.sum(training.y == 1))
#
## 模型调优
#k_fold=StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
model = xgb.XGBClassifier(max_depth=4,objective='binary:logistic',booster='gbtree',
                          n_estimators=73,
                          learning_rate=0.03,
                               colsample_bytree=0.8,
                               colsample_bylevel=1,
                               subsample=0.8,
                               gamma=10,
                               min_child_weight=10,
                               scale_pos_weight=float(ratio),
                               reg_alpha=10,
                               reg_lambda=10,
                               seed=11)
#params = model_method.get_bayes_param(model,training,features)
#model.set_params(**params)
model = model_method.bayessearchcv(model,training,features)
#params = model_method.hyperopt_searchcv(model,training,features)
#model.set_params(**params)
model = model.fit(training[features],training.y,eval_metric='auc',verbose=1)

df_evaluation = model_method.model_evaluation([training,testing,oot_data,data_ms,data_xfxj,data_yqg],['众邦-build',
                                              '众邦-test','众邦-oot','民生-cross_valid','幸福消金-cross_valid','洋钱罐-cross_valid'],model,features)

with open('sh_model.bat','wb') as f:
    pickle.dump(model,f)
training['prob']=model.predict_proba(training[features])[:,1]
testing['prob']=model.predict_proba(testing[features])[:,1]
oot_data['prob']=model.predict_proba(oot_data[features])[:,1]
build_cut,bins = model_method.ks_lift_chart(training['y'],training['prob'],'train')
test_cut=model_method.ks_lift_chart(testing['y'],testing['prob'],'testing',bins=bins)
oot_cut = model_method.ks_lift_chart(oot_data['y'],oot_data['prob'],'oot',bins=bins)

month_ks=model_method.month_ks(testing,'app_date')

feature_results = var_cut.get_feature_result(training[features+['y']],'y')

importance_df = model_method.get_xgboost_importances(model,return_df=True)
data_var = model_method.var_avg_plot([training,testing,oot_data],importance_df.index.tolist()[:10],q=10)
model_method.var_lift_plot(training['y'],training['xx2337'],'xx2337')

model_method.var_cut_plot([training,testing],importance_df.index.tolist()[:10],q=10)

var_psi_ie = model_method.var_psi_chart(training,testing,importance_df.index.tolist()[:10],'app_date')

#model_method.get_plot_tree(model)
#PSI(testing[testing['y']==1]['prob'],training[training['y']==1]['prob'])
#for i in importance_df.index.tolist()[:10]:
#    print(PSI(testing[i],training[i]))

import seaborn as sns
sns.set_style('whitegrid')
sns.stripplot(x='app_date',y='xx2392',hue='y',data=training,jitter = True,dodge=True)


