import pandas as pd
import pandas_profiling
import sys

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", palette="muted", color_codes=True, context='paper')
sns.set_style({'font.sans-serif': ['simhei', 'Arial']})

# 重塑数据类型，减少数据内存使用量
def reduce_mem_usage(df):
    start_mem_b = df.memory_usage().sum()
    start_mem_mb = start_mem_b / 1024 **2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem_mb))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem_b = df.memory_usage().sum()
    end_mem_mb = end_mem_b / 1024 **2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem_mb))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
def reduce_usage(df):
    def mem_usage(pandas_obj):
        if isinstance(pandas_obj,pd.DataFrame):
            usage_b = pandas_obj.memory_usage(deep=True).sum()
        else:
            usage_b = pandas_obj.memory_usage(deep=True)
        usage_mb = usage_b/1024**2
        return "{:03.2f} MB".format(usage_mb)
    for dtype in ['float','int','object']:
        selected_dtype=df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 **2
        print('Average memory usage for {} columns: {:03.2f} MB'.format(dtype,mean_usage_mb))
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')
    print(f'df_int mem usage is: {mem_usage(df_int)}')
    print(f'converted_int mem usage is: {mem_usage(converted_int)}')
    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric,downcast='float')
    print(f'df_float mem usage is: {mem_usage(df_float)}')
    print(f'converted_float mem usage is: {mem_usage(converted_float)}')
    df_obj = df.select_dtypes(include=['object'])
    converted_obj = pd.DataFrame()
    for col in df_obj.columns:
        num_unique_values = len(df_obj[col].unique())
        num_total_values = len(df_obj[col])
        if num_unique_values/num_total_values < 0.5:
            converted_obj.loc[:,col]=df_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = df_obj[col]
    print(f'df_obj mem usage is: {mem_usage(df_obj)}')
    print(f'converted_obj mem usage is: {mem_usage(converted_obj)}')

    df[converted_float.columns]=converted_float
    df[converted_int.columns]=converted_int
    df[converted_obj.columns]=converted_obj

    return df

# 输出数据概况HTML
def profile_html(df,title):
    data = reduce_mem_usage(df)
    prf = pandas_profiling.ProfileReport(data,title=title)
    prf.to_file('report.html')

# 查看数据中特征缺失，唯一值情况
def nuique_missing_value(df):
    data = reduce_mem_usage(df)
    print(f'there are {data.isnull.any().sum()} columns in dataset with missing values')

    # have_null_fea_dict = (data.isnull().sum()/len(data)).to_dict
    # fea_null_morethanhalf = {}
    # for k,v in have_null_fea_dict.items():
    #     if v>0.5:
    #         fea_null_morethanhalf[k] = v
    missing = data.isnull().sum()/len(data)
    missing = missing[missing>0]
    missing.sort_values(inplace=True)
    missing.plot.bar()

    one_value_fea = [col for col in data.columns if data[col].nunique()<=1]
    print(f'there are {len(one_value_fea)} columns in data with one unique values and they are \n{one_value_fea}')
    return one_value_fea

# 特征数值类型及对象类型
def feature_type(df):
    data = reduce_mem_usage(df)
    numerical_fea = list(data.select_dtypes(exclude=['category']).columns)
    category_fea = list(filter(lambda x:x not in numerical_fea,list(data.columns)))
    #划分数值型变量中的连续型变量和离散型变量
    numerical_serial_fea = [col for col in numerical_fea if data[col].nunique>10]
    numerical_noserial_fea = [col for col in numerical_fea if data[col].nunique<=10]
    # 离散型变量查看
    for col in numerical_noserial_fea:
        print(data[col].value_counts())

    # 连续型变量查看
    f = pd.melt(data,value_vars=numerical_serial_fea)
    g = sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False)
    g = g.map(sns.distplot,"value")
    return numerical_fea,category_fea

# 变量分布可视化
class var_dist_plot(object):
    def __init__(self,df,label,category_fea,numrical_serial_fea):
        self.data =df.copy()
        self.label = label
        self.category_fea = category_fea
        self.numrical_serial_fea = numrical_serial_fea
        self.data_fr = self.data.loc[self.data[label]==1]
        self.data_nofr = self.data.loc[self.data[label]==0]
    def single_var_dist(self,single_var):
        data = self.data
        plt.figure(figsize=(8,8))
        sns.barplot(data[single_var].value_counts(dropna=False)[:20],
                    data[single_var].value_counts(dropna=False).keys()[:20])
        plt.show()
    # 类别变量在不同y值上的分布
    def category_var_plot_labels(self):
        category_fea = self.category_fea
        data_fr = self.data_fr
        data_nofr = self.data_nofr
        for fea in category_fea:
            fig,((ax1,ax2)) = plt.subplots(1,2,figsize=(15,6))
            data_fr.groupby(fea)[fea].count().plot(kind='barh',ax = ax1,title=f'count of {fea} label=1')
            data_nofr.groupby(fea)[fea].count().plot(kind='barh',ax = ax2,title=f'count of {fea} label=0')
            plt.show()

    # 连续变量在不同y值上的分布
    def numrical_var_plot_labels(self):
        numrical_serial_fea = self.numrical_serial_fea
        data =self.data
        label = self.label
        data_fr = self.data_fr
        data_nofr = self.data_nofr
        # 变量在不同y值上的分布
        for fea in numrical_serial_fea:
            fig,((ax1,ax2)) = plt.subplots(1,2,figsize=(15,6))
            data_fr[fea].plot(kind='hist',bins=100,title = f'{fea} - label=1',color = 'r',ax = ax1,xlim=(-3,10))
            data_nofr[fea].plot(kind='hist',bins=100,title = f'{fea} - label=0',color = 'r',ax = ax2,xlim=(-3,10))
            plt.show()
        # 数据集不同y值的计数分布与变量在不同y值求和分布对比
        for fea in numrical_serial_fea:
            total = len(data)
            total_amt = data.groupby(label)[fea].sum().sum()
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plot_tr = sns.countplot(x=label, data=data)
            plot_tr.set_title("dataset Distribution \n 0: good user | 1: bad user", fontsize=14)
            plot_tr.set_xlabel("Is fraud by count", fontsize=16)
            plot_tr.set_ylabel('Count', fontsize=16)
            for p in plot_tr.patches:
                height = p.get_height()
                plot_tr.text(p.get_x() + p.get_width() / 2.,
                             height + 3,
                             '{:1.2f}%'.format(height / total * 100),
                             ha="center", fontsize=15)

            percent_amt = (data.groupby([label])[fea].sum())
            percent_amt = percent_amt.reset_index()
            plt.subplot(122)
            plot_tr_2 = sns.barplot(x=label, y=fea, dodge=True, data=percent_amt)
            plot_tr_2.set_title(f"Total Amount in {fea}  \n 0: good user | 1: bad user", fontsize=14)
            plot_tr_2.set_xlabel("Is fraud by percent", fontsize=16)
            plot_tr_2.set_ylabel('Total Loan Amount Scalar', fontsize=16)
            for p in plot_tr_2.patches:
                height = p.get_height()
                plot_tr_2.text(p.get_x() + p.get_width() / 2.,
                               height + 3,
                               '{:1.2f}%'.format(height / total_amt * 100),
                               ha="center", fontsize=15)
            plt.show()













