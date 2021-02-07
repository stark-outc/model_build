# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 17:48
# @Author  : Daiyong
from copy import deepcopy
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import pymysql
from scipy.stats import distributions
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
import toad
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine


engine_outer = create_engine(
    "mysql+pymysql:")

class XgbParameterSelect:
    def __init__(self, param_grid, grid_search):
        self.param_grid = param_grid
        self.grid_search = grid_search
        self.bst = grid_search.estimator
    def fit(self,training_data,training_label):
        self.grid_search.fit(training_data, training_label)
    def auto_parameter(self, training_data,training_label):
        for i, j in self.param_grid.items():
            _dict_new = {i:j}
            self.grid_search.param_grid = _dict_new
            print(self.grid_search)
            self.fit(training_data, training_label)
            print('best_params:', self.grid_search.best_params_)
            self.grid_search.estimator.set_params(**self.grid_search.best_params_)



def lr_model(x, y, offx, offy, C):
    """
    :param x: 训练集自变量
    :param y: 训练集因变量
    :param offx: 测试集自变量
    :param offy: 测试集因变量
    :param C: C为正则化系数λ的倒数，通常默认为1
    :return: 无返回值
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=C, class_weight='balanced')
    # penalty : str, ‘l1’（‘liblinear’） or ‘l2’（‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’）, default: ‘l2’
    # a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    # b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    # c) newton - cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    # d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
    model.fit(x, y)

    y_pred = model.predict_proba(x)[:, 1]
    fpr_dev, tpr_dev, _ = roc_curve(y, y_pred)
    train_ks = abs(fpr_dev - tpr_dev).max()
    print('train_ks : ', train_ks)

    y_pred = model.predict_proba(offx)[:, 1]
    fpr_off, tpr_off, _ = roc_curve(offy, y_pred)
    off_ks = abs(fpr_off - tpr_off).max()
    print('off_ks : ', off_ks)

    from matplotlib import pyplot as plt
    plt.plot(fpr_dev, tpr_dev, label='train')
    plt.plot(fpr_off, tpr_off, label='off')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()
    return model


class MysqlService:
    def __init__(self, db_host, db_user, db_pwd, db_port=3306):
        # 获取数据库参数
        self.__db_host = db_host
        self.__db_port = db_port
        self.__db_user = db_user
        self.__db_pwd = db_pwd
        self.connect = create_engine(
            'mysql+pymysql://{}:{}@{}:{}/mk_kuanbiao?charset=utf8mb4'.format(
                self.__db_user, self.__db_pwd, self.__db_host, self.__db_port))

    def get_sql_data(self, db_name, sql):
        con = pymysql.connect(
            host=self.__db_host,
            user=self.__db_user,
            passwd=self.__db_pwd,
            db=db_name,
            port=self.__db_port,
            cursorclass=pymysql.cursors.DictCursor,
            charset='utf8')
        try:
            with con.cursor() as cur:
                if 'DELETE' in sql:
                    cur.execute(sql)

                else:
                    cur.execute(sql)
                    # col = cur.description
                    data = cur.fetchall()
                    data = pd.DataFrame(data)
                    return data
        except Exception as e:
            print(e)
        finally:
            con.commit()
            con.close()


def scatter(title, x, y):
    trace = go.Scatter(x=x, y=y,
                       mode='markers',
                       marker=dict(
                           sizemode='diameter',
                           sizeref=1,
                           size=25,
                           # size= feature_dataframe['AdaBoost feature importances'].values,
                           # color = np.random.randn(500), #set color equal to a variable
                           color=y,
                           colorscale='Portland',
                           showscale=True
                       ),
                       text=x
                       )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        title=title,
        hovermode='closest',
        #         xaxis= dict(
        #             title= 'Pop',
        #             ticklen= 5,
        #             zeroline= False,
        #             gridwidth= 2,
        #         ),
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='scatter2010')


def visual_variable(df, variable_list, target_col):
    plt.figure(figsize=(12, 4*len(variable_list)))
    for i, feature in enumerate(variable_list):
        plt.subplot(len(variable_list), 1, i + 1)
        sns.kdeplot(df.loc[df[target_col] == 0, feature], label='target == 0')
        sns.kdeplot(df.loc[df[target_col] == 1, feature], label='target == 1')
        plt.title('Distribution of %s by Target Value' % feature)
        plt.xlabel('%s' % feature)
        plt.ylabel('Density')
    plt.tight_layout(h_pad = 2.5)


def br_sepcial_decode(df):
    import json
    col_set = set()
    for i in df[df['response_code'] == '0000']['response_data'].tolist():
        i2 = json.loads(i)
        col_set = col_set | set(i2.keys())
    for i in list(col_set):
        df[i] = df['response_data'].map(lambda x: np.nan if str(x) == 'null' else (
            json.loads(x)[i] if i in json.loads(x).keys() else np.nan
        ))
    return df


def feature_woe_iv(x_series, y_series, nan: str = '未知'):
    """
    :param x_series: 已分箱或离散型变量
    :param y_series: y变量（二分类）
    :param nan: 变量缺失值填充值，默认‘未知’
    :return: DataFrame
    """
    x_series = x_series.fillna(nan)
    df = pd.concat([x_series, y_series], axis=1)
    df.columns = ['x', 'y']

    grouped = df.groupby('x')['y']  # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / \
                            result_df['good'].sum()  # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比
    result_df['total_pct'] = result_df['total'] / \
                             result_df['total'].sum()  # 总客户占比

    result_df['bad_rate'] = result_df['bad'] / result_df['total']  # 坏比率

    result_df['woe'] = np.log(
        result_df['bad_pct'] /
        result_df['good_pct'])  # WOE
    result_df['iv'] = (result_df['bad_pct'] -
                       result_df['good_pct']) * result_df['woe']  # IV

    print("该变量IV = {}".format(result_df['iv'].sum()))
    result_df = result_df.reset_index()
    return result_df


def feature_select(df, target, exclude, empty=0.9, iv=0.02, corr=0.7, return_drop=False):
    dev_slct1, drop_lst = toad.selection.select(df, df[target], empty=empty,
                                                iv=iv, corr=corr, return_drop=return_drop, exclude=exclude)
    if return_drop:
        return dev_slct1, drop_lst
    else:
        return dev_slct1


def mutil_feature_iv(df, target, method='dt', **kwargs):
    # method: dt,chi,quantile,step,kmeans, 5种方法
    # C:\Users\jiangcheng\Anaconda3\Lib\site - packages\toad\merge.pyx
    # DTMerge(feature, target, nan=-1, n_bins=None, min_samples=1)
    # StepMerge(feature, nan=None, n_bins=None, clip_v=None, clip_std=None, clip_q=None)
    # QuantileMerge(feature, nan=-1, n_bins=None, q=None)
    # KMeansMerge(feature, target=None, nan=-1, n_bins=None, random_state=1)
    # ChiMerge(feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=-1, balance=True)
    # 以上方法，n_bins如为空，默认取10
    iv_data = []
    for i in df.columns.tolist():
        try:
            dt_new = toad.stats.IV(df[i], df[target], method='dt', **kwargs)
            iv_data.append([i, dt_new])
        except ValueError as e:
            print(e)
            print(i)
    iv_data = pd.DataFrame(iv_data, columns=['col_name', 'IV值'])
    return iv_data


def split_month_self(df,label_col,month,frac=0.8, random_seed=101):
    df2 = df.copy()
    if label_col not in df2.columns.tolist():
        assert KeyError('输入数据集不包含y')
    model_X = []
    model_Y = []
    for i, j in df2.groupby([month]):
#        print(i, j.shape)
        _model_X = j[j[label_col] == 0]
        _model_Y = j[j[label_col] == 1]
        _model_X[month] = i
        _model_Y[month] = i
        _model_X = _model_X.sample(frac=frac, replace=False, random_state=random_seed)
        _model_Y = _model_Y.sample(frac=frac, replace=False, random_state=random_seed)
        model_X.append(deepcopy(_model_X))
        model_Y.append(deepcopy(_model_Y))
    model_X = pd.concat(model_X, sort=False)
    model_Y = pd.concat(model_Y, sort=False)
    training = pd.concat([model_X, model_Y], sort=False)
    df2['train_flag'] = df2.index.map(
        lambda x: 1 if x in training.index.tolist() else 0)
    testing = df2[df2['train_flag'] == 0]
    return training, testing


def split_self(df,label_col, frac=0.8, random_seed=101):
    df2 = df.copy()
    if label_col not in df.columns.tolist():
        assert KeyError('输入数据集不包含y')
    model_X = df2[df2[label_col] == 0]
    model_Y = df2[df2[label_col] == 1]
    model_X = model_X.sample(frac=frac, replace=False, random_state=random_seed)
    model_Y = model_Y.sample(frac=frac, replace=False, random_state=random_seed)
    training = pd.concat([model_X, model_Y], sort=False)
    df2['train_flag'] = df2.index.map(
        lambda x: 1 if x in training.index.tolist() else 0)
    testing = df2[df2['train_flag'] == 0]
    return training, testing.drop('train_flag', axis=1)


def ks(y, prob, title):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, prob)
    ks = max(tpr - fpr)
    print("项目:{},ks值{}".format(title, ks))
    return ks


def score_standard_cal(score, target, return_model=False, **args):
    '分数统一校准'
    X = score.copy()
    y = target.copy()

    from sklearn.linear_model import LogisticRegression
    # model_lr = LogisticRegression(**args, solver='lbfgs', class_weight='balanced')
    model_lr = LogisticRegression(**args, solver='lbfgs')
    model_lr.fit(X, y)

    if return_model:
        return model_lr
    else:
        return model_lr.predict_proba(X)[:, 1]


def AUC(y, prob, title):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)
#    print("项目：{},AUC值{}".format(title, roc_auc))
    return roc_auc


def AUC_plot(y, prob, title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, prob)
    roc_auc = auc(tpr,fpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, lw=2, label='{}:ROC curve (area = {:0.3f})'.format(title, roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()


def ks_plot_self(y, prob, title, bins=20, return_detail=False):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, prob)
    ks = max(tpr - fpr)
    data = pd.DataFrame(data={'y': y.values, 'prob': prob})
    data.sort_values('prob', ascending=False, inplace=True)
    data['prob_n'] = pd.qcut(data['prob'], q=bins, labels=[i for i in range(bins)], retbins=False, duplicates='drop')
    data['y'] = data['y'].map({1: 'bad', 0: 'good'})
    cross_freq = pd.crosstab(index=data['prob_n'], columns=data['y'], values=data['prob'], aggfunc='count')
    cross_freq.sort_index(ascending=False, inplace=True)
    crossdens = cross_freq.cumsum(axis=0) / cross_freq.sum()
    crossdens['gap'] = abs(crossdens['bad'] - crossdens['good'])
    kong = pd.DataFrame(np.zeros((1, 3)), columns=['bad', 'good', 'gap'])
    crossdens = pd.concat([kong, crossdens], sort=False)
    crossdens.reset_index(inplace=True, drop=True)
    p1 = plt.figure()
    p1.add_subplot()
    plt.plot(crossdens.index, crossdens['good'], lw=2, label='fpr')
    plt.plot(crossdens.index, crossdens['bad'], lw=2, label='tpr')
    plt.plot(crossdens.index, crossdens['gap'], lw=2, label='ks:{:.3f}'.format(ks))
    plt.legend(loc=2)
    if return_detail:
        return crossdens
    else:
        pass


def ks_plot(y, prob, title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y, prob)
    p1 = plt.figure()
    p1.add_subplot()
    data = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'ks':tpr-fpr})
    ks = max(tpr-fpr)
    data.sort_values(['thresholds'], ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)
    plt.plot(data.index, data['fpr'], lw=2, label='{}:fpr'.format(title))
    plt.plot(data.index, data['tpr'], lw=2, label='{}:tpr'.format(title))
    plt.plot(data.index, data['ks'], lw=2, label='{}:ks,value:{:.3f}'.format(title, ks))
    plt.legend(loc=2)


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


def psi_for_continue_var(expected_array, actual_array, bins=10, bucket_type='bins', detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算连续型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param bins: number of percentile ranges to bucket the values into，分箱数, 默认为10
    :param bucket_type: string, 分箱模式，'bins'为等距均分，'quantiles'为按等频分箱
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value:
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_continue_var(expected_array=df['acture_score'][:400],
                             actual_array=df['acture_score'][401:],
                             bins=5, bucket_type='bins', detail=0)
    >>> 0.0059132756739701245
    ------------
    >>> psi_for_continue_var(expected_array=df['acture_score'][:400],
                             actual_array=df['acture_score'][401:],
                             bins=5, bucket_type='bins', detail=1)
    >>>
    	acture_score_range	expecteds	expected(%)	actucalsactucal(%)ac - ex(%)ln(ac/ex)psi	max
    0	[0.021,0.2095]	120.0	30.00	152.0	31.02	1.02	0.033434	0.000341
    1	(0.2095,0.398]	117.0	29.25	140.0	28.57	-0.68	-0.023522	0.000159
    2	(0.398,0.5865]	81.0	20.25	94.0	19.18	-1.07	-0.054284	0.000577	<<<<<<<
    3	(0.5865,0.7751]	44.0	11.00	55.0	11.22	0.22	0.019801	0.000045
    4	(0.7751,0.9636]	38.0	9.50	48.0	9.80	0.30	0.031087	0.000091
    5	>>> summary	400.0	100.00	489.0	100.00	NaN	NaN	0.001214	<<< result
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    import math
    import numpy as np
    import pandas as pd

    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    # expected_array = pd.Series(expected_array)
    # actual_array = pd.Series(actual_array)
    if expected_array.shape[0] == 0 or actual_array.shape[0] == 0:
        return 999999

    if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
        raise Exception("输入数据expected_array只能是数值型, 不能为string类型")

    """step1: 确定分箱间隔"""

    def scale_range(input_array, scaled_min, scaled_max):
        '''
        ----------------------------------------------------------------------
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        ----------------------------------------------------------------------
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_min: float, 放缩后的最大值
        ----------------------------------------------------------------------
        :return input_array: numpy array of original values, 放缩后的数列
        ----------------------------------------------------------------------
        '''
        input_array += -np.min(input_array)  # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, 请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array

    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回999999
    if np.min(expected_array) == np.max(expected_array):
        return 999999

    breakpoints = np.arange(0, bins + 1) / (bins) * 100  # 等距分箱百分比
    if 'bins' == bucket_type:  # 等距分箱
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif 'quantiles' == bucket_type:  # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    """step2: 统计区间内样本占比"""

    def generate_counts(arr, breakpoints):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        :return acture_score_range_array: 分箱区间
        ----------------------------------------------------------------------
        '''

        def count_in_range(arr, low, high, start):
            '''
            ----------------------------------------------------------------------
            功能: 统计给定区间内的样本数(Counts elements in array between low and high values)
            ----------------------------------------------------------------------
            :param arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            ----------------------------------------------------------------------
            :return cnt_in_range: int, 给定区间内的样本数
            ----------------------------------------------------------------------
            '''
            if start:
                cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
            return cnt_in_range

        cnt_array = np.zeros(len(breakpoints) - 1)
        acture_score_range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i - 1] = count_in_range(arr, breakpoints[i - 1], breakpoints[i], i == 1)
            if 1 == i:
                acture_score_range_array[i - 1] = '[' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                    round(breakpoints[i], 4)) + ']'
            else:
                acture_score_range_array[i - 1] = '(' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                    round(breakpoints[i], 4)) + ']'

        return (cnt_array, acture_score_range_array)

    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    acture_score_range_array = generate_counts(expected_array, breakpoints)[1]

    """step3: 区间放缩"""

    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
             Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0:  # 实际占比
            a_perc = 0.001
        if e_perc == 0:  # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value

    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['acture_score_range'] = acture_score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.001) \
                                                                    / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'acture_score_range': '>>> summary',
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)': 100,
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)': 100,
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max': '<<< result'}], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif not save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)

    return psi_value


def multi_psi_for_continue_var(expected_frame, actual_frame, bins=10, bucket_type='bins', detail=False,
                               save_file_path=None):
    col_list = expected_frame.columns.tolist()
    if detail == True:
        psi_value = []
        for i in col_list:
            expected_array = expected_frame[i]
            actual_array = actual_frame[i]
            _psi = psi_for_continue_var(expected_array, actual_array, bins=bins, bucket_type=bucket_type, detail=detail,
                                        save_file_path=save_file_path)
            if not isinstance(_psi, int):
                _psi['col_name'] = str(i)
                print(i)
                psi_value.append(_psi)
            else:
                print(i)
        psi_all = pd.concat(psi_value, sort=False)
    elif detail == False:
        psi_value = {}
        for i in col_list:
            expected_array = expected_frame[i]
            actual_array = actual_frame[i]
            _psi = psi_for_continue_var(expected_array, actual_array, bins=bins, bucket_type=bucket_type, detail=detail,
                                        save_file_path=save_file_path)
            psi_value[i] = _psi
        psi_all = pd.DataFrame(psi_value, sort=False)
    return psi_all


def get_sql_data(sql):
    """
    :param sql: 连接卢成数据库的sql,账户密码已在代码中
    :return:DataFrame
    """
    con = pymysql.connect(
        host='',
        user='',
        passwd='',
        db='',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor,
        charset='utf8')
    try:
        with con.cursor() as cur:
            cur.execute(sql)
            data = cur.fetchall()
            data = pd.DataFrame(data)
            return data
    except Exception as e:
        print(e)
    finally:
        con.close()


def event_name_check(df):
    df['event_name'] = df[['event_name', 'sign_no']].apply(
        lambda x: x['event_name'] if str(x['sign_no'])[0] != 'U' else (
            '幸福消金1' if int(x['sign_no'].split('U')[1]) <= 10000 else '幸福消金2'
        ), axis=1)
    df['event_name'] = df['event_name'].map(lambda x: '民生易贷' if str(x)[0] == '民' else x)
    return df


def label_y_check(df):
    df['y'] = df['y'].map(lambda x:int(x.split('_')[0]))
    df = df[df['y']!=99]
    return df


def tx_risk_code(x):
    if str(x) == 'nan':
        return np.nan
    else:
        return_dict = {}
        for j in x:
            return_dict.update(j)
        return return_dict

def risk_code_jiexi(df):
    time1 = time.time()
    df['general_tag_jiexi'] = df['general_tag'].map(lambda x: np.nan if x == '0' or str(x) == 'None' else (
        [dict(zip([i.split("-")[0]], [i.split("-")[1]])) for i in x.split('|')]
    ))
    tx_rename_col_dict = {
        'general_v4_20191106_score': 'model_score0',
        'ss1_score': 'model_score1',
        'ss2_score': 'model_score2',
        'ss3_score': 'model_score3',
        'ss4_score': 'model_score4',
        'ss5_score': 'model_score5',
        'general_v5_20200227_score': 'model_score6'
    }
    df.rename(columns=tx_rename_col_dict,inplace=True)
    tx_score_dict = {
        '疑似公开信息失信': 'risk_code01',
        '疑似金融黑产相关': 'risk_code02',
        '疑似信贷恶意行为': 'risk_code03',
        '疑似养卡套现欺诈': 'risk_code04',
        '疑似资料伪造包装': 'risk_code05',
        '疑似资料仿冒行为': 'risk_code06',
        '疑似身份信息不符': 'risk_code07',
        '疑似恶意中介代办': 'risk_code08',
        '疑似涉黄涉恐行为': 'risk_code09',
        '疑似线上赌博行为': 'risk_code10',
        '疑似风险投机行为': 'risk_code11',
        '疑似营销活动欺诈': 'risk_code12',
        '疑似欺诈网络环境': 'risk_code13',
        '疑似欺诈设备环境': 'risk_code14',
        '疑似风险设备环境': 'risk_code15',
        '疑似风险手机账号': 'risk_code16',
        '疑似手机猫池欺诈': 'risk_code17',
        '疑似异常支付行为': 'risk_code18',
        '疑似线上养号小号': 'risk_code19'
    }
    df['general_tag_jiexi2'] = df['general_tag_jiexi'].map(lambda x: tx_risk_code(x))
    for i, j in tx_score_dict.items():
        df[j] = df['general_tag_jiexi2'].map(lambda x:x[i] if pd.notna(x) and i in x.keys() else np.nan)
    time2 = time.time()
    print('腾讯风险数据解析耗时{}s'.format(time2 - time1))
    return df.drop(['general_tag_jiexi','general_tag_jiexi2'], axis=1)


def mysql_data_out(event_name_list, data_list=None):
    if not isinstance(event_name_list, list):
        raise TypeError('event_name_list must be list')
    if len(event_name_list) == 1:
        event_name_list.extend(['凑数', '凑数'])
    df = get_sql_data(
        'select * from three where event_name in {}'.format(tuple(event_name_list)))
    if data_list is None:
        return df
    else:
        if not isinstance(event_name_list, list):
            raise TypeError('data_list must be list')
        data_list = data_list
    for i in data_list:
        _df = get_sql_data(
            'select * from {} where event_name in {}'.format(i, tuple(event_name_list)))
        if i == 'tx_risk':
            _df = risk_code_jiexi(_df)
        if len(_df) == 0:
            pass
        else:
            print(f'输出{i}')
            drop_col = [i for i in _df.columns.tolist(
            ) if i in df.columns.tolist() and i != 'sign_no']
            df = pd.merge(df, _df[[i for i in _df.columns.tolist(
            ) if i not in drop_col]], how='left', on='sign_no')
    return df


def sao_biao():
    from sqlalchemy import VARCHAR, DATETIME
    res = pd.DataFrame()

    list1 = ['label_y', 'three', 'tx_risk', 'tx_dc', 'md5prase', ]
    list2 = ['bairong', 'bairong_reduction', 'bairong_special']
    list3 = ['hailian_half1', 'haoduoshu', 'tianchuang', 'youzu']
    listall = list1 + list2 + list3

    for i in listall:
        _df = pd.read_sql_query('select event_name,count(event_name) from %s group by event_name' % i, engine_outer)
        _df.set_index('event_name', inplace=True)
        _df.columns = [i]
        res = pd.concat([res, _df], axis=1, sort=False)
    res = res.fillna(0).applymap(int)

    _df = pd.read_sql_query('select event_name,create_time from event group by event_name', engine_outer)
    _df.set_index('event_name', inplace=True)
    res = pd.concat([res, _df], axis=1, sort=False)

    sao_biao_res = res.sort_values(by='create_time')
    sao_biao_res['create_time'] = sao_biao_res['create_time'].map(lambda x: x.strftime('%Y-%m-%d'))
    sao_biao_res.reset_index().to_sql('ahead_table_count', engine_outer, if_exists='replace', index=False, dtype={
        'index': VARCHAR(length=255),
        'create_time': DATETIME()
    })

    return sao_biao_res


def time_biao():
    from sqlalchemy import VARCHAR
    df = pd.read_sql_query('select event_name,app_date from three', engine_outer)
    res = df.groupby([df['event_name'], df['app_date'].map(str).map(lambda x: x[:6])]) \
        .size().unstack('event_name').fillna(0).applymap(int).T
    
    res.reset_index().to_sql('ahead_table_time', engine_outer, if_exists='replace', index=False, dtype={
        'app_date': VARCHAR(length=255)
    })
    return res


def out_of_fold_train(clf, train, test, K=10, random_seed=11, shuffle=True, save_model=True):
    import pickle
    from sklearn.model_selection import KFold, StratifiedKFold
    kf = StratifiedKFold(n_splits=K, random_state=random_seed, shuffle=shuffle)
    y = train.y
    X = train.drop("y", axis=1)
    X_test = test.drop("y", axis=1)
    y_valid_pred = 0 * y
    y_test_pred = 0
    psi_test_prob = []
    y_test = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        kf_y_train, kf_y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        kf_x_train, kf_x_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
        eval_set = [(kf_x_valid, kf_y_valid)]
        fit_model = clf.fit(kf_x_train, kf_y_train,
#                            eval_set=eval_set,
#                            eval_metric='auc',
#                            early_stopping_rounds=50,
                            verbose=False
                            )
#        print(" fold:{}, Best N trees ={}, Best auc ={} ".format(i, fit_model.best_iteration + 1, fit_model.best_score))
        if save_model:
            pickle.dump(fit_model, open('br_ys_model_' + str(i) + '.dat', 'wb'))
        pred = fit_model.predict_proba(kf_x_valid)[:, 1]
        y_valid_pred.iloc[test_index] = pred
        y_test_pred += fit_model.predict_proba(X_test)[:, 1]
        psi_test_prob.append(pd.DataFrame(fit_model.predict_proba(X_test)[:, 1], columns=['prob']))
        y_test.extend(list(test.y == 1))
        del kf_x_train, kf_x_valid, kf_y_train, kf_y_valid
    y_test_pred /= K
    psi_test_prob = pd.concat(psi_test_prob, sort=False)['prob']
    print('1样本psi', psi_for_continue_var(
        y_valid_pred[train.y == 1],# k折交叉验证中所有验证集在原训练样本为1的样本上的预测值
        psi_test_prob[y_test],  #test上为1的样本交叉验证后的预测值
        bucket_type='quantiles',
        
        detail=False,
        bins=10
    ))
    print('全量样本psi', psi_for_continue_var(y_valid_pred, psi_test_prob,
                                                  bucket_type='quantiles',
                                                  detail=False,
                                                  bins=10
                                                  ))
    AUC(y, y_valid_pred, '交叉验证-验证集')
    AUC(test.y, y_test_pred, '交叉验证-测试集')
    ks(y, y_valid_pred, '交叉验证-验证集')
    ks(test.y, y_test_pred, '交叉验证-测试集')
    return y_valid_pred, y_test_pred


def get_bayes_param(train_data, features,bayes_params=None):
    from bayes_opt import BayesianOptimization
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    ratio = np.sum(train_data.y == 0) / float(np.sum(train_data.y == 1))
    k_fold=StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
    def rfc_cv(learning_rate, n_estimators, max_depth,
               min_child_weight, gamma, colsample_bytree,
               subsample, reg_lambda, reg_alpha):
        estimator = xgb.XGBClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth),objective='binary:logistic',booster='gbtree',
                                      min_child_weight=int(min_child_weight), gamma=int(gamma),scale_pos_weight=float(ratio),
                                      learning_rate=learning_rate, colsample_bytree=colsample_bytree,seed=11,
                                      subsample=subsample, reg_lambda=int(reg_lambda), reg_alpha=int(reg_alpha))
        cval = cross_val_score(estimator, train_data[features], train_data.y, scoring='roc_auc', cv=k_fold)
        return cval.mean()

    def optimize_rfc(params):
        optimizer = BayesianOptimization(f=rfc_cv, pbounds=params, random_state=11)
        optimizer.maximize(n_iter=20)
        return optimizer
    if bayes_params is None:
        adj_params = {'learning_rate': (0.01, 0.15),
                      'n_estimators': (10, 400),
                      'max_depth': (2, 7),
                      'min_child_weight': (1, 300),
                      'gamma': (0, 10),
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
    for i in ['n_estimators', 'max_depth', 'gamma', 'min_child_weight', 'reg_lambda', 'reg_alpha']:
        params[i] = int(round(params[i]))
    return params


def group_metric(df, group_col, score_key='score'):
    _ks = []
    score_name = []
    score_list = [j for j in df.columns.tolist() if score_key in j]
    for m, n in df.groupby(group_col):
        if m == '众邦开发1W':
            n = n[~n['month'].isin(['201911'])]
        _ks2 = [m, n.shape[0]]
        for x, i in enumerate(score_list):
            _score_name = score_list[x]
            if _score_name not in score_name:
                score_name.append(_score_name)
            _ks1 = ks(n.y, n[i], m + "_" + i)
            AUC(n.y, n[i], m + "_" + i)
            _ks2.append(_ks1)
        
        _ks.append(_ks2)
    _ks = pd.DataFrame(_ks, columns=[group_col, '数据量'] + score_name)
    return _ks
def select_seed(model_data,features):
    import xgboost as xgb
    seed_dict={}
    for i in range(1,15,1):
#        print(i)
        training, testing = split_month_self(model_data, 'y','app_date', frac=0.8, random_seed=i)
        split_data = model_data.copy()
        split_data['y'] = split_data.index.map(lambda x: 1 if x in testing.index.tolist() else 0)
        check_train = xgb.DMatrix(split_data[features], label=split_data.y)
        check_cv = xgb.cv(xgb.XGBClassifier().get_xgb_params(), check_train,
                          num_boost_round=100,
                          nfold=5,
                          metrics='auc',
                          early_stopping_rounds=50,
                          )
        check_cv = check_cv.loc[check_cv.index.max(), :]
        seed_dict[i]=abs(check_cv['test-auc-mean']-0.5)
    min_value=min(seed_dict.values())
    for key,value in seed_dict.items():
        if value==min_value:
            best_seed=key  
    return best_seed


def lift_plot(df,y_true,y_prob,title='test'):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    total = df[y_true].count()
    bad = df[y_true].sum()
    bucket = pd.qcut(df[y_prob].rank(method='first'),10)
    d1=df.groupby(bucket)
    
    d2=pd.DataFrame()
    d2['total'] = d1['y'].count()
    d2.sort_index(ascending=True,inplace=True)
    d2['bad'] = d1['y'].sum()
    d2['good']=d2['total']-d2['bad']
    d2['bad_rate'] =d2['bad']/bad
    d2['cumbad_rate'] =d2['bad'].cumsum()/bad
    d2['total_rate'] = d2['total']/total
    d2['cumtotal_rate'] =d2['total'].cumsum()/total
    d2['lift'] = d2['cumbad_rate']/d2['cumtotal_rate']
    plt.plot(d2['cumtotal_rate'],d2['lift'],'bo-', linewidth=2)
    plt.axhline(1, color='gray', linestyle='--')
    plt.ylim([1.0,2.5])
    plt.title(title, fontsize=15)
    plt.show()

    return d2
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
 
    def Getps_ks(self,df,real_data,data):
        #real_data为真实值，data为原数据
   
        d = []
        for i in data:
            pre_data = [1 if line >=i else 0 for line in data]
            d.append(self.ComuTF(real_data,pre_data))
        return max(d),data[d.index(max(d))]
    
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


