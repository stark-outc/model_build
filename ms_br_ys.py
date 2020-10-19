# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 11:45
# @Author  : Daiyong
import pickle

import pandas as pd
import numpy as np
import toad
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from copy import deepcopy
import bs_meathod
import xgboost as xgb


def group_metric(df, group_col):
    _ks = []
    score_name = []
    score_list = [j for j in df.columns.tolist() if 'score' in j]
    for m, n in df.groupby(group_col):
        if m == '众邦开发1W':
            n = n[~n['month'].isin(['201911'])]
        _ks2 = [m]
        for x, i in enumerate(score_list):
            _score_name = score_list[x]
            if _score_name not in score_name:
                score_name.append(_score_name)
            _ks1 = bs_meathod.ks(n.y, n[i], m + "_" + i)
            bs_meathod.AUC(n.y, n[i], m + "_" + i)
            _ks2.append(_ks1)
        _ks.append(deepcopy(_ks2))
    _ks = pd.DataFrame(_ks, columns=[group_col] + score_name)
    return _ks


def out_of_fold_train(clf, train, test, K=10, random_seed=11, shuffle=True):
    kf = StratifiedKFold(n_splits=K, random_state=random_seed, shuffle=shuffle)
    y = train.y
    X = train.drop("y", axis=1)
    X_test = test.drop("y", axis=1)
    y_valid_pred = 0 * y
    y_test_pred = 0
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        kf_y_train, kf_y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        kf_x_train, kf_x_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
        eval_set = [(kf_x_valid, kf_y_valid)]
        fit_model = clf.fit(kf_x_train, kf_y_train,
                            eval_set=eval_set,
                            eval_metric='auc',
#                            early_stopping_rounds=50,
                            verbose=False
                            )
        print(" fold:{}, Best N trees ={}, Best auc ={} ".format(i, fit_model.best_iteration + 1, fit_model.best_score))
        pickle.dump(fit_model, open('br_ys_model_' + str(i) + '.dat', 'wb'))
        pred = fit_model.predict_proba(kf_x_valid)[:, 1]
        y_valid_pred.iloc[test_index] = pred
        y_test_pred += fit_model.predict_proba(X_test)[:, 1]
        del kf_x_train, kf_x_valid, kf_y_train, kf_y_valid
    y_test_pred /= K
    bs_meathod.AUC(y, y_valid_pred, 'AUC for full training set')
    bs_meathod.AUC(test.y, y_test_pred, '测试集')
    bs_meathod.ks(y, y_valid_pred, 'AUC for full training set')
    bs_meathod.ks(test.y, y_test_pred, '测试集')
    return y_valid_pred, y_test_pred


def kq_ys(df, spec_type, type_mark):
    kequn_old_type = spec_type
    metric_col = ['allnum', 'orgnum']
    date_col = ['als_d7', 'als_d15', 'als_m1', 'als_m3', 'als_m6', 'als_m12']
    yewu_ys_col = []
    normal_col = []
    for i in date_col:
        for j in metric_col:
            _col = [i + x + j for x in kequn_old_type]
            _test_data = df[[i + x + j for x in kequn_old_type]].copy()
            _test_data.fillna(0, inplace=True)
            for y in kequn_old_type:
                _col2 = deepcopy(_col)
                nl_col = _col[kequn_old_type.index(y)]
                normal_col.append(nl_col)
                _col2.remove(nl_col)
                ys_col = [i + x + type_mark + j for x in kequn_old_type][kequn_old_type.index(y)]
                _test_data[ys_col] = np.apply_along_axis(lambda x: x[0] / x.sum() if x.sum() != 0 else 0, axis=1,
                                                         arr=_test_data[[nl_col] + _col2].values)
                yewu_ys_col.append(_test_data[[ys_col]].copy())
    yewu_ys_data = pd.concat(yewu_ys_col, axis=1)
    return yewu_ys_data


def average_num(x):
    if 'm3' in x:
        return 90
    elif 'm6' in x:
        return 180
    elif 'm12' in x:
        return 360
    else:
        raise ValueError('变量无时间变量')


def br_ys_func(df):
    """
    :param df: 所有百融变量DataFrame
    :return: 衍生后的百融变量DataFrame
    """
    df2 = df[[i for i in df.columns.tolist() if 'als_' in i]].copy()
    # 业务类型衍生,
    yewu_type = ['_id_pdl_', '_id_caon_', '_id_rel_', '_id_caoff_', '_id_cooff_', '_id_af_', '_id_coon_', '_id_oth_']
    metric_col = ['allnum', 'orgnum']
    date_col = ['als_d7', 'als_d15', 'als_m1', 'als_m3', 'als_m6', 'als_m12']
    yewu_ys_col = []
    normal_col = []
    for i in date_col:
        for j in metric_col:
            _col = [i + x + j for x in yewu_type]
            _test_data = df2[[i + x + j for x in yewu_type]].copy()
            _test_data.fillna(0, inplace=True)
            for y in yewu_type:
                _col2 = deepcopy(_col)
                nl_col = _col[yewu_type.index(y)]
                normal_col.append(nl_col)
                _col2.remove(nl_col)
                ys_col = [i + x + 'Y1_' + j for x in yewu_type][yewu_type.index(y)]
                _test_data[ys_col] = np.apply_along_axis(lambda x: x[0] / x.sum() if x.sum() != 0 else 0, axis=1,
                                                         arr=_test_data[[nl_col] + _col2].values)
                yewu_ys_col.append(_test_data[[ys_col]].copy())
    yewu_ys_data = pd.concat(yewu_ys_col, axis=1)

    # 客群类型
    kq1_type = ['_id_nbank_p2p_', '_id_nbank_mc_', '_id_nbank_ca_', '_id_nbank_cf_', '_id_nbank_com_', '_id_nbank_oth_']
    kq1_data = kq_ys(df2, kq1_type, 'kq1_')
    kq2_type = ['_id_nbank_nsloan_', '_id_nbank_autofin_', '_id_nbank_sloan_', '_id_nbank_cf_', '_id_nbank_cons_',
                '_id_nbank_finlea_', '_id_nbank_else_']
    kq2_data = kq_ys(df2, kq2_type, 'kq2_')

    # 度量值衍生
    all_num_col = [i for i in df2.columns.tolist() if 'allnum' in i or 'orgnum' in i]
    dl1_test_data = df2[all_num_col].copy()
    dl1_test_data.fillna(0, inplace=True)
    for i in all_num_col:
        if 'allnum' in i:
            i2 = i.replace('allnum', 'divorg')
            i3 = i.replace('allnum', 'orgnum')
            dl1_test_data[i2] = np.apply_along_axis(lambda x: x[0] / x[1] if x[1] != 0 else 0, axis=1,
                                                    arr=dl1_test_data[[i, i3]].values)

    all_num_col2 = [i for i in df2.columns.tolist() if
                    ('orgnum' in i or 'allnum' in i) and ('m12' in i or 'm3' in i or 'm6' in i)]
    dl2_test_data = df2[all_num_col2 + [i for i in df2.columns.tolist() if '_id_max_inteday' in i]].copy()
    dl2_test_data.fillna(0, inplace=True)
    for i in all_num_col2:
        num = average_num(i)
        i3 = i.replace('num', 'numadj')
        _var = "_".join(i.split('_')[:2]) + '_id_max_inteday'
        if num >= 90:
            dl2_test_data[i3] = np.apply_along_axis(lambda x: x[0] / (num - x[1]), axis=1,
                                                    arr=dl2_test_data[[i, _var]].values)
        else:
            dl2_test_data[i3] = 0

    # 夜间占比
    metric_col = ['allnum', 'orgnum']
    night_week_col = ['_week_', '_night_']
    bank_or_nbank = ['id_bank', 'id_nbank']
    night_week_data = pd.DataFrame()
    for i in date_col:
        for j in bank_or_nbank:
            for m in night_week_col:
                for n in metric_col:
                    _col = i + '_' + j + m + n
                    _col2 = i + '_' + j + '_' + n
                    ys_col = i + '_' + j + m + n + '_wnrate'
                    night_week_data[_col + '_fz'] = df2[_col].fillna(0)
                    night_week_data[_col2 + '_fm'] = df2[_col2].fillna(0)
                    night_week_data[ys_col] = np.apply_along_axis(lambda x: x[0] / x[1] if x[1] != 0 else 0, axis=1,
                                                                  arr=night_week_data[
                                                                      [_col + '_fz', _col2 + '_fm']].values)
    ys_data_all = pd.concat([
        yewu_ys_data,
        kq1_data,
        kq2_data,
        dl1_test_data[[i for i in dl1_test_data.columns.tolist() if 'divorg' in i]],
        dl2_test_data[[i for i in dl2_test_data.columns.tolist() if 'adj' in i]],
        night_week_data[[i for i in night_week_data.columns.tolist() if 'wnrate' in i]]
    ], axis=1, sort=False)

    return ys_data_all


if __name__ == '__main__':
    ms_kf_br_data = bs_meathod.mysql_data_out(['民生益贷', '民商惠updateTx', '民生易贷第三批'],
                                              data_list=['three', 'label_y', 'bairong'])
    # 数据规整
    ms_kf_br_data = bs_meathod.event_name_check(ms_kf_br_data)
    ms_kf_br_data = bs_meathod.label_y_check(ms_kf_br_data)
    print(ms_kf_br_data.shape)

    # 变量处理
    ms_br_ys_data = br_ys_func(ms_kf_br_data)
    model_data = pd.concat([ms_kf_br_data, ms_br_ys_data], axis=1, sort=False)
    model_data['month'] = model_data['app_date'].map(lambda x: str(int(x))[:6])

    # 切分训练和测试
    kf_data = model_data[~model_data['month'].isin(['202001', '202002', '202003', '202004'])]
    OOT_data = model_data[model_data['month'].isin(['202001', '202002', '202003', '202004'])]

    training, testing = bs_meathod.split_month_self(kf_data, 'y')

    # 计算相关性和IV，进行剔除
    # ex_lis = basic_col + ['month']
    #
    # dev_slct1, drop_lst = toad.selection.select(training, training['y'], empty=0.9,
    #                                             iv=0.02, corr=0.9, return_drop=True, exclude=ex_lis)
    # print("keep:", dev_slct1.shape[1],
    #       "drop empty:", len(drop_lst['empty']),
    #       "drop iv:", len(drop_lst['iv']),
    #       "drop corr:", len(drop_lst['corr']))  # # keep: 448 drop empty: 150 drop iv: 510 drop corr: 358

    # 查看训练和测试集之间的psi，剔除psi 高于0.05的变量
    # psi_data = []
    # for i in dev_slct1.columns.tolist():
    #     if i not in ex_lis:
    #         psi = bs_meathod.psi_for_continue_var(training[i], testing[i],
    #                                               detail=False)
    #         psi_data.append([i, psi])
    # psi_data = pd.DataFrame(psi_data, columns=['col', 'psi'])
    # psi_drop_col = psi_data[psi_data['psi'] > 0.05].col.tolist()
    # features = [i for i in dev_slct1.columns.tolist() if
    #             i not in ex_lis and i not in psi_drop_col and i != 'train_flag']  # 196个变量

    # 切分训练和验证集
    last_model = pickle.load(
        open(r'C:\Users\jiangcheng\PycharmProjects\数据测试\数据测试\分数融合模型依赖文件\ms_br_ys_xgb_20200706.dat', 'rb'))

    features = pd.read_excel(r'C:\Users\jiangcheng\PycharmProjects\模型上线文档\民生_br+tx+hds_开发文档\民生_百融腾讯好多数特殊名单_xgboost_需求文档20200804.xlsx',
                             sheet_name='sheet4.百融xgboost模型过程', header=1)['变量名称'].tolist()

    train, valid = bs_meathod.split_month_self(training, 'y')
    x_train, y_train = train[features], train.y
    x_valid, y_valid = valid[features], valid.y
    ratio = np.sum(train.y == 0) / float(np.sum(train.y == 1))
    dtrain = xgb.DMatrix(training[features], label=training.y)

    br_clf = xgb.XGBClassifier(max_depth=6,
                               n_estimators=73,
                               learning_rate=0.06,
                               booster='gbtree',
                               objective='binary:logistic',
                               n_jobs=3,
                               gamma=1,
                               subsample=1,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               min_child_weight=110,
                               scale_pos_weight=float(ratio),
                               reg_alpha=1,
                               reg_lambda=1,
                               seed=11)
    # 确定参数学习率和树的数量
    cv_result = xgb.cv(br_clf.get_xgb_params(),
                       dtrain,
                       num_boost_round=br_clf.get_xgb_params()['n_estimators'],
                       nfold=5,
                       metrics='auc',
                       early_stopping_rounds=50,
                       callbacks=[xgb.callback.early_stop(30),
                                  xgb.callback.print_evaluation(period=1, show_stdv=True)])

    # 网格搜索参数
    param_grid = {
        # 'max_depth': [1, 2, 3, 4, 5, 6, 7],
        'min_child_weight': [10 * (i + 1) for i in range(30)],
        'gamma': [i for i in range(1, 30, 2)],
        'subsample': [i / 10.0 for i in range(5, 11)],
        'colsample_bytree': [i / 10.0 for i in range(5, 11)],
        'reg_lambda': [i for i in range(1, 30, 2)],
        'reg_alpha': [i for i in range(1, 30, 2)]
    }

    grid_search = GridSearchCV(
        br_clf,
        param_grid,
        scoring='roc_auc',
        cv=5)
    grid_self = bs_meathod.XgbParameterSelect(param_grid, grid_search)
    grid_self.auto_parameter(training[features], training.y)

    # 查看效果
    bs_meathod.ks(y_train, br_clf.predict_proba(x_train, ntree_limit=br_clf.n_estimators)[:, 1], 'train')
    bs_meathod.ks(y_valid, br_clf.predict_proba(x_valid, ntree_limit=br_clf.best_ntree_limit)[:, 1], 'valid')
    bs_meathod.ks(testing.y, br_clf.predict_proba(testing[features], ntree_limit=br_clf.n_estimators)[:, 1], 'test')
    bs_meathod.ks(OOT_data.y, br_clf.predict_proba(OOT_data[features], ntree_limit=br_clf.n_estimators)[:, 1],
                  'oot')
# new
# 项目:test,ks值0.21435558576148406
# 项目:oot,ks值0.2004785924797009
# old
# 项目:test,ks值0.21704367332026447
# 项目:oot,ks值0.1938052473364491
    
    # bs_meathod.ks(testing.y, clf_old.predict_proba(testing[clf_old.get_booster().feature_names], ntree_limit=clf_old.n_estimators)[:, 1], 'test')
    # bs_meathod.ks(OOT_data.y, clf_old.predict_proba(OOT_data[clf_old.get_booster().feature_names], ntree_limit=clf_old.n_estimators)[:, 1],
    #               'oot')
    # bs_meathod.ks(ms_kf_br_data.y, br_clf.predict_proba(ms_kf_br_data[features], ntree_limit=br_clf.best_ntree_limit)[:, 1],
    #               'total')
    # ms_kf_br_data['br_score'] = br_clf.predict_proba(ms_kf_br_data[features], ntree_limit=br_clf.best_ntree_limit)[:, 1]

    # 保存训练、测试及OOT标识
    # ms_kf_br_data['flag'] = ms_kf_br_data.index.map(lambda x: 'training' if x in training.index.tolist() else (
    #     'testing' if x in testing.index.tolist() else ('oot' if x in OOT_data.index.tolist() else 'unknow')
    # ))

    # 交叉验证
    # clf = xgb.XGBClassifier(max_depth=5,
    #                         n_estimators=94,
    #                         learning_rate=0.08,
    #                         booster='gbtree',
    #                         objective='binary:logistic',
    #                         n_jobs=3,
    #                         gamma=30,
    #                         subsample=0.8,
    #                         colsample_bytree=0.5,
    #                         colsample_bylevel=1,
    #                         min_child_weight=100,
    #                         scale_pos_weight=float(ratio),
    #                         reg_alpha=40,
    #                         reg_lambda=20,
    #                         seed=11)
    #
    train_score, test_score = out_of_fold_train(br_clf, training[features + ['y']], testing[features + ['y']], K=5)
    br_clf.fit(training[features], training.y, eval_metric='auc', verbose=1)

    # 验证其他项目, 加载旧模型
    clf_old = pickle.load(open(r'C:\Users\jiangcheng\PycharmProjects\数据测试\分数融合模型依赖文件\ms_br_ys_xgb_20200706.dat', 'rb'))
    br_ys_data = bs_meathod.mysql_data_out(['幸福招金', '众邦开发1W', '众邦开发3W'], ['three', 'label_y', 'bairong'])
    br_ys_data = bs_meathod.label_y_check(br_ys_data)
    br_ys_data = bs_meathod.event_name_check(br_ys_data)
    br_ys_data_oth = br_ys_func(br_ys_data)
    br_ys_data = pd.concat([br_ys_data, br_ys_data_oth], axis=1, sort=False)

    br_ys_data['prob'] = br_clf.predict_proba(br_ys_data[features], ntree_limit=br_clf.n_estimators)[:, 1]
    br_ys_data['prob_old'] = last_model.predict_proba(br_ys_data[last_model.get_booster().feature_names],
                                                      ntree_limit=last_model.n_estimators)[:, 1]

    # 交叉验证分数
    # prob = br_ys_data['prob'] * 0
    # for i in range(5):
    #     prob_name = 'prob_cross_' + str(i)
    #     clf_c = pickle.load(open('br_ys_model_{}.dat'.format(i), 'rb'))
    #     br_ys_data[prob_name] = clf_c.predict_proba(br_ys_data[features], ntree_limit=clf_c.best_iteration + 1)[:, 1]
    #     prob += br_ys_data[prob_name]
    # br_ys_data['cross_prob'] = prob / 5
    br_ys_data['month'] = br_ys_data['app_date'].map(lambda x: str(x)[:6])
    ks_data = []
    for i, j in br_ys_data.groupby('event_name'):
        if i == '众邦开发1W':
            j = j[j['month'] != '201911']
        ks = bs_meathod.ks(j.y, j.prob, i)
        ks2 = bs_meathod.ks(j.y, j.prob_old, i)
        auc = bs_meathod.AUC(j.y, j.prob, i)
        auc2 = bs_meathod.AUC(j.y, j.prob_old, i)
        ks_data.append([i, ks, ks2, auc, auc2])
    ks_data = pd.DataFrame(ks_data, columns=['项目', 'ks', 'ks_old',  'auc', 'auc_old'])

# 存模型文件和pmml
# pickle.dump(br_clf, open(r'ms_br_xgb_20200806_online.dat', 'wb'))
# from sklearn2pmml.pipeline import PMMLPipeline
# from sklearn2pmml import sklearn2pmml
#
# pipeline = PMMLPipeline([("classifier", br_clf)])
# sklearn2pmml(pipeline, "ms_br_xgb_20200806_online.pmml",
#              with_repr=True)
# before_data = model_data[~model_data['app_date'].isin['202001']]