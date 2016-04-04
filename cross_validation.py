import numpy as np
import pandas as pd
import time
import re
from sitefunc.func import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

'''*****************************************************************'''
'''****************************load data****************************'''
'''*****************************************************************'''
tot = pd.read_csv('data_transform/tot.csv')
X = tot[tot['target']!=-1].copy()
y = X['target'].copy()
data_info = pd.read_csv('data_transform/data_info.csv', index_col=0)
groupc = eval(data_info.loc['groupc', 'info'])
groupc.remove('v107')
groupc.remove('v110')
groupc.remove('v75')
groupc.remove('v79')
#the 'logit_feat' is a leaky transform for cross validation
#so drop it now, and generate it later in cv loop
X.drop('logit_feat', axis=1, inplace=True)


'''*****************************************************************'''
'''******************optional feature engineering*******************'''
'''*****************************************************************'''
def create_logit_feat(xtr, xcv, ytr, ycv, cat_feat=groupc, 
                      logit=LogisticRegression(C=10, solver='lbfgs', max_iter=300)):
    x = xtr.append(xcv)
    cat = x[cat_feat]
    cat = to_onehot(cat)
    cat_tr = cat.loc[ytr.index]
    cat_cv = cat.loc[ycv.index]
    logit.fit(cat_tr, ytr)
    logit_feat_tr = logit.predict_log_proba(cat_tr)[:, 1]
    logit_feat_cv = logit.predict_log_proba(cat_cv)[:, 1]
    return logit_feat_tr, logit_feat_cv

def create_bernoulli_feat(xtr, xcv, ytr, ycv, cat_feat=groupc, nb=BernoulliNB()):
    pass

'''*****************************************************************'''
'''************************cross validation*************************'''
'''*****************************************************************'''

'''random forest, extra trees, gradient boosting are available'''
def get_model_name_list():
    model_list = []
    model_list.extend([RandomForestClassifier, RandomForestRegressor])
    model_list.extend([ExtraTreesClassifier, ExtraTreesRegressor])
    model_list.extend([XGBClassifier, XGBRegressor])
    return model_list
    
'''give estimator(in rf, et, gb), train data & cv data, return train loss & cv loss'''
'''if the estimator is a gb model, also return the best iteration of early stopping'''    
def train_model(estimator, xtr, xcv, ytr, ycv):
    model_list = get_model_name_list()
    #for rfc, rfr, etc, etr
    if type(estimator) in model_list[:4]:        
        estimator.fit(xtr, ytr)
        #for rfc, rtc
        if hasattr(estimator, 'predict_proba'):
            train_predict = estimator.predict_proba(xtr)
            cv_predict = estimator.predict_proba(xcv)
        #for rfr, etr
        else:
            train_predict = estimator.predict(xtr)
            cv_predict = estimator.predict(xcv)
        best_iter = 0
    #for xgbc, xgbr 
    elif type(estimator) in model_list[4:]:
        estimator.fit(xtr, ytr, early_stopping_rounds=35, eval_metric='logloss', 
                      eval_set=[(xcv, ycv)], verbose=True)
        best_iter = estimator.best_iteration
        #for xgbc
        if hasattr(estimator, 'predict_proba'):
            train_predict = estimator.predict_proba(xtr, ntree_limit=best_iter)
            cv_predict = estimator.predict_proba(xcv, ntree_limit=best_iter)
        #for xgbr
        else:
            train_predict = estimator.predict(xtr, ntree_limit=best_iter)
            cv_predict = estimator.predict(xcv, ntree_limit=best_iter)
    train_loss = log_loss(ytr, train_predict)
    cv_loss = log_loss(ycv, cv_predict)
    return train_loss, cv_loss, best_iter
    
'''n-fold cross validation, with optional stacking feature engineering'''
def stacking_model(estimator, X, y, folds=5, **kw):
    skf = StratifiedKFold(y, n_folds=folds)
    train_loss_v = []
    cv_loss_v = []
    best_iter_v = []
    for tr_idc, cv_idc in skf:
        #split into train data and cv data 
        xtr = X.ix[tr_idc]
        ytr = y[tr_idc]
        xcv = X.ix[cv_idc]
        ycv = y[cv_idc]
        #optional feature engineering
        if 'logit_feat' in kw:
            tr_logit_feat, cv_logit_feat = kw['logit_feat'](xtr, xcv, ytr, ycv)
            xtr['logit_feat'] = tr_logit_feat
            xcv['logit_feat'] = cv_logit_feat
        #train model
        train_loss, cv_loss, best_iter = train_model(estimator, xtr, xcv, ytr, ycv)
        train_loss_v.append(train_loss)
        cv_loss_v.append(cv_loss)
        best_iter_v.append(best_iter)
    return train_loss_v, cv_loss_v, best_iter_v
    
'''execute the n-fold cv, and write cv infos @ data_transform/cv_info.csv'''
def exec_model(estimator, X, y, folds, **kw):
    start_time = time.time()
    train_loss_v, cv_loss_v, best_iter_v = stacking_model(estimator, X, y, folds, **kw)
    #compute loss.mean and loss.std
    train_loss = np.mean(train_loss_v)
    train_std = np.std(train_loss_v)
    cv_loss = np.mean(cv_loss_v)
    cv_std = np.std(cv_loss_v)
    best_iter_mean = np.mean(best_iter_v)
    best_iter_max = np.max(best_iter_v)
    #output message according to estimator's type
    model_list = get_model_name_list()
    model = re.sub("[^A-Z]", "", str(type(estimator)))
    rounds = estimator.n_estimators
    depth = estimator.max_depth
    if type(estimator) in model_list[:4]:
        criterion = estimator.criterion        
        weight = estimator.min_samples_leaf
        features = estimator.max_features
        subsample = np.nan
        bootstrap = estimator.bootstrap
        learning_rate = np.nan
        #best_iter = np.nan
    if type(estimator) in model_list[4:]:
        criterion = estimator.objective
        weight = estimator.min_child_weight
        features = estimator.colsample_bytree
        subsample = estimator.subsample
        bootstrap = np.nan
        learning_rate = estimator.learning_rate
        #best_iter = estimator.best_iteration
    used_time = time.time() - start_time
    #result to file    
    message = pd.DataFrame([[model, rounds, criterion, depth, weight, features, subsample, 
                             bootstrap, learning_rate, best_iter_mean, best_iter_max, cv_loss, 
                             cv_std, train_loss, train_std, used_time, folds, X.shape, list(kw)]])
    message.to_csv('data_transform\cv_info.csv', index=False, header=False, mode='a')
    #print loop message
    print('model=%s, rounds=%d, criterion=%s, depth=%d, weight=%d, features=%f, subsample=%f, rate=%f done!' 
            % (model, rounds, criterion, depth, weight, features, subsample, learning_rate))
    print('cv_loss=%f, cv_std=%f, train_loss=%f, train_std=%f\n' % (cv_loss, cv_std, train_loss, train_std))


'''*****************************************************************'''
'''************************parameter tuning*************************'''
'''*****************************************************************'''
#just some samples here
#one group of parameters may take 30~120 minutes
#the actual time can be found at data_transform/cv_info.csv
for depth in [10, 20, 30, 40, 50, 60]:
    for weight in [1, 2, 5, 10, 20, 30]:
        rfc = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=weight, 
                                     criterion='entropy', max_features=0.5, random_state=9527)
        etc = ExtraTreesClassifier(n_estimators=600, max_depth=depth, min_samples_leaf=weight, 
                                     criterion='entropy', max_features=0.5, random_state=9527)
        gbc = XGBClassifier(learning_rate=0.01, n_estimators=1500, max_depth=depth, base_score=0.76,
                            min_child_weight=weight, colsample_bytree=0.5, subsample=0.85,seed=9527)            
        exec_model(rfc, X.drop(['ID', 'target'], axis=1), y, 5)
        exec_model(etc, X.drop(['ID', 'target'], axis=1), y, 5)
        exec_model(gbc, X.drop(['ID', 'target'], axis=1), y, 5)
    

''''''





