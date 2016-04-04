import pandas as pd
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor

'''*****************************************************************'''
'''****************************load data****************************'''
'''*****************************************************************'''
tot = pd.read_csv('data_transform/tot.csv')
X = tot[tot['target']!=-1].copy()
tst = tot[tot['target']==-1].copy()
y = X['target'].copy()
tst_ID = tst['ID'].copy().values
#submit_attr = 'PredictedProb'

'''*****************************************************************'''
'''************************submission format************************'''
'''*****************************************************************'''
def make_submission(proba, filepath, filename, ID=tst_ID):
    df = pd.DataFrame({'ID':ID, 'PredictedProb': proba})
    df.to_csv(os.path.join(filepath, filename), index=False)
filepath = 'output/param_1'


'''*****************************************************************'''
'''***************************Extra Trees***************************'''
'''*****************************************************************'''
#etc
etc = ExtraTreesClassifier(n_estimators=3000, max_depth=30, min_samples_leaf=2, verbose=True,
                           criterion='entropy', max_features=0.55, random_state=9527)
etc.fit(X.drop(['target', 'ID'], axis=1), y)
yetc = etc.predict_proba(tst.drop(['target', 'ID'], axis=1))[:, 1]
make_submission(yetc, filepath, 'yetc_3000.csv')

#etc with gini
etc_gini = ExtraTreesClassifier(n_estimators=3000, max_depth=30, min_samples_leaf=2, verbose=True,
                           criterion='gini', max_features=0.55, random_state=9527)
etc_gini.fit(X.drop(['target', 'ID'], axis=1), y)
yetc_gini = etc_gini.predict_proba(tst.drop(['target', 'ID'], axis=1))[:, 1]
make_submission(yetc_gini, filepath, 'yetc_3000_gini.csv')

#etr
etr = ExtraTreesRegressor(n_estimators=3000, max_depth=30, min_samples_leaf=2, verbose=True,
                          criterion='mse', max_features=0.55, random_state=9527)
etr.fit(X.drop(['target', 'ID'], axis=1), y)
yetr = etr.predict(tst.drop(['target', 'ID'], axis=1))
make_submission(yetr, filepath, 'yetr_3000.csv')


'''*****************************************************************'''
'''**************************Random Forest**************************'''
'''*****************************************************************'''
#rfc
rfc = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_leaf=5, verbose=True,
                             criterion='entropy', max_features=0.5, random_state=9527)
rfc.fit(X.drop(['target', 'ID'], axis=1), y)
yrfc = rfc.predict_proba(tst.drop(['target', 'ID'], axis=1))[:, 1]
make_submission(yrfc, filepath, 'yrfc.csv')

#rfr
rfr = RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_leaf=20, verbose=True,
                            criterion='mse', max_features=0.5, random_state=9527)
rfr.fit(X.drop(['target', 'ID'], axis=1), y)
yrfr = rfr.predict(tst.drop(['target', 'ID'], axis=1))
make_submission(yrfr, filepath, 'yrfr.csv')


'''*****************************************************************'''
'''*************************Gradient Boost**************************'''
'''*****************************************************************'''
#gbc
gbc = XGBClassifier(n_estimators=1100, max_depth=10, min_child_weight=2, learning_rate=0.01, 
                    silent=False, subsample=0.85, colsample_bytree=0.5, base_score=0.76, seed=9527)
gbc.fit(X.drop(['target', 'ID'], axis=1), y)
ygbc = gbc.predict_proba(tst.drop(['target', 'ID'], axis=1))[:, 1]
make_submission(ygbc, filepath, 'ygbc_1100.csv')

#gbr
gbr = XGBRegressor(n_estimators=1100, max_depth=10, min_child_weight=2, learning_rate=0.01, 
                   silent=False, subsample=0.85, colsample_bytree=0.5, base_score=0.76, seed=9527)
gbr.fit(X.drop(['target', 'ID'], axis=1), y)
ygbr = gbr.predict(tst.drop(['target', 'ID'], axis=1))
make_submission(ygbr, filepath, 'ygbr.csv')



















