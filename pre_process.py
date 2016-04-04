import pandas as pd
from sitefunc import *

'''*****************************************************************'''
'''****************************load data****************************'''
'''*****************************************************************'''
#load data set
X = pd.read_csv('data/train.csv')
tst = pd.read_csv('data/test.csv')
y = X['target'].copy()
tot = X.append(tst)
tot['target'] = tot['target'].fillna(-1)
tot.index = list(range(tot.shape[0]))
#load data info
data_info = pd.read_csv('data_transform/data_info.csv', index_col=0)
groupc = eval(data_info.loc['groupc', 'info'])


'''*****************************************************************'''
'''************create a catfeats-based logistic feature*************'''
'''*****************************************************************'''
from sklearn.linear_model import LogisticRegression
#make one hot cols
oh = to_onehot(tot[groupc])
oh.drop('v22', axis=1, inplace=True)
xoh = G.loc[X.index].copy()

#compute the pearson correlation and choose the top 200 as input
pcoh = compute_pearsonr(xoh, y).abs().sort_values(ascending=False)
pcoh_top200 = pcoh[:200].index
oh = oh[pcoh_top200]
xoh = xoh[pcoh_top200]

#logistic regression
logit = LogisticRegression(C=10, solver='lbfgs', max_iter=300)
logit.fit(xoh, y)
logit_feat = logit.predict_log_proba(oh)[:, 1]
tot['logit_feat'] = logit_feat


'''*****************************************************************'''
'''****************************factorize****************************'''
'''*****************************************************************'''
for col in groupc:
    tot[col] = pd.factorize(tot[col], na_sentinel=-9999)[0]


'''*****************************************************************'''
'''************************numeric transform************************'''
'''*****************************************************************'''
tot['v50'] = tot['v50'] ** 0.125
#tot['v62'] = np.log(tot['v62'] + 0.1)
tot = tot.fillna(-9999)


'''*****************************************************************'''
'''***********************drop some features************************'''
'''*****************************************************************'''
tot.drop(['v8', 'v25', 'v46', 'v54', 'v63', 'v89'], axis=1, inplace=True) #group1 feats
tot.drop(['v107', 'v79', 'v75', 'v110'], axis=1, inplace=True) #cat feats


'''*****************************************************************'''
'''*****************************output******************************'''
'''*****************************************************************'''
tot.to_csv('data_transform/tot.csv', index=False)
























