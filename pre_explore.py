import numpy as np
import pandas as pd
from sitefunc.func import *


'''*****************************************************************'''
'''****************************load data****************************'''
'''*****************************************************************'''
X = pd.read_csv('data/train.csv')
tst = pd.read_csv('data/test.csv')
y = X['target'].copy()
X.drop(['ID'], axis=1, inplace=True)
tst.drop(['ID'], axis=1, inplace=True)
tot = X.append(tst)
tot['target'] = tot['target'].fillna(-1)
tot.index = list(range(tot.shape[0]))


'''*****************************************************************'''
'''************************target & features************************'''
'''*****************************************************************'''
#y.mean() = 0.7612, not balanced target
y.hist()
#categorical / numerical features
cat_feat = []
num_feat = []
for i in range(1, tot.shape[1]):
    if tot[tot.columns[i]].dtype == object:
        cat_feat.append(tot.columns[i])
    else:
        num_feat.append(tot.columns[i])
cat = tot[cat_feat]
num = tot[num_feat]


'''*****************************************************************'''
'''***************************null values***************************'''
'''*****************************************************************'''
#cat feats
cat_null = cat.isnull().sum()
cat_null.sort_values(ascending=False, inplace=True)
cat_null.hist(bins=20)
#find that v3_null = v31_null
(cat[['v3', 'v31']].isnull().sum(axis=1) == 1).sum()

#num feats
num_null = num.isnull().sum()
num_null.sort_values(ascending=False, inplace=True)
num_null.hist(bins=20)
#num feats are not missing at random
group3 = num_null[num_null>100000].index
group2 = num_null[(num_null<100000) & (num_null>99000)].index
group1 = num_null[(num_null<98000) & (num_null>97000)].index
group4 = num_null[num_null<1500].index
#some feats in group4 look like discrete feats (only integers)
groupd = pd.Index(['v38', 'v62', 'v72', 'v129'])
group4 = group4.drop(groupd)


'''*****************************************************************'''
'''******************explore categorical featrues*******************'''
'''*****************************************************************'''
cat_values = pd.Series(index=cat.columns)
for i in range(cat.shape[1]):
    value_count = cat[cat.columns[i]].value_counts()
    cat_values[cat.columns[i]] = len(value_count)
cat_values.sort_values(ascending=False, inplace=True)
'''
v22 looks like noise for it has too many different values (not sure)
precisely, 23419 values in 228714 samples
'''

#contingency matching
mat_contingency_match = pd.DataFrame(index=cat.columns, columns=cat.columns)
pairs_contingency_match = []
for i in range(cat.shape[1]):
    for j in range(i, cat.shape[1]):
        if i == j:
            mat_contingency_match.ix[i, j] = np.nan
        else:
            row_match, col_match = \
            contingency_match(cat[cat.columns[i]], cat[cat.columns[j]])
            mat_contingency_match.ix[i, j] = row_match
            mat_contingency_match.ix[j, i] = col_match
            if row_match > 0.9:
                pairs_contingency_match.append((cat.columns[i], cat.columns[j], row_match))
            if col_match > 0.9:        
                pairs_contingency_match.append((cat.columns[j], cat.columns[i], col_match))
'''
the pairs with v74/v3 are all fake correlation because v74/v3 are extremely unbalanced
v74: A-86, B-227228, C-1400
v3:  A-433, B-108, C-221224

get three groups of categorical features
v91  ←→  v107
v125 -→  v112
v79  -→  v47  -→  v110
'''

#onehot encoding
#cat_onehot = to_onehot(cat, unique_values=500)
pairs_onehot_match = []
for i in range(cat.shape[1]):
    for j in range(i, cat.shape[1]):
        if i != j:
            print('loop (%d,%d), matching %s & %s' % (i, j, cat.columns[i], cat.columns[j]))
            has_onehot_match, onehot_match_set = \
            onehot_match(cat[cat.columns[i]], cat[cat.columns[j]], thresold=0.9)
            if has_onehot_match == True:
                pairs_onehot_match.append(onehot_match_set)
'''
except v79-v47-v110/v91-v107 groups, these one hot columns can match
v3 nan ~ v31 nan, rate = 1.0
v52 nan ~ v91 nan ~ v107 nan, rate = 1.0
v112 S ~ v125 AE, rate = 1.0
v47 H ~ v56 AX, rate = 1.0
v71 F ~ v75 D, rate = 0.9999
v3 C ~ v74 B, rate = 0.9609
v110 C ~ v3 nan, rate = 0.9274
v3 nan ~ v47 D, rate = 0.9115

find that v71 & v75 can be merged into one column,
v71 F ~ v75 D
v71 C/B ~ v75 B
'''

#onehot correlation, "pc" for "pearson correlation", "oh" for "one hot"
catx_onehot = to_onehot(X[cat_feat], unique_values=500)
pcoh = compute_pearsonr(catx_onehot, y).abs().sort_values(ascending=False)
#use following code explore every feature in detail
onehot_corr_cols = catx_onehot.columns.str.startswith('v112')
compute_pearsonr(catx_onehot[catx_onehot.columns[onehot_corr_cols]], y).abs().hist()
'''
can extract some one hot columns with high correlation here
most of the top columns are from v31, v56, v66, v79(v47, v110), v113, 
and the botoom ones are from v56(you again XD!), v112(v125)
'''


'''*****************************************************************'''
'''********************explore numeric featrues*********************'''
'''*****************************************************************'''
#data max, min
dmax = tot.max().sort_values(ascending=False)
dmin = tot.min().sort_values(ascending=False)
'''
it seems that all the feats (except the dicrete ones) are scaled in [0, 20], 
then add some random noise around 10e-7
'''

#data group
g1 = tot[group1]
g2 = tot[group2]
g3 = tot[group3]
g4 = tot[group4.append(groupd)]
xg1 = X[group1]
xg2 = X[group2]
xg3 = X[group3]
xg4 = X[group4.append(groupd)]

#data mean(), data std()
g1mean = g1.mean().sort_values()
g2mean = g2.mean().sort_values()
g3mean = g3.mean().sort_values()
g4mean = g4.mean().sort_values()
g1std = g1.std().reindex(g1mean.index)
g2std = g2.std().reindex(g2mean.index)
g3std = g3.std().reindex(g3mean.index)
g4std = g1.std().reindex(g4mean.index)

#compute linear correlation with y, "pc" for "pearson correlation"
pcy1 = compute_pearsonr(xg1, y).abs().sort_values(ascending=False)
pcy2 = compute_pearsonr(xg2, y).abs().sort_values(ascending=False)
pcy3 = compute_pearsonr(xg3, y).abs().sort_values(ascending=False)
pcy4 = compute_pearsonr(xg4, y).abs().sort_values(ascending=False)
'''
group4 feats (not missing values) are highly correlated with target, 
especially v50, whose pc-value reaches 0.2417
most of group1 feats are uncorrelated

may drop some uncorrelated feats later 
'''

#compute linear correlation in group feats
pcg1 = compute_pearsonr(g1).abs()
pcg2 = compute_pearsonr(g2).abs()
pcg3 = compute_pearsonr(g3).abs()
pcg4 = compute_pearsonr(g4).abs()
pcg1g2 = compute_pearsonr(g1, g2).abs() #max = 0.7611
pcg1g3 = compute_pearsonr(g1, g3).abs() #max = 0.8090
pcg1g4 = compute_pearsonr(g1, g4).abs() #max = 0.1334
pcg2g3 = compute_pearsonr(g2, g3).abs() #max = 0.8249
pcg2g4 = compute_pearsonr(g2, g4).abs() #max = 0.1692
pcg3g4 = compute_pearsonr(g3, g4).abs() #max = 0.1422

#for i in range(pcg2.shape[1]):
#    hc = pcg2[pcg2.columns[i]][pcg2[pcg2.columns[i]]>0.9]
#    if hc.shape[0] > 0:
#        print(pcg2.columns[i], hc.index)
'''
each group has some highly correlated (pc>0.9) feats inside,
but no such feats between any two groups
may drop some redundant feats later

g1: (cluster 1)
v105: v89, v54, v46, v63, v8, v25
v89:  v105, v54, v46, v63, v25
v54:  v105, v89, v46, v63, v25
v46:  v105, v89, v54, v63, v8, v25
v63:  v105, v89, v54, v46, v8, v25
v8:   v105, v46, v63, v25
v25:  v105, v89, v54, v46, v63, v8
['v105', 'v89', 'v54', 'v46', 'v63', 'v8', 'v25']

g1: (cluster 2)
v109: v128
v108: v128
v128: v109, v108
['v109', 'v108', 'v128']

g1: (pairs)
v81:  v5

g2: (cluster 1)
v121: v33, v83
v33:  v121, v111, v83, v55
v111: v33,  v83
v83:  v121, v33, v111, v55
v55:  v33, v83
['v121', 'v33', 'v111', 'v83', 'v55']

g2: (cluster 2)
v32:  v15, v73, v86
v15:  v73, v32
v73:  v15, v32
v86:  v32
['v32', 'v15', 'v73', 'v86']

g2: (cluster 3)
v43:  v116, v26
v116: v43
v26:  v43, v60
v60:  v26
['v43', 'v116', 'v26', 'v60']

g2: (cluster 4)
v29:  v41, v96, v67, v77
v41:  v29, v96, v67, v49
v96:  v29, v41
v67:  v29, v41, v77
v49:  v41
v77:  v29, v67
['v29', 'v41', 'v96', 'v67', 'v49', 'v77']

g2: (cluster 5)
v76:  v17, v64
v17:  v76, v64, v48
v64:  v76, v17, v106, v48
v106: v64, v48
v48:  v17, v106, v64
['v76', 'v17', 'v64', 'v106', 'v48']

g2: (pairs)
v69:  v115
v118: v97
v92:  v95
v20:  v65
v68:  v39
v58:  v100
v11:  v53
v13:  v104
v37:  v1

g4: (cluster 1)
v34:  v40, v114
v40:  v34, v114
v114: v34, v40

g4: (pairs)
v12:  v10
'''

#rank match for tree methods
#but it doesn't give extra information out of pearsonr 
rkg2 = pd.DataFrame(index=g2.columns, columns=g2.columns)
for i in range(g2.shape[1]):
    for j in range(g2.shape[1]):
        if i > j:
            rkg2.ix[i, j] = rkg2.ix[j, i]
        if i != j:        
            rkg2.ix[i, j] = rank_match(g2[g2.columns[i]], g2[g2.columns[j]])
    print(i)
    
    
'''*****************************************************************'''
'''********************output data information**********************'''
'''*****************************************************************'''
data_info = pd.Series()
data_info['group1'] = list(group1)
data_info['group2'] = list(group2)
data_info['group3'] = list(group3)
data_info['group4'] = list(group4)
data_info['groupd'] = list(groupd)
data_info['groupc'] = cat_feat
data_info = pd.DataFrame(data_info, columns=['info'])
data_info.to_csv('data_transform/data_info.csv')

''''''