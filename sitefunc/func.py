import pandas as pd
import numpy as np

'''encode categorical features to one-hot features'''
def to_onehot(X, unique_values=500):
    newX = pd.DataFrame(index=X.index)
    cols = X.shape[1]
    for i in range(cols):
        value_types = len(set(X.ix[:, i]))
        if((value_types>2)&(value_types<=unique_values)):
            temp_onehot = pd.get_dummies(X.ix[:, i], prefix=X.columns[i], prefix_sep='__', dummy_na=True)
            nan_col = temp_onehot.columns[temp_onehot.columns.str.endswith('nan')]
            if temp_onehot[nan_col].sum().sum() == 0:
                temp_onehot.drop(nan_col, axis=1, inplace=True)
            newX[temp_onehot.columns] = temp_onehot
        else:
            newX[X.columns[i]] = X.ix[:, i]
    return newX

'''make the contingency of two categorical features'''
def make_contingency(col1, col2, unique_values=500):
    if (col1.dtype != object) | (col2.dtype != object):
        print("Column type must be object!")
        return
    #trun null values to string 'NaN'
    c1 = col1.copy()
    c2 = col2.copy()
    c1[c1.isnull()] = 'NaN'
    c2[c2.isnull()] = 'NaN'
    uni_val1 = list(set(c1))
    uni_val2 = list(set(c2))
    #avoid too many different values
    if (len(uni_val1) > unique_values) | (len(uni_val2) > unique_values):
        print("Too many values !")
        return pd.DataFrame()
    contingency = pd.DataFrame(index=uni_val1, columns=uni_val2)
    value_count = (c1 + '__' + c2).value_counts()
    for i in range(len(uni_val1)):
        for j in range(len(uni_val2)):
            key_str = uni_val1[i] + '__' + uni_val2[j]
            if key_str in value_count.index:
                contingency.ix[i, j] = value_count[key_str]
            else:
                contingency.ix[i, j] = 0
    return contingency.astype(int)
    
'''test whether two categorical features can match or not by contingency'''
def contingency_match(col1, col2):
    contingency = make_contingency(col1, col2)
    #row match
    #large row_match: each row value can respectively approximately match one col value
    row_sum = contingency.sum(axis=1) + 1 #"+1" for avoiding deviding zero
    row_max = contingency.max(axis=1) + 1
    row_match = (row_max / row_sum).prod()
    '''row_match = (row_max / row_sum).prod() ** (1/len(row_max))'''
    #column match
    col_sum = contingency.sum(axis=0) + 1
    col_max = contingency.max(axis=0) + 1
    col_match = (col_max / col_sum).prod()
    return row_match, col_match
    
def onehot_match(col1, col2, thresold=0.9):
    oh1 = to_onehot(pd.DataFrame(col1, columns=[col1.name]))
    oh2 = to_onehot(pd.DataFrame(col2, columns=[col2.name]))
    has_match = False
    match_set = []
    for i in range(oh1.shape[1]):
        for j in range(oh2.shape[1]):
            match_index = (oh1.ix[:, i]==1) | (oh2.ix[:, j]==1)
            match_rate = (oh1.ix[match_index, i] == oh2.ix[match_index, j]).mean()
            if match_rate > thresold:
                has_match = True
                match_set.append((oh1.columns[i], oh2.columns[j], match_rate))
    return has_match, match_set
'''compute pearsonr by using scipy.stats.pearsonr'''
'''support vector-vector, matrix-vector, matrix-matrix'''
def compute_pearsonr(x, y=None):
    from scipy.stats import pearsonr
    def compute_pearson_columns(col1, col2):
        if (col1.dtype==np.object) | (col2.dtype==np.object):
            corr = np.nan
        else:
            index = ((col1.notnull()) & (col2.notnull()))
            corr = pearsonr(col1[index], col2[index])[0]
        return corr    
    def compute_pearson_mixture(mat, col):
        pcorr = pd.Series(index=mat.columns)
        for i in range(mat.shape[1]):
            pcorr[i] = compute_pearson_columns(mat.ix[:, i], y)
        return pcorr
    def compute_pearson_mixture_face(x, y):
        if (len(x.shape) == 1) & (len(y.shape) == 2):
            return compute_pearson_mixture(y, x)
        elif (len(x.shape) == 2) & (len(y.shape) == 1):
            return compute_pearson_mixture(x, y)
    def compute_pearson_matrix(mat1, mat2=None, verbose=True):
        if mat2 is None:
            mat2 = mat1       
        len1 = mat1.shape[1]
        len2 = mat2.shape[1]
        pcorr_matrix = pd.DataFrame(index=mat1.columns, columns=mat2.columns)
        for i in range(len1):
            for j in range(len2):                
                if (mat2 is None) & (i > j):
                    pcorr_matrix.ix[i, j] = pcorr_matrix.ix[j, i]  
                elif i != j:
                    pcorr_matrix.ix[i, j] = compute_pearson_columns(mat1.ix[:, i], mat2.ix[:, j])
            if verbose == True:
                print("loop %d done!" % i)
        return pcorr_matrix
    #compute pearson
    if y is None:
        y = x
    if (len(x.shape) == 1) & (len(y.shape) == 1):
        return compute_pearson_columns(x, y)
    elif (len(x.shape) == 2) & (len(y.shape) == 2):
        return compute_pearson_matrix(x, y)
    else:
        return compute_pearson_mixture_face(x, y)

'''match numeric features using value ranking'''   
def rank_match(col1, col2):
    notnull_index = (col1.notnull()) & (col2.notnull())
    rank1 = col1[notnull_index].rank()
    rank2 = col2[notnull_index].rank()
    rank_match_rate = compute_pearsonr(rank1, rank2)
    return rank_match_rate
    
'''
def scaled_distance(col):
    sorted_col = col.dropna().sort_values().round(8)
    rolling_distance = sorted_col.rolling(window=2).apply(lambda x: x[1]-x[0])
    rolling_distance = rolling_distance[rolling_distance>0.000003]
    return round(rolling_distance.mode().values[0], 8)
''' 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
