# Feature Selector

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, RFE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn import metrics

def top_corr(x,y,n):
    """
    x Dataframe
    y Series or array

    Returns series of ftr, score for top n most correlated features
    Score is calculated the sum of abs correlations for all classes
    """

    class_dummies = pd.get_dummies(y).add_prefix('class_')
    corrs = []
    for col in x.columns:
        corrs.append(class_dummies.apply(lambda c: c.corr(x[col])).abs().sum())

    ret = pd.DataFrame([corrs],columns=x.columns).T.squeeze().nlargest(n)
    # standardize
    ret = ret = (ret-ret.min())/(ret.max()-ret.min())
    ret.name = 'corrs'

    return ret


def top_chi2(x, y, n):
    """
    x Dataframe
    y Series/Array -- class labels
    n Int
    """

    features = x.columns

    # all features must be positive
    x_norm = MinMaxScaler().fit_transform(x)

    selector = SelectKBest(chi2, k=n)
    selector.fit(x_norm, y)
    # bool index on selected columns
    selected = selector.get_support()

    chi2_scores = pd.DataFrame(list(zip(features, selector.scores_)), columns=['ftr', 'chi2_score'])
    chi2_ftrs = chi2_scores.loc[selected]

    ret = chi2_ftrs.sort_values('chi2_score', ascending=False).head(n).set_index('ftr').squeeze()
    ret = (ret-ret.min())/(ret.max()-ret.min())
    return ret

def top_rfe(mod, x, y, n, step=0.05, **params):
    selector = RFE(mod(**params), n, step, 1)
    selector.fit(x, y)
    selected = selector.get_support()

    rfe_ftrs = np.asarray(x.columns)[selected]
    rfe_ftrs = pd.Series(1, index = rfe_ftrs)
    return rfe_ftrs


def top_lasso(x,y,n,step=0.1, verbose=0):
    xscaled = MinMaxScaler().fit_transform(x.values)
    C = n/xscaled.shape[1]

    direction='down'
    num_non_zero = xscaled.shape[1]
    while num_non_zero != N:
        if verbose:
            print("Fitting Lasso with C =",C)
        l = LogisticRegression(penalty='l1', C=C)
        l.fit(xscaled, y)
        mask = l.coef_.mean(0)!=0
        num_non_zero = mask.sum()
        if verbose:
            print('Num Non-Zero Features:', num_non_zero)
        if num_non_zero == N:
            ret = pd.Series(np.abs(l.coef_).mean(0)[mask], index = x.columns[mask])
            ret.name='l1'
            ret = (ret-ret.min())/(ret.max()-ret.min())
            return ret
        elif num_non_zero > N:
            C *= 1-step
            new_direction = 'down'
        else:
            C *= 1+step
            new_direction = 'up'

        # if we change direction (overshot) lower step
        if new_direction != direction:
            step*=0.5
            direction=new_direction
            if verbose:
                print('New Step:', step)

def run_ftr_selection(X, Y, n, rf_params, gb_params):

    top_corrs = top_corr(X, Y, n)
    topchi2 = top_chi2(X, Y, n)

    print('Fitting LogReg')
    rfe_lreg_ftrs = top_rfe(LogisticRegression, X, Y, n, 0.1)
    print()
    print('Fitting RF')
    rfe_rf_ftrs = top_rfe(RandomForestClassifier, X, Y, n, 0.1, **rf_params)
    print()
    print('Fitting GB')
    rfe_gb_ftrs = top_rfe(GradientBoostingClassifier, X, Y, n, 0.1, **gb_params)

    l1_ftrs = top_lasso(X, Y, n, 0.5, verbose=1)

    all_scores = [top_corrs, topchi2, rfe_lreg_ftrs, rfe_rf_ftrs, rfe_gb_ftrs, l1_ftrs]


    all_scores = pd.concat(all_scores, axis=1)
    all_scores.columns = ['corrs','chi2', 'rfe_lreg', 'rfe_rf', 'rfe_gb', 'l1']

    return all_scores.sum(1).nlargest(N)
