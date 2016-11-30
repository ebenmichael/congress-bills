"""
Testing suite for different models. Tests IPM, LC-IPM, Logitic Regression, Random Forest
Author: Eli Ben-Michael
"""
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from models.idealpoint import IdealPointModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from models.VB import VB


def to_sparse_features(interactions):
    """Convert a matrix of (bill, user, vote) interactions into a sparse
       feature matrix of indicators for (bill, user) and a target vector of
       votes
    Args:
        interactions: ndarray, shape (n_interactions, 3), matrix of
                      (bill, user, vote)
    Returns:
        X: sparse.csr_matrix, shape (n_interactions, n_bills * n_users), sparse
           matrix of indicators
        y: ndarray, length n_interactions, target vector of votes
    """
    # one hot encoder does all the work
    one_hot = preprocessing.OneHotEncoder()
    X = one_hot.fit_transform(interactions[:, :2])
    y = interactions[:, 2]
    return(X, y)


def eval_models(models, interactions, n_folds=5):
    """Use K-Fold cross validation to get estimates of model performance
    Args:
        models: list, list of objects to train
        interactions: ndarray, shape (n_interactions, 3), matrix of
                      (bill, user, vote)
        n_folds: int, number of folds for K fold CV
    Returns:
        test_acc: list, classification accuracy of models on test data
        test_auc: list, AUC of models on test data
    """
    test_acc = np.zeros((n_folds, len(models)))
    test_auc = np.zeros((n_folds, len(models)))
    # iterate over the models
    kfold = model_selection.KFold(n_folds, shuffle=True)
    for k, (train_idxs, test_idxs) in enumerate(kfold.split(interactions)):
        for j, model in enumerate(models):
            # treat IPM differently because it has a different api
            if type(model) == IdealPointModel:
                vb = VB()
                # raun variational inference
                vb.run(model, interactions[train_idxs, :])
                probs = model.predict_proba(interactions[test_idxs, :])
            # if comparing to the all yes model predict 1 for everything
            elif model == "yes":
                probs = np.ones(len(test_idxs))
            else:
                X_sp, y = to_sparse_features(interactions)
                # fit the model
                model.fit(X_sp[train_idxs, :], y[train_idxs])
                # evaluate the model
                probs = model.predict_proba(X_sp[test_idxs, :])[:, 1]
            # test accuracy
            acc = np.mean((probs > 0.5) == interactions[test_idxs, 2])
            # test auc
            auc = metrics.roc_auc_score(interactions[test_idxs, 2], probs)
            test_acc[k, j] = acc
            test_auc[k, j] = auc
    return(test_acc, test_auc)
