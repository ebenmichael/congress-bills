"""
Testing suite for different models. Tests IPM, LC-IPM, Logitic Regression
Author: Eli Ben-Michael
"""
import numpy as np
# from sklearn.model_selection import KFold
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from lcipm.models.idealpoint import IdealPointModel
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from lcipm.models.VB import VB
from lcipm.models.lcipm import LCIPM
import os
import sys


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


def eval_models(models, interactions, membership=None, n_folds=5):
    """Use K-Fold cross validation to get estimates of model performance
    Args:
        models: list, list of objects to train
        interactions: ndarray, shape (n_interactions, 3), matrix of
                      (bill, user, vote)
        membership: ndarray, shape (n_users, n_users), caucus co membership
                    data for LCIPM
        n_folds: int, number of folds for K fold CV
    Returns:
        test_acc: ndarray, shape n_models, n_folds, classification
        accuracy of models on test data
        test_auc: list, AUC of models on test data
    """
    test_acc = np.zeros((len(models), n_folds))
    test_auc = np.zeros((len(models), n_folds))
    # iterate over the models
    kfold = KFold(interactions.shape[0], n_folds, shuffle=True)
    for k, (train_idxs, test_idxs) in enumerate(kfold):
        for j, model in enumerate(models):
            print("Fold: " + str(k) + " Model: " + str(type(model)))
            # treat IPM differently because it has a different api
            if type(model) == IdealPointModel:
                vb = VB(maxLaps=50)
                # run variational inference
                vb.run(model, interactions[train_idxs, :])
                probs = model.predict_proba(interactions[test_idxs, :2])
            elif type(model) == LCIPM:
                vb = VB(maxLaps=50)
                # run variational inference
                vb.run(model, (interactions[train_idxs, :], membership))
                probs = model.predict_proba(interactions[test_idxs, :2])
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
            test_acc[j, k] = acc
            test_auc[j, k] = auc
        # save after each fold just in case
        np.savetxt("test_acc.dat", test_acc)
        np.savetxt("test_auc.dat", test_auc)
    return(test_acc, test_auc)


def eval_and_save(models, interactions, membership=None, n_folds=5,
                  outdir=None):
    """Use K-Fold cross validation to get estimates of model performance
       and save results
    Args:
        models: list, list of objects to train
        interactions: ndarray, shape (n_interactions, 3), matrix of
                      (bill, user, vote)
        membership: ndarray, shape (n_users, n_users), caucus co membership
                    data for LCIPM
        n_folds: int, number of folds for K fold CV
        outdir: string, directory to save to, defaults to current directory
    """
    # evaluate the models
    test_acc, test_auc = eval_models(models, interactions, membership,
                                     n_folds)

    if outdir is None:
        outdir = os.getcwd()

    # save the results
    np.savetxt(os.path.join(outdir, "test_acc.dat"), test_acc)
    np.savetxt(os.path.join(outdir, "test_auc.dat"), test_auc)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: modeleval.py outdir data/combined_data/interactions.dat"
              + "data/combined_data/membership.dat")
    else:
        outdir = sys.argv[1]
        interactions = np.loadtxt(sys.argv[2])
        caucus = np.loadtxt(sys.argv[3])
        caucus = np.dot(caucus, caucus.T)
        models = [LogisticRegression(C=1), LogisticRegression(C=0.1),
                  LogisticRegression(C=0.01)]
        """, IdealPointModel(1),
                  IdealPointModel(2)]
        for k in [1, 2, 3, 4, 10]:
            for s in [1, 2]:
                models.append(LCIPM(448, s, k))

        for sigma in [0.01, 0.1, 1, 100]:
            models.append(LCIPM(448, 2, 2, ip_prior_var=sigma))
        """
        eval_and_save(models, interactions, membership=caucus, outdir=outdir)
