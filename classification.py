from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP, UnsupervisedSpatialFilter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.decomposition import PCA
from constants import *

def create_grid():
# Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [80, 90, 100, 110, 120],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    return GridSearchCV(estimator = rf, param_grid = param_grid,
                              cv = 3, n_jobs = -1, verbose = 2)


def create_opt():
    opt = BayesSearchCV(
        RandomForestClassifier(),
        {
            'bootstrap': [True, False],
            'max_depth': Integer(60, 100),
            'max_features': Integer(1, 8),
            'min_samples_leaf': Integer(1, 8),
            'min_samples_split': Integer(2, 12),
            'n_estimators': Integer(50, 500)
        },
        n_iter=50,
        cv=4
    )
    return opt

def create_classifier(features, labels, clf_type = 'lda'):
    '''

    Parameters
    ----------
    features
    labels
    clf_type

    Returns
    -------

    '''
    lda = LinearDiscriminantAnalysis()
    rnf = RandomForestClassifier(n_estimators=20, max_depth=50)
    clf = Pipeline([('LDA', lda)])

    if clf_type == 'rnf':
        clf = Pipeline([('RNF', rnf)])

    scores = cross_val_score(clf, features, labels, cv=4)
    lda.fit(features, labels)
    return clf, scores

def create_CSP(epochs, components = 8, two_classes = False):
    '''

    Parameters
    ----------
    epochs

    Returns
    -------

    '''
    labels = epochs.events[:,2]
    epochs_data_train = epochs.get_data()[:, :, FS*READY_DUR:]
    cv = ShuffleSplit(4, test_size=0.2, random_state=42)
    # cv_split = cv.split(epochs_data_train)

    if two_classes: #delete right
        epochs_data_train = epochs_data_train[labels != 3]
        labels = labels[labels != 3]

    # Assemble a classifier
    csp = CSP(n_components=components, reg='ledoit_wolf', log=True, norm_trace=False, transform_into='average_power',
              cov_est='epoch')
    csp_features = Pipeline([('asd', UnsupervisedSpatialFilter(PCA(11), average=True)), ('asdd', csp)]).fit_transform(
        epochs_data_train, labels)

    return csp_features, labels