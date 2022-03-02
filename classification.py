from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
import numpy as np

def create_classifier(features, labels):
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([('LDA', lda)])
    scores = cross_val_score(clf, features, labels, cv=4)
    lda.fit(features, labels)
    return lda, scores

def create_CSP(epochs):
    scores = []
    labels = epochs.events[:,2]
    epochs_data_train = epochs.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data_train, labels)
    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)