import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def knn_classification(X, y, k_neighbors=5):

    # Apply SMOTE to oversample the minority class
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)

    # Perform 5-fold cross-validation and get the scores
    cv_scores = cross_val_score(knn_classifier, X, y, cv=5)

    # Fit the model on the resampled data, make predictions, and evaluate
    knn_classifier.fit(X_resampled, y_resampled)

    y_pred = knn_classifier.predict(X_test)

    # Print scores
    print("Cross-Validation Scores:")
    print(cv_scores)
    print("Mean CV Score:", cv_scores.mean())

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy KNN:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report KNN:")
    print(report)

    # Calculate the probability scores, false/true positive rate, thresholds, ROC Curve for the test set
    y_pred_proba = knn_classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    auc_score = roc_auc_score(y_test, y_pred_proba)

    return knn_classifier, fpr, tpr, auc_score


# knn_model, fpr, tpr, auc_score = knn_classification(X, y, k_neighbors=5)
