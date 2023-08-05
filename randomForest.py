## import of the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

## definition of RF as a Funktion to make it  accessible from other scripts
def random_forest(X, y):
    # Apply SMOTE to oversample the minority class (due to the fact that we only have 20 deceased people in one of our datasets wewant to make the minor class more important for feature importance
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Split the data into training and testing sets (now we have enogh test and trainingssamples due to smote)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    # build the RF-estimator 
    random_forest = RandomForestClassifier(n_estimators=100)
    # using cross validation instead of splitting the data
    v_score = cross_val_score(random_forest, X, y, cv=5)
    # Fit the model on the resampled data, make predicition and evaluate by printing the Validation scores
    random_forest.fit(X_resampled, y_resampled)
    y_pred = random_forest.predict(X_test)
    print("Cross-Validation Scores:")
    print(v_score)
    print("Mean CV Score:", v_score.mean())
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy RF:", accuracy)
    report = classification_report(y_test, y_pred)
    print("Classification Report RF:")
    print(report)

    # Calculate the probability scores, false/true positive rate, thresholds, ROC Curve for the test set
    y_pred_proba = random_forest.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    # we need to return this data to use it to plot the ROC curves
    return random_forest, random_forest.feature_importances_, fpr, tpr, auc_score
