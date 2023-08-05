from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score

# Function for logistic regression
def logistic_regression(X, y):

    # Apply SMOTE to oversample the minority class (due to the fact that the first dataset hasonly 20 people that die and we want to lay mor emphasis on the reasons why people die)
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Perform Logistic regression, fit the model on the resampled data, make predicition and evaluate
    model = LogisticRegression()

    # additionally calculate a cross validation score (it calculates accuracy without the need to split data in test & train samples and is therefore very handy when the sample size is not that big)
    v_score = cross_val_score(model, X, y, cv=10)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    # Print the model accuracys for evaluating   
    print("Cross-Validation Scores:")
    print(v_score)
    print("Mean CV Score:", v_score.mean())
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy LR:", accuracy)
    report = classification_report(y_test, y_pred)
    print("Classification Report LR:")
    print(report)

    # Calculate further measures to make the available for later plotting of the ROC-Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    return model, fpr, tpr, auc_score
