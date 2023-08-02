from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score


def logistic_regression(X, y):

    # Apply SMOTE to oversample the minority class
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Perform Logistic regression, fit the model on the resampled data, make predicition and evaluate
    model = LogisticRegression()
    v_score = cross_val_score(model, X, y, cv=10)

    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)

        
    print("Cross-Validation Scores:")
    print(v_score)
    print("Mean CV Score:", v_score.mean())

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy LR:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report LR:")
    print(report)

    # Calculate the probability scores, false/true positive rate, thresholds, ROC Curve for the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    auc_score = roc_auc_score(y_test, y_pred_proba)

    return model, fpr, tpr, auc_score



#data = pd.read_csv('MC_cleaned_dataset.csv')
# Separate the features (X) and the target variable (y)
#X = data.drop("deceased", axis=1)
#y = data["deceased"]

#logistic_regression(X, y)


