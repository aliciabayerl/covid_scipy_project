import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import graphviz
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

def decision_tree(X, y):

    # Perform label encoding for categorical variables
    #label_encoder = LabelEncoder()
    #for column in X.select_dtypes(include='object').columns:
    #    X[column] = label_encoder.fit_transform(X[column])

   # Apply SMOTE to oversample the minority class
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Train Decision Tree, fit the model on the resampled data, make predicition and evaluate
    decision_tree = DecisionTreeClassifier(max_depth=6)
    v_score = cross_val_score(decision_tree, X, y, cv=10)

    decision_tree.fit(X_resampled, y_resampled)

    y_pred = decision_tree.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    print("Accuracy DT:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report DT:")
    print(report)

    print("Cross-Validation Scores:")
    print(v_score)
    print("Mean CV Score:", v_score.mean())

    # Visualize the decision tree using graphviz
    dot_data = export_graphviz(decision_tree, out_file=None, 
                               feature_names=X.columns, class_names=['Not Deceased', 'Deceased'], 
                               filled=True, rounded=True, special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

    # Calculate the probability scores, false/true positive rate, thresholds, ROC Curve for the test set
    y_pred_proba = decision_tree.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    auc_score = roc_auc_score(y_test, y_pred_proba)

    return decision_tree, fpr, tpr, auc_score