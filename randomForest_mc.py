import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('modified_dataset.csv')

# Separate the target variable (deceased) from the features
X = data.drop(columns=['deceased'])
y = data['deceased']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Introducing weights for better results bcs deceased so small
class_weights = {0: 1, 1: 5}
random_forest = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions 
y_pred = random_forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report with zero devision = 1 because deceased number so small :( !! Consider in results)
#print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

# Feature Importances for trained model:
feature_importances = random_forest.feature_importances_

importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the features in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)
# Result highest: low oxygen, age, diabetes, hospitalized, bmi....
# --> good at predicting non-deceased cases..
# Further steps ??: Reduce number of non-deceased to balance or assign higher weights to deceased cases


#  Before Assigning Class Weights:

# Accuracy: 0.98
# Precision (class 1): 0.0
# Recall (class 1): 0.0
# F1-score (class 1): 0.0

# After Assigning Class Weights:

# Accuracy: 0.96
# Precision (class 1): 0.0
# Recall (class 1): 0.0
# F1-score (class 1): 0.0

# --> The accuracy has slightly decreased from 0.98 to 0.96 after using class weights
# --> even after using class weights, the model still fails to predict any positive instances (deceased) correctly