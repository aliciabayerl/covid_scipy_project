import pandas as pd
#from logistic_regression import logistic_regression
from randomForest import random_forest
#from decisionTree_mc import decision_tree
#from knn_classification import knn_classification
import matplotlib.pyplot as plt
#from kmeans import kmeans_clustering


data = pd.read_csv('casefile_breinigt.csv')
# Separate the features + the target variable
X = data[["age_group", "sex", "exposure_yn", "underlying_conditions_yn"]]
y1 =data['hosp_yn']
y2 =data['icu_yn']
y3 =data['death_yn']

# Train the models and get the results
rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = random_forest(X, y1)

plt.figure(figsize=(14, 6))

# Plot the ROC curve for Random Forest
plt.subplot(1, 4, 1)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest(AUC = {:.2f})'.format(rf_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest for hospitalized people - ROC Curve')
plt.legend(loc='lower right')

# Feature importances Random Forest
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)

feature_importancesRF.to_csv('feature_importances_casefile_hospitalized.csv', index=False)


rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = random_forest(X, y2)
# Plot the ROC curve for Random Forest
plt.subplot(1, 4, 2)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest(AUC = {:.2f})'.format(rf_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest  for people in ICU - ROC Curve')
plt.legend(loc='lower right')

# Feature importances Random Forest
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)

feature_importancesRF.to_csv('feature_importances_casefile_icu.csv', index=False)



rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = random_forest(X, y3)
# Plot the ROC curve for Random Forest
plt.subplot(1, 4, 3)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest (AUC = {:.2f})'.format(rf_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest for people that died- ROC Curve')
plt.legend(loc='lower right')


plt.tight_layout()
plt.show()

# Feature importances Random Forest
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)

feature_importancesRF.to_csv('feature_importances_casefile_dead.csv', index=False)
