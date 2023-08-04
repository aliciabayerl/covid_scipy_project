## import libraries, functions from different files
import pandas as pd
from logistic_regression import logistic_regression
from randomForest import random_forest
from decisionTree import decision_tree
from knn_classification import knn_classification
import matplotlib.pyplot as plt
from kmeans import kmeans_clustering

## load cleaned casefile dataset
data = pd.read_csv('cf_cleaned_dataset.csv')
# Separate the features and the target variables
X = data[["age_group", "sex", "exposure_yn", "underlying_conditions_yn"]]
y1 =data['hosp_yn']
y2 =data['icu_yn']
y3 =data['death_yn']

## Model with random forest
#here only RF is used, due to the fact, that RF performed best with the medical conditions dataset
# Train the RF for the different dependent variables and get the results
# for hospitalized people
rf1_model, feature1_importancesRF, rf1_fpr, rf1_tpr, rf1_auc = random_forest(X, y1)
# for people in ICU
rf2_model, feature2_importancesRF, rf2_fpr, rf2_tpr, rf2_auc = random_forest(X, y2)
# for people that died
rf3_model, feature3_importancesRF, rf3_fpr, rf3_tpr, rf3_auc = random_forest(X, y3)

## general predefinition for the whole plot
plt.figure(figsize=(14, 6))

# Plot the ROC curve for RF for hospitalized people
plt.subplot(1, 3, 1)
plt.plot(rf1_fpr, rf1_tpr, color='blue', lw=2, label='Random Forest(AUC = {:.2f})'.format(rf1_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest for hospitalized people - ROC Curve')
plt.legend(loc='lower right')

# Plot the ROC curve  for RF for people in ICU
plt.subplot(1, 3, 2)
plt.plot(rf2_fpr, rf2_tpr, color='blue', lw=2, label='Random Forest(AUC = {:.2f})'.format(rf2_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest  for people in ICU - ROC Curve')
plt.legend(loc='lower right')

# Plot the ROC curve for people that died
plt.subplot(1, 3, 3)
plt.plot(rf3_fpr, rf3_tpr, color='blue', lw=2, label='Random Forest (AUC = {:.2f})'.format(rf3_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest for people that died- ROC Curve')
plt.legend(loc='lower right')

## Plot the data
plt.tight_layout()
plt.show()
# Plot speichern
plt.savefig('casefile_RF_ROC_Curves')

## Calculate the featureimportance for each predicted variable
# Feature importances RF for hospitalized people
feature1_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature1_importancesRF})
feature1_importancesRF = feature1_importancesRF.sort_values(by='Importance', ascending=False)
feature1_importancesRF.to_csv('feature_importances_cf_hospitalized.csv', index=False)

# Feature importances RF for people in ICU
feature2_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature2_importancesRF})
feature2_importancesRF = feature2_importancesRF.sort_values(by='Importance', ascending=False)
feature2_importancesRF.to_csv('feature_importances_cf_icu.csv', index=False)

# Feature importances RRF for people that died
feature3_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature3_importancesRF})
feature3_importancesRF = feature3_importancesRF.sort_values(by='Importance', ascending=False)
feature3_importancesRF.to_csv('feature_importances_cf_dead.csv', index=False)
