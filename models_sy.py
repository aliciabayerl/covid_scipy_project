##import of required libraries
import pandas as pd
from logistic_regression import logistic_regression
from randomForest import random_forest
from decisionTree import decision_tree
from knn_classification import knn_classification
import matplotlib.pyplot as plt
from kmeans import kmeans_clustering

## load cleaned symptoms dataset
data = pd.read_csv('sy_cleaned_dataset.csv')
# Separate the features + the target variable
X = data.drop("deceased", axis=1)
y = data["deceased"]

## Model with random forest
# here only RF is used, due to the fact, that RF performed best with the medical conditions dataset
# Train the models and get the results
rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = random_forest(X, y)

# general plotting definitions
plt.figure(figsize=(7, 6))

# Plot the ROC curve for RF
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest (AUC = {:.2f})'.format(rf_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest - ROC Curve')
plt.legend(loc='lower right')

## Plot the data
plt.tight_layout()
## save Plot
plt.savefig('symptoms_RF_ROC_Curves')

## Calculate Feature importances RF
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)
## Export feature importance
feature_importancesRF.to_csv('feature_importances2.csv', index=False)