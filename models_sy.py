import pandas as pd
from ScriptsModels import randomForest
import matplotlib.pyplot as plt
import os

## load cleaned symptoms dataset

current_dir = os.getcwd()
datasets_folder = 'Datasets'  

data = pd.read_csv(os.path.join(datasets_folder, 'mc_cleaned_dataset.csv'))

# Separate the features + the target variable
X = data.drop("deceased", axis=1)
y = data["deceased"]

## Model with random forest
# here only RF is used, due to the fact, that RF performed best with the medical conditions dataset
# Train the models and get the results
rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = randomForest.random_forest(X, y)

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
roc_plot_filename = os.path.join('Plots', 'sy_RF_ROC_Curves.png')
plt.savefig(roc_plot_filename)

## Calculate Feature importances RF
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)

# Save the feature importances CSV in the 'FeatureImportances' folder
feature_importances_filename = os.path.join('FeatureImportances', 'feature_importances_sy.csv')
feature_importancesRF.to_csv(feature_importances_filename, index=False)