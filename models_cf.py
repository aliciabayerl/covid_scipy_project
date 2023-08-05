## import libraries, functions from different files
import pandas as pd
import matplotlib.pyplot as plt
import os
from ScriptsModels import randomForest


current_dir = os.getcwd()
datasets_folder = 'Datasets'  

# Load data from the Datasets folder
data = pd.read_csv(os.path.join(datasets_folder, 'cf_cleaned_dataset.csv'))
## load cleaned casefile dataset
#data = pd.read_csv('cf_cleaned_dataset.csv')
# Separate the features and the target variables
X = data[["age_group", "sex", "exposure_yn", "underlying_conditions_yn"]]
y1 =data['hosp_yn']
y2 =data['icu_yn']
y3 =data['death_yn']

## Model with random forest
#here only RF is used, due to the fact, that RF performed best with the medical conditions dataset
# Train the RF for the different dependent variables and get the results
# for hospitalized people
rf1_model, feature1_importancesRF, rf1_fpr, rf1_tpr, rf1_auc = randomForest.random_forest(X, y1)
# for people in ICU
rf2_model, feature2_importancesRF, rf2_fpr, rf2_tpr, rf2_auc = randomForest.random_forest(X, y2)
# for people that died
rf3_model, feature3_importancesRF, rf3_fpr, rf3_tpr, rf3_auc = randomForest.random_forest(X, y3)

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
# Plot speichern

# Save the plot in the 'Plots' folder
plot_filename = os.path.join('Plots', 'casefile_RF_ROC_Curves.png')
plt.savefig(plot_filename)

# Calculate the feature importance for each predicted variable
# Feature importances RF for hospitalized people
feature1_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature1_importancesRF})
feature1_importancesRF = feature1_importancesRF.sort_values(by='Importance', ascending=False)
feature1_importances_filename = os.path.join('FeatureImportances', 'feature_importances_cf_hospitalized.csv')
feature1_importancesRF.to_csv(feature1_importances_filename, index=False)

# Feature importances RF for people in ICU
feature2_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature2_importancesRF})
feature2_importancesRF = feature2_importancesRF.sort_values(by='Importance', ascending=False)
feature2_importances_filename = os.path.join('FeatureImportances', 'feature_importances_cf_icu.csv')
feature2_importancesRF.to_csv(feature2_importances_filename, index=False)

# Feature importances RF for people that died
feature3_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature3_importancesRF})
feature3_importancesRF = feature3_importancesRF.sort_values(by='Importance', ascending=False)
feature3_importances_filename = os.path.join('FeatureImportances', 'feature_importances_cf_dead.csv')
feature3_importancesRF.to_csv(feature3_importances_filename, index=False)
