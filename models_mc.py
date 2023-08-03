import pandas as pd
from logistic_regression import logistic_regression
from randomForest import random_forest
from decisionTree import decision_tree
from knn_classification import knn_classification
import matplotlib.pyplot as plt
from kmeans import kmeans_clustering


data = pd.read_csv('mc_cleaned_dataset.csv')
# Separate the features + the target variable
X = data.drop("deceased", axis=1)
y = data["deceased"]

# Train the models and get the results
rf_model, feature_importancesRF, rf_fpr, rf_tpr, rf_auc = random_forest(X, y)
lr_model, lr_fpr, lr_tpr, lr_auc = logistic_regression(X, y)
dt_model, dt_fpr, dt_tpr, dt_auc = decision_tree(X, y)
knn_model, kn_fpr, kn_tpr, kn_auc = knn_classification(X, y, k_neighbors=5)

# K-means clustering
kmeans_configs = [2, 3, 4, 5]  # Different number of clusters to try
silhouette_scores = []
for n_clusters in kmeans_configs:
    _, silhouette = kmeans_clustering(X, y, n_clusters=n_clusters)
    silhouette_scores.append(silhouette)

plt.figure(figsize=(14, 6))

# Plot the ROC curve for Random Forest
plt.subplot(2, 3, 1)
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label='Random Forest (AUC = {:.2f})'.format(rf_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest - ROC Curve')
plt.legend(loc='lower right')

# Plot the ROC curve for Logistic Regression
plt.subplot(2, 3, 2)
plt.plot(lr_fpr, lr_tpr, color='blue', lw=2, label='Logistic Regression (AUC = {:.2f})'.format(lr_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression - ROC Curve')
plt.legend(loc='lower right')

# Plot the ROC curve for Decision Tree
plt.subplot(2, 3, 3)
plt.plot(dt_fpr, dt_tpr, color='blue', lw=2, label='Decision Tree (AUC = {:.2f})'.format(dt_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree - ROC Curve')
plt.legend(loc='lower right')

# Plot the ROC curve for KNN
plt.subplot(2, 3, 4)
plt.plot(kn_fpr, kn_tpr, color='blue', lw=2, label='K-nearest-neighbors (AUC = {:.2f})'.format(kn_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-nearest-neighbors - ROC Curve') 
plt.legend(loc='lower right')

# Plot kmeans silhouette score
plt.subplot(2, 3, 5)
plt.plot(kmeans_configs, silhouette_scores, color='blue', lw=2, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('K-means - Silhouette Score')
plt.xticks(kmeans_configs)
plt.grid()


plt.tight_layout()
plt.show()

# Feature importances Random Forest
feature_importancesRF = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importancesRF})
feature_importancesRF = feature_importancesRF.sort_values(by='Importance', ascending=False)

feature_importancesRF.to_csv('feature_importances.csv', index=False)