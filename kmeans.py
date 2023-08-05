## import of the required libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

# definition of kmeans as a function to make it accessible from other files
def kmeans_clustering(X, y, n_clusters=2):
    # Apply SMOTE to oversample the minority class (used because the first dataset showed only 20 people that died and we wanted to lay more emphasis on this minor group)
    smote = SMOTE(sampling_strategy='auto')
    # smote gives data that oversamples the minor class
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # build the kmeans model and fit it to the data
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_model.fit(X)
    cluster_labels = kmeans_model.labels_
    # Compute silhouette score as the performance metric
    silhouette = silhouette_score(X, cluster_labels)
    return kmeans_model, silhouette
