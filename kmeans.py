from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE


def kmeans_clustering(X, y, n_clusters=2):

    # Apply SMOTE to oversample the minority class
    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_model.fit(X)
    cluster_labels = kmeans_model.labels_
    
    # Compute silhouette score as the performance metric
    silhouette = silhouette_score(X, cluster_labels)
    
    return kmeans_model, silhouette