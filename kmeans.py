from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score

def kmeans_clustering(X, n_clusters=2):
    
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_model.fit(X)
    cluster_labels = kmeans_model.labels_
    silhouette = silhouette_score(X, cluster_labels)

    #y_kmeans = kmeans_model.predict(X)

    ## Genauigkeitsberechnung
    #knnscore = accuracy_score(X,y_kmeans)
    

    return kmeans_model, silhouette
