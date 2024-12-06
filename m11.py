import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv("C:/Users/visha/Downloads/CAT 1 Breast cancer dataset - breast-cancer.csv")

# Data Preprocessing
data = data.dropna()

# Select numerical columns for clustering
numerical_cols = data.select_dtypes(include=[np.number]).columns
X = data[numerical_cols]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal number of clusters for KMeans
inertia = []
silhouette_scores = []
K_range = range(2, 11) 
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Plotting Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# Choose optimal k based on the elbow point or highest silhouette score
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  
print(f"Optimal number of clusters determined: {optimal_k}")

# Applying Agglomerative Clustering with the optimal number of clusters
hc = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_scaled)

# KMeans Clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Silhouette score for evaluation
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score for KMeans Clustering: {silhouette_avg}")

# Predefined user input
user_input = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.8,
    'area_mean': 1001.0,
    'smoothness_mean': 0.1184,
    'compactness_mean': 0.2776,
    'concavity_mean': 0.3001,
    'concave points_mean': 0.2019,
    'symmetry_mean': 0.1866,
    'fractal_dimension_mean': 0.05999,
    'radius_se': 1.095,
    'texture_se': 0.9053,
    'perimeter_se': 8.589,
    'area_se': 153.4,
    'smoothness_se': 0.006399,
    'compactness_se': 0.04006,
    'concavity_se': 0.05373,
    'concave points_se': 0.02056,
    'symmetry_se': 0.0225,
    'fractal_dimension_se': 0.004571,
    'radius_worst': 25.38,
    'texture_worst': 17.33,
    'perimeter_worst': 184.6,
    'area_worst': 2019.0,
    'smoothness_worst': 0.1622,
    'compactness_worst': 0.3454,
    'concavity_worst': 0.4268,
    'concave points_worst': 0.2634,
    'symmetry_worst': 0.2419,
    'fractal_dimension_worst': 0.07871
}

# Convert the input dictionary to a DataFrame and scale it
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Predict the cluster for the input using KMeans and Agglomerative Clustering
kmeans_cluster = kmeans.predict(input_scaled)[0]
hc_cluster = hc.fit_predict(np.vstack([X_scaled, input_scaled]))[-1]

print(f"KMeans: Input data belongs to cluster {kmeans_cluster}")
print(f"Agglomerative Clustering: Input data belongs to cluster {hc_cluster}")

# Visualization: Select two features to plot the clusters
feature_x = numerical_cols[0] 
feature_y = numerical_cols[1]  

# Plot the KMeans clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[feature_x], X[feature_y], c=y_kmeans, cmap='rainbow', s=50)
plt.scatter(input_df[feature_x], input_df[feature_y], color='black', marker='X', s=300, label='New Input')
plt.title('KMeans Clustering Visualization')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.legend()
plt.show()

# Plot the Hierarchical Clusters
plt.figure(figsize=(10, 7))
plt.scatter(X[feature_x], X[feature_y], c=y_hc, cmap='rainbow', s=50)
plt.scatter(input_df[feature_x], input_df[feature_y], color='black', marker='X', s=300, label='New Input')
plt.title('Agglomerative Clustering Visualization')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.legend()
plt.show()
