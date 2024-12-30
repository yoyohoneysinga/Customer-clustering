# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset (Assuming a CSV file)
try:
    data = pd.read_csv('Customer clustering/data.csv')
    print("Data loaded successfully")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

# Preview the dataset
print(data.head())

# Check for missing values in the relevant columns
print("Checking for missing values in relevant columns...")
print(data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]].isnull().sum())

# Select only the relevant features
features = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

# Handle missing values (if any) BEFORE scaling
features = features.fillna(features.mean())  # Replace NaN with the mean of each column

# Check if there are any NaN values after filling
if features.isnull().sum().any():
    print("There are still NaN values in the features after filling with the mean.")
else:
    print("No NaN values in the features after filling with the mean.")

# Feature Scaling (Standardization)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Check for NaN values in scaled features
if np.any(np.isnan(scaled_features)):
    print("There are NaN values in the scaled features!")
else:
    print("Scaled features do not contain NaN values.")

# Elbow Method to find the optimal number of clusters
inertia = []
sil_scores = []
range_n_clusters = range(2, 11)  # Try different number of clusters (e.g., 2 to 10)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)  # Increase n_init for better convergence
    kmeans.fit(scaled_features)
    
    # Inertia (Sum of squared distances to closest centroid)
    inertia.append(kmeans.inertia_)
    
    # Silhouette Score
    sil_score = silhouette_score(scaled_features, kmeans.labels_)
    sil_scores.append(sil_score)

# Plot the Elbow Method (Inertia)
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Plot the Silhouette Score for different K values
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, sil_scores, marker='o', linestyle='-', color='r')
plt.title('Silhouette Score for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Based on the plots, choose the optimal number of clusters, e.g., K = 4 (Example)
optimal_k = 4

# Apply KMeans with the optimal K value
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(scaled_features)

# Add the cluster labels to the original dataset
data['Cluster'] = clusters

# Show the results with cluster assignments
print(data.head())

# Visualize the cluster
