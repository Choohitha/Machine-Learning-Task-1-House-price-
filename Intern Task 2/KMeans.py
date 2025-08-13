
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Check dataset exists
file = "C:\\Users\\Choohitha\\Downloads\\Mall_Customers (1).csv"
if not os.path.isfile(file):
    print(f"Error: {file} not found. Please download and place it in this folder.")
    exit()

# 2. Load the dataset
df = pd.read_csv(file)
print("First 5 rows:")
print(df.head())

# 3. Select clustering features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 4. Elbow method to select k
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()
# 5. Apply K-Means with optimal k (e.g., 5)
k = 5  # choose based on your elbow plot
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X)
# 6. Plot clusters
plt.figure(figsize=(8, 6))
for cluster in range(k):
    plt.scatter(X[labels == cluster, 0], X[labels == cluster, 1], s=100, label=f'Cluster {cluster+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', marker='X', label='Centroids')
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.legend()
plt.show()
