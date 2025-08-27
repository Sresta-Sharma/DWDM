import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Generate 1000 random 2D points
data = np.random.rand(1000, 2) * 200

# KMeans++ Initialization
km = KMeans(n_clusters=4, init="k-means++")

start = time.process_time()
km.fit(data)
end = time.process_time()

print("Time Taken:", end - start)
print("Centers:\n", km.cluster_centers_)

# Plot
colors = ["r", "g", "b", "y"]
labels = km.labels_

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[labels[i]], marker='.')
    
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='x', s=100, color='black')
plt.title("K-Means++ Clustering")
plt.savefig("kmeans_plus_plus_lab6.png")
plt.show()
