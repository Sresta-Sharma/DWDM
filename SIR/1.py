import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate 1000 random 2D points between 0 and 200
data = np.random.rand(1000, 2) * 200

# Apply KMeans clustering with 4 clusters
km = KMeans(n_clusters=4, init="random")

start_time = time.process_time()
km.fit(data)
end_time = time.process_time()

print("Time Taken:", end_time - start_time)
print("Cluster Centers:\n", km.cluster_centers_)

# Plotting
colors = ["r", "g", "b", "y"]
markers = ["+", "x", "*", "d"]
labels = km.labels_

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], color=colors[labels[i]], marker=markers[labels[i] % 4])
    
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, linewidths=3, marker='o')
plt.title("K-Means Clustering (4 Clusters)")
plt.savefig("kmeans_lab1.png")
plt.show()
