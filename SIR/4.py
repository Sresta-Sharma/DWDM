import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Given data points
points = np.array([(2,10),(2,5),(8,4),(5,8),(7,5),(6,4),(1,2),(4,9)])

# Apply KMeans clustering
km = KMeans(n_clusters=3, init='random')
km.fit(points)

print("Cluster Centers:\n", km.cluster_centers_)

# Plot
colors = ["r", "g", "b"]
labels = km.labels_
for i in range(len(points)):
    plt.scatter(points[i][0], points[i][1], color=colors[labels[i]])

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='x', s=100)
plt.title("K-Means on Given Points")
plt.savefig("kmeans_lab4.png")
plt.show()
