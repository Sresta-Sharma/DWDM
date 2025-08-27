import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Given points
x = [2, 2, 8, 5, 7, 6]
y = [10, 5, 4, 8, 5, 4]
data = list(zip(x, y))

# Agglomerative clustering using Ward linkage
linked = linkage(data, method='ward')

# Dendrogram
plt.figure(figsize=(8, 4))
dendrogram(linked, labels=["P1", "P2", "P3", "P4", "P5", "P6"])
plt.title("Dendrogram - Agglomerative Clustering")
plt.savefig("dendrogram_lab2.jpg")
plt.show()
