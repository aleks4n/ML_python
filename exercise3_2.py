from sklearn.datasets import load_iris
import numpy as np

# Load the data 
data = load_iris()

# Print the data
X = data.data

# K-means clusttering with K=3


# Generate 3 random indices from 0 to n_samples
indices = np.random.choice(X.shape[0], size=3, replace=False)

# Select the points
points = X[indices]

# Assing the points to the clusters
clusters = {i: [] for i in range(3)}

# Plot the clusters

import matplotlib.pyplot as plt

#Plot the SSE against the number of iterations
sse = []

for _ in range(100):
    # Assing the points to the clusters
    clusters = {i: [] for i in range(3)}

    # Assign each point to the closest cluster
    for point in X:
        # Compute the distances
        distances = [np.linalg.norm(point - c) for c in points]
        # Get the closest cluster
        cluster = np.argmin(distances)
        # Add the point to the cluster
        clusters[cluster].append(point)

    # Update the clusters
    for i in range(3):
        points[i] = np.mean(clusters[i], axis=0)

    # Compute the SSE
    s = 0
    for i in range(3):
        s += np.sum((clusters[i] - points[i]) ** 2)
    sse.append(s)

#Print the best Centroids
print('Best Centeroids')
print(points)

# Print best labels
print('Best Labels')
for i in range(3):
    print(f'Cluster {i}: {len(clusters[i])} points')

print('Sum of Squared Errors')
print(sse[-1])





plt.figure()  # Create a new figure
for i in range(3):
    cluster = np.array(clusters[i])
    plt.scatter(cluster[:, 0], cluster[:, 1])
plt.scatter(points[:, 0], points[:, 1], c='black', s=100)
plt.title('K-means clustering')


# Plot the SSE
plt.figure()  # Create another new figure
plt.plot(sse)
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.title('SSE over iterations')


plt.show()