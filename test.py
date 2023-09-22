# %%
import matplotlib.pyplot as plt
from functions.kmeans import KMeans
from functions.data_by_country import data_by_country
from utils.data_loader import load_data
import sklearn.cluster as sk
import numpy as np

# %%
# Load your dataset
df = load_data()
df = data_by_country(df)
df = df.pivot(index="dt", columns="Country",
              values='AverageTemperature').dropna()
data = df.loc["2013-01-01"] - df.loc["1889-01-01"]
data = data.sort_values()

# %%

kmeansSK = sk.KMeans(n_clusters=4)
resultSK = kmeansSK.fit(np.array(data).reshape(-1, 1))

# %%
plt.figure(figsize=(10, 6))
for i in range(resultSK.n_clusters):
    plt.scatter(np.array(data)[resultSK.labels_ == i], [
                1]*sum(resultSK.labels_ == i), label=f'Cluster {i+1}')
plt.scatter(resultSK.cluster_centers_, [
            1]*len(resultSK.cluster_centers_), c='red', marker='X', s=100, label='Centroids')
plt.yticks([])
plt.title('KMeans Clustering Results')
plt.xlabel('Data Points')
plt.legend()
plt.show()

# %%
kmeans = KMeans()
result = kmeans.fit(data)
print(result)

# %%

plt.figure(figsize=(10, 1))
plt.scatter(result[0], [1]*len(result[0]), marker='o')
plt.scatter(result[1], [1]*len(result[1]), marker='o')
plt.scatter(result[2], [1]*len(result[2]), marker='o')
plt.scatter(kmeans.centroids, [1]*3, marker='x')
plt.yticks([])  # Hide y-axis ticks
plt.title('Distribution of Temperatures')
plt.xlabel('Temperature')
plt.show()

print("test")

# %%
