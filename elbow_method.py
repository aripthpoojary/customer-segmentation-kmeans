import matplotlib
matplotlib.use('Agg')  # disable GUI

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dataset
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 20, 30, 32, 34, 36,
                      50, 52, 54, 56, 58, 60, 70, 72, 74, 76],
    'Spending_Score': [39, 40, 42, 43, 44, 45, 60, 62, 64, 65,
                       20, 22, 24, 25, 26, 27, 80, 82, 84, 85]
}

df = pd.DataFrame(data)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Elbow method
wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot and save
plt.figure(figsize=(8, 6))
plt.plot(K, wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)

plt.savefig("/storage/emulated/0/elbow_method.png")
plt.close()

print("✅ Elbow graph saved")
print("📁 Location: Internal Storage → elbow_method.png") 