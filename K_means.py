import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv('C:/Users/shree/Desktop/DAA_lab/ML_exam/Mall_Customers.csv')
features=['Annual Income (k$)','Spending Score (1-100)']
X=data[features]
scaler=StandardScaler()
scaled_data=scaler.fit_transform(X)
em_model=GaussianMixture(n_components=5,random_state=42)
em_clusters=em_model.fit_predict(scaled_data)
em_kmeans=KMeans(n_clusters=5,random_state=42)
kmeans_clusters=em_kmeans.fit_predict(scaled_data)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=em_clusters, cmap='viridis')
plt.title('EM Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 2, 2)  # Corrected index to 2 for the second subplot
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_clusters, cmap='viridis')
plt.title('Kmeans clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.tight_layout()
plt.show()
