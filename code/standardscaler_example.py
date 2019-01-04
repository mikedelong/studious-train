import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([1.0, 0.0, 2.0, 3.0, 4.0])
print(data)

scaled = (data - np.mean(data)) / np.std(data)
print(scaled)

scaler = StandardScaler().fit(X=data.reshape(-1, 1))
print(scaler.transform(data.reshape(1, -1)))
print(scaler.mean_)
print(scaler.scale_)
