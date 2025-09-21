import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, 45, 50, 60],
    'Salary': [30000, 40000, 60000, 80000, 100000]
})

print("Before Normalization:\n", data)

# Apply Min-Max Normalization
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

print("\nAfter Normalization:\n", normalized_data)