import numpy as np

def min_max_scaler_array(data_array):
    data_array = np.array(data_array)
    return (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))

def standard_scaler_array(data_array):
    data_array = np.array(data_array)
    return (data_array - np.mean(data_array)) / np.std(data_array)

# Example data
data = np.array([5, 12, 9, 21, 30])

min_max_scaled = min_max_scaler_array(data)
standard_scaled = standard_scaler_array(data)

print("Original Data:", data)
print("Min-Max Scaled:", min_max_scaled)
print("Standard Scaled:", standard_scaled)
