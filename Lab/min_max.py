import numpy as np

def min_max_scaler_array(data_array):
    data_array = np.array(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)

    if min_val == max_val:
        return np.zeros_like(data_array, dtype=float)

    scaled = (data_array - min_val) / (max_val - min_val)
    return scaled

# Example usage
data = np.array([12000, 1800, 250, 30, 3])
scaled = min_max_scaler_array(data)

print("Original Data:", data)
print("Min-Max Scaled Data:", scaled)
