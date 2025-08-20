from os import path, makedirs
import numpy as np

def normalization(data: np.array) -> np.array:
    min_value, max_value = np.min(data), np.max(data)
    diff = max_value - min_value
    return (data - min_value) / diff if diff != 0 else data

def standardization(data: np.array) -> np.array:
    mean_data = np.mean(data)
    std_data = np.std(data)
    return (data - mean_data) / std_data if std_data != 0 else data

def save_data(use: str, X:np.array, y:np.array, sufix: str) -> None:
    use = use.strip().lower()
    try:
        if use not in ["train", 'validation', "test"]:
           raise TypeError("Only 'train', 'validation' and 'test' uses are avaiable for the parameter 'use'.")
    
        folder_data = f"data/{use}/"
        # Create directory if it doesn't exist
        makedirs(path.dirname(folder_data), exist_ok=True)

        data = np.column_stack((X, y))
        save_path = folder_data + f"{use}_{sufix}_data"
        np.save(save_path + ".npy", data)
        np.savetxt(save_path + ".csv", data, delimiter=",", fmt="%.3f")
        
    except(TypeError, ValueError) as e:
        print(e)