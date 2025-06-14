import matplotlib.pyplot as plt
import numpy as np
import os

file = "ums/2025-06-12_11-38-49/umbrella_data.npz"

data = np.load(file)
print(data.keys())
# centers = data['centers']