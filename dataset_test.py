import numpy as np 

dataset = np.load("Glycine.npz")

print(dataset.files)
for key in dataset.files:
    print(f'key: {key} \t\t shape: {dataset[key].shape}')