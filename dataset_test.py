import numpy as np 
import matplotlib.pyplot as plt

dataset = np.load("Simulation/BrClN-Q/BrClN-Q.npz")

print(dataset.files) 
for key in dataset.files:
    print(f'key: {key} \t\t shape: {dataset[key].shape}')

print(len(dataset['E']))
# print(dataset['F'][: ,0, 0].shape)
# plt.plot(dataset['F'][:,0, 0])
# plt.plot(dataset['F'][:,0, 1])
# plt.plot(dataset['F'][:,0, 2])
plt.plot(dataset['E'] - np.min(dataset['E']))
plt.show()