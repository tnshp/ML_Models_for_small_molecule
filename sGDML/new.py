import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io
import argparse

data = np.load('Datasets/Glycine.npz')

print(data['E'])