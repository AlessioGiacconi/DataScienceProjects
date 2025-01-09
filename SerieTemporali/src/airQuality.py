import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset_path: str = '../DataScienceProjects/SerieTemporali/Dataset/athens_data.csv'
raw_dataset = pd.DataFrame(pd.read_csv(dataset_path))
description = raw_dataset.describe()
pass