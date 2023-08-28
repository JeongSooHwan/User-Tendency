import numpy as np
import pandas as pd
from sklearn.cluster import KMeans



k = 3
model = KMeans(n_clusters=k, random_state=10)

model.fit()