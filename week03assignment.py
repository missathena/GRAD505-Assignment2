import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_data.head())
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
print(PlantGrowth)

#1a
width =  iris_data['sepal width (cm)'].tolist()
sns.histplot(width,bins=10,kde=True)
#plt.show()
#1c
mean = np.mean(width)
mean = np.round(mean,1)
median = np.median(width)
print(f"mean: {mean} median: {median}")