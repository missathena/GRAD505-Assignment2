import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn import datasets

iris = datasets.load_iris()
iris_data = DataFrame(iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target_names[iris.target]
print(iris_data.head())
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
print(PlantGrowth)

#1a
width =  iris_data['sepal width (cm)'].tolist()
sns.histplot(width,bins=10,kde=True)
plt.show()

#1c
mean = np.mean(width)
mean = np.round(mean,1)
median = np.median(width)
print(f"mean: {mean} median: {median}")

#1d
percentile27 = np.percentile(width,[27])
print(f"percentile27: {percentile27}")

#1e.
sns.scatterplot(data=iris_data,x=iris_data["sepal length (cm)"], y=iris_data["sepal width (cm)"], hue=iris_data["species"])
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("sepal length (cm) and sepal width (cm)")
plt.show()

sns.scatterplot(data=iris_data,x=iris_data["petal length (cm)"], y=iris_data["petal width (cm)"], hue=iris_data["species"])
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("petal length (cm) and petal width (cm)")
plt.show()

sns.scatterplot(data=iris_data,x=iris_data["sepal length (cm)"], y=iris_data["petal length (cm)"], hue=iris_data["species"])
plt.xlabel("speal length (cm)")
plt.ylabel("petal length (cm)")
plt.title("speal length (cm) and petal length (cm)")
plt.show()

sns.scatterplot(data=iris_data,x=iris_data["sepal width (cm)"], y=iris_data["petal width (cm)"], hue=iris_data["species"])
plt.xlabel("speal width (cm)")
plt.ylabel("petal width (cm)")
plt.title("speal width (cm) and petal width (cm)")
plt.show()

sns.scatterplot(data=iris_data, x=iris_data["species"], y=iris_data.index, hue=iris_data["species"])
plt.xlabel("species")
plt.ylabel("index")
plt.title("species count")
plt.show()

#2a.
weight = 3.3
weight_max = 6.3
weight_range = np.arange(weight,weight_max , 0.3)
sns.histplot(data=PlantGrowth["weight"], bins=weight_range)
plt.show()

#2b
sns.boxplot(data=PlantGrowth,x="weight",hue="group")
plt.show()

#2d
trt1 = pd.DataFrame(PlantGrowth[PlantGrowth["group"] == "trt1"])
trt2 = pd.DataFrame(PlantGrowth[PlantGrowth["group"] == "trt2"])
print(trt1.count())
trt2_min = trt2['weight'].min()
print(trt2_min)
less_than = pd.DataFrame(trt1[trt1["weight"] < trt2_min])
less_than_percent = (less_than.count()/trt1.count())*100
print(less_than_percent)
