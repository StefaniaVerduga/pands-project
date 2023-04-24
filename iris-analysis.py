# iris-analysis.py
# Author: Stefania Verduga

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Loading the dataset
data = pd.read_csv("iris_dataset.csv")

# Dataset overview
print("Overview of the Fisher's Iris Dat Set", file=open("iris-data-summary.txt", "w"))
print(data.head(), file=open("iris-data-summary.txt", "a"))
print("=============================================", file=open("iris-data-summary.txt", "a"))
print("Basic descriptive statistics", file=open("iris-data-summary.txt", "a"))
print(data.describe(), file=open("iris-data-summary.txt", "a"))
print("=============================================", file=open("iris-data-summary.txt", "a"))
print("Basic information about data type", file=open("iris-data-summary.txt", "a"))
print(data.info(), file=open("iris-data-summary.txt", "a"))
print("=============================================", file=open("iris-data-summary.txt", "a"))
print("Number of samples on each class", file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').size(), file=open("iris-data-summary.txt", "a"))
print("=============================================", file=open("iris-data-summary.txt", "a"))
print("Mean values categorized by specie", file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').mean(), file=open("iris-data-summary.txt", "a"))

# https://barcelonageeks.com/diagrama-de-caja-y-exploracion-de-histograma-en-datos-de-iris/
# Histogram for sepal length
plt.figure(figsize = (8, 6))
x = data["sepal_length"]
plt.hist(x, bins = 10, color = "lightblue")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Frequency")
plt.show()

# Histogram for sepal width
plt.figure(figsize = (8, 6))
x = data["sepal_width"]
plt.hist(x, bins = 10, color = "lightblue")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Frequency")
plt.show()

# Histogram for petal length
plt.figure(figsize = (8, 6))
x = data["petal_length"]
plt.hist(x, bins = 10, color = "lightblue")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Frequency")
plt.show()

# Histogram for petal width
plt.figure(figsize = (8, 6))
x = data["petal_width"]
plt.hist(x, bins = 10, color = "lightblue")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Frequency")
plt.show()

# https://www.educba.com/seaborn-histogram/
fig, ax = plt.subplots(figsize=(8,6))

sns.histplot(data=data, x="sepal_length", hue="class", bins=10)
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length by Class")
plt.show()

sns.histplot(data=data, x="sepal_width", hue="class", bins=10)
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Width by Class")
plt.show()

sns.histplot(data=data, x="petal_length", hue="class", bins=10)
plt.ylabel("Frequency")
plt.title("Histogram of Petal Length by Class")
plt.show()

sns.histplot(data=data, x="petal_width", hue="class", bins=10)
plt.ylabel("Frequency")
plt.title("Histogram of Petal Width by Class")
plt.show()