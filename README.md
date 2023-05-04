# Programming and Scripting Project

Course: HDip in Computer in Data Analytics  
Module: Programming and Scripting  
Author: Stefania Verduga  

The "pands-project" repository contains the Fisher's Iris research and data study as part of the Final Project of the Programming and Scripting module.

## Table of Contents

1. [Description](#Description)
2. [Dataset Information](#Dataset-Information)
3. [Dataset Code and Analysis](#Dataset-Code-and-Analysis)
4. [Plots](#Plots)
5. [Conclusion](#Conclusion)
6. [References](#References)

## Description

The Iris flower data set or Fisher’s Iris data set is one of the most famous multivariate data set used for testing various Machine Learning Algorithms. It was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The Use of Multiple Measurements in Taxonomic Problems" as an example of linear discriminant analysis.
The dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). These measures were used to create a linear discriminant model to classify the species. 

![Iris Species](https://github.com/StefaniaVerduga/pands-project/blob/main/Pictures/iris.png)


## Dataset Information

The Iris Dataset consists of 50 samples from each of three species of Iris flowers: Iris setosa, Iris virginica, and Iris versicolor. 
Attribute information:

1. Sepal length in cm
2. Sepal width in cm
3. Petal length in cm
4. Petal width in cm
5. Class: Iris Setosa - Iris Versicolor - Iris Virginica

These features are used to classify the flowers into their respective species. The dataset is often used for machine learning tasks such as classification and clustering.

## Dataset Code and Analysis

### Load the dataset

The first step for the analysis is obtain the data set information by downloading it from: [https://archive.ics.uci.edu/ml/datasets/iris] and convert it in comma-separated values (csv). This data source can be founded in the repository under the name of [iris_dataset.csv](https://github.com/StefaniaVerduga/pands-project/blob/main/iris_dataset.csv)

In order to process all this information we need to import some necessary packages for the project:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
```
* Numpy: used to perform a wide variety of mathematical operations on arrays.
* Pandas: used to perform data manipulation and analysis.
* Matplotlib: used for data visualization and graphical ploting.
* Seaborn: built on top of matplotlib with similar functionalities.
* Sys: provides access to some variables and functions that interact with the interpreter.

### Analysis of the Iris Dataset

The CSV file previously downloaded, is read into our repository using the Pandas 'read_csv' method and it is stored in a pandas DataFrame object named 'data'.
```
data = pd.read_csv("iris_dataset.csv")
```

The first step in this analysis is to investigate the structure of our data and calculate general statistical features such as mean, standard deviation, max/min values. This information is gathered and saved in a summary report named ['iris-data-summary.txt'](https://github.com/StefaniaVerduga/pands-project/blob/main/iris-data-summary.txt)

The next lines of codes print out various summary statistics about the dataset.

```
print("Overview of the Fisher's Iris Dat Set", file=open("iris-data-summary.txt", "w"))
print(data.head(), file=open("iris-data-summary.txt", "a"))
print("Basic descriptive statistics", file=open("iris-data-summary.txt", "a"))
print(data.describe(), file=open("iris-data-summary.txt", "a"))
print("Basic information about data type", file=open("iris-data-summary.txt", "a"))
print(data.info(), file=open("iris-data-summary.txt", "a"))
print("Number of samples on each class", file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').size(), file=open("iris-data-summary.txt", "a"))
print("Mean values categorized by specie", file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').mean(), file=open("iris-data-summary.txt", "a"))
```

* The "head" method in Pandas is used to display the first few rows of the dataset.
* The "describe" method displays the count, mean, standard deviation, minimum and maximum values for each column in the dataset.
```
Basic descriptive statistics
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```
The variable "counts" represents the number of non-missing values for each feature. In this case, there are 150 values for each feature, which means that there are not missing values in the dataset.
In this table, we can find also the mean value of each feature  or the minimun and maximum value. For example, the minimum value of the sepal length is 4.3 cm and the maximum value is 7.9 cm being one of the highest values within all categories for the different Iris species.
* The "info" method includes basic information about the data type. All the input attributes (0-3) are in float and the output attribute (4) is in object.
* The "groupby" method prints out the number of samples on each class in the dataset.
* The final line of code prints out the mean values categorized by species and uses the "groupby" method to do so, followed by the variable "class" which is the variable name assigned for each iris specie.
```
Mean values categorized by specie
                 sepal_length  sepal_width  petal_length  petal_width
class                                                                
Iris-setosa             5.006        3.418         1.464        0.244
Iris-versicolor         5.936        2.770         4.260        1.326
Iris-virginica          6.588        2.974         5.552        2.026
```
The information provided shows the mean values of sepal length, sepal width, petal length, and petal width for each category of iris species: Iris setosa, Iris versicolor, and Iris virginica.
As per the information displayed in the table above, we can see that the specie with the highest values for almost all the categories is the Iris Virginica, and on the contrary the Iris Setosa would be the specie with the smallest values.