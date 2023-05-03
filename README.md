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

![Iris Species](/Users/stefania/Pands/pands-project/iris.png)


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

The first step for the analysis is obtain the data set information by downloading it from: [https://archive.ics.uci.edu/ml/datasets/iris] and convert it in comma-separated values (csv). This data source can be founded in the repository under the name of [https://github.com/StefaniaVerduga/pands-project/blob/main/iris_dataset.csv]

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



