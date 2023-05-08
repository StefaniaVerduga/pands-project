# Programming and Scripting Project

Course: HDip in Computer in Data Analytics  
Module: Programming and Scripting  
Author: Stefania Verduga  

The "pands-project" repository contains the Fisher's Iris research and data study as part of the Final Project of the Programming and Scripting module.

## Table of Contents

1. [Description](#Description)
2. [Dataset Information](#Dataset-Information)
3. [Dataset Code and Analysis](#Dataset-Code-and-Analysis)
- [Loading the dataset](#Loading-the-dataset)
- [Analysis of the Iris Dataset](#Analysis-of-the-Iris-Dataset)
4. [Plots](#Plots)
- [Histograms](#Histograms)
- [Scatterplots](#Scatterplots)
- [Pairplot](#Pairplot)
5. [Conclusion](#Conclusion)
6. [References](#References)

## Description

The Iris flower data set or Fisher’s Iris data set is one of the most famous multivariate data set used for testing various Machine Learning Algorithms. It was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The Use of Multiple Measurements in Taxonomic Problems" as an example of linear discriminant analysis.
The dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). These measures were used to create a linear discriminant model to classify the species. [01]

Iris Species [02]
![Iris Species](https://github.com/StefaniaVerduga/pands-project/blob/main/Pictures/iris.png) 

## Dataset Information

The Iris Dataset consists of 50 samples from each of three species of Iris flowers: Iris setosa, Iris virginica, and Iris versicolor. 
Attribute information:

1. Sepal length in cm
2. Sepal width in cm
3. Petal length in cm
4. Petal width in cm
5. Class: Iris Setosa - Iris Versicolor - Iris Virginica

These features are used to classify the flowers into their respective species. The dataset is often used for machine learning tasks such as classification and clustering. [03]

## Dataset Code and Analysis

### Loading the dataset

The first step for the analysis is obtain the data set information by downloading it from: [https://archive.ics.uci.edu/ml/datasets/iris] and convert it in comma-separated values (csv). This data source can be founded in the repository under the name of [iris_dataset.csv](https://github.com/StefaniaVerduga/pands-project/blob/main/iris_dataset.csv)

In order to process all this information we need to import some necessary packages for the project:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
* **Numpy**: used to perform a wide variety of mathematical operations on arrays. NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation. [04]
* **Pandas**: used to perform data manipulation and analysis.Pandas is a Python library for data analysis. It is built on top of two core Python libraries—matplotlib for data visualization and NumPy for mathematical operations. Pandas acts as a wrapper over these libraries, allowing you to access many of matplotlib's and NumPy's methods with less code. [05]
* **Matplotlib**: used for data visualization and graphical ploting. Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. [06]
* **Seaborn**: built on top of matplotlib with similar functionalities. Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. [07]

### Analysis of the Iris Dataset

The CSV file previously downloaded, is read into our repository using the Pandas 'read_csv' method and it is stored in a pandas DataFrame object named 'data'. [08]
```
data = pd.read_csv("iris_dataset.csv")
```

The first step in this analysis is to investigate the structure of our data and calculate general statistical features such as mean, standard deviation, max/min values. This information is gathered and saved in a summary report named ['iris-data-summary.txt'](https://github.com/StefaniaVerduga/pands-project/blob/main/iris-data-summary.txt)

The next lines of codes print out various summary statistics about the dataset.[09]
In Python, the 'open' function is used to open files and performs operations on them. It returns a file object that can be used to read, write or manipulate the content on the file.
There are different modes to open a file in Python, in this case the mode used was 'w' which opens a file for writing, creates a new file if it does not exist or truncates the file if it exists. 
For the next lines of code, the mode used was 'a' which is the append mode, where the data is added at the end of the file. [10]

```
print("Overview of the Fisher's Iris Dat Set", file=open("iris-data-summary.txt", "w"))
print(data.head(), file=open("iris-data-summary.txt", "a"))
print(data.describe(), file=open("iris-data-summary.txt", "a"))
print(data.info(), file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').size(), file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').mean(), file=open("iris-data-summary.txt", "a"))
```

* The "head" method in Pandas is used to display the first few rows of the dataset. It returns the first 5 rows if a number is not specified. [11]
* The "describe" method returns the description of the data columns. It displays the count, mean, standard deviation, minimum and maximum values for each column in the dataset. [12]
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
The variable "counts" represents the number of non-missing values for each feature. In this case, there are 150 values for each feature, which means that there are not missing values in the dataset. [13]
In this table, we can find also the mean value of each feature  or the minimun and maximum value. For example, the minimum value of the sepal length is 4.3 cm and the maximum value is 7.9 cm being one of the highest values within all categories for the different Iris species.
* The "info" method prints information about the data that we are analizing. The information contains the number of columns, column labels (in this case the features), column data types and the number of cells in each column. All the input attributes (0-3) are in float and the output attribute (4) is in object. [14]
* The "groupby" method prints out the number of samples on each class in the dataset. This functions is used to split the data into groups based on some criteria. It helps to aggregate data efficiently. [15]
* The final line of code prints out the mean values categorized by species and uses the "groupby" method to do so, followed by the variable "class" which is the variable name assigned for each iris specie. Doing this, we can get the mean value of the sepal length, sepal width, petal length and petal width for each specie of Iris. [16]
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

## Plots

In order to evaluate the data set it is necessary to create some plots to condense the information so that we are able to draw the main conclusions. For this project I have created some histograms, scatterplots and pairplot.

### Histograms

![Overview Histogram](https://github.com/StefaniaVerduga/pands-project/blob/main/Histograms/overview-Histogram.png)

![Sepal Length by Specie](https://github.com/StefaniaVerduga/pands-project/blob/main/Histograms/Sepal-length-class.png)

![Sepal Width by Specie](https://github.com/StefaniaVerduga/pands-project/blob/main/Histograms/Sepal-width-class.png)

![Petal Length by Specie](https://github.com/StefaniaVerduga/pands-project/blob/main/Histograms/Petal-length-class.png)

![Petal Width by Specie](https://github.com/StefaniaVerduga/pands-project/blob/main/Histograms/Petal-width-class.png)

## References
[01][https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5]

[02][https://www.codecademy.com/courses/machine-learning/lessons/machine-learning-clustering/exercises/iris-dataset]

[03][http://www.lac.inpe.br/~rafael.santos/Docs/CAP394/WholeStory-Iris.html#:~:text=The%20Iris%20Dataset%20contains%20four,model%20to%20classify%20the%20species.]

[04][https://numpy.org/doc/stable/user/whatisnumpy.html]

[05][https://mode.com/python-tutorial/libraries/pandas/]

[06][https://matplotlib.org/]

[07][https://seaborn.pydata.org/#:~:text=Seaborn%20is%20a%20Python%20data,attractive%20and%20informative%20statistical%20graphics.]

[08][https://www.w3schools.com/python/pandas/pandas_csv.asp]

[09][https://www.w3schools.com/python/python_file_open.asp]

[10][https://www.programiz.com/python-programming/file-operation#:~:text=Opening%20Files%20in%20Python,txt%20with%20the%20following%20content.&text=Now%2C%20let's%20try%20to%20open,using%20the%20open()%20function.]

[11][https://www.w3resource.com/pandas/dataframe/dataframe-head.php]

[12][https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm#:~:text=The%20describe()%20function%20computes,pertaining%20to%20the%20DataFrame%20columns.&text=This%20function%20gives%20the%20mean,given%20summary%20about%20numeric%20columns.]

[13][https://www.sharpsightlabs.com/blog/pandas-describe/]

[14][https://www.w3schools.com/python/pandas/ref_df_info.asp#:~:text=The%20info()%20method%20prints,method%20actually%20prints%20the%20info.]

[15][https://www.geeksforgeeks.org/python-pandas-dataframe-groupby/]

[16][https://stackoverflow.com/questions/49970309/pandas-groupby-with-mean]