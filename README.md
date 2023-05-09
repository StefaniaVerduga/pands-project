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

```python
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
```python
data = pd.read_csv("iris_dataset.csv")
```

The first step in this analysis is to investigate the structure of our data and calculate general statistical features such as mean, standard deviation, max/min values. This information is gathered and saved in a summary report named ['iris-data-summary.txt'](https://github.com/StefaniaVerduga/pands-project/blob/main/iris-data-summary.txt)

The next lines of codes print out various summary statistics about the dataset.[09]
In Python, the 'open' function is used to open files and performs operations on them. It returns a file object that can be used to read, write or manipulate the content on the file.
There are different modes to open a file in Python, in this case the mode used was 'w' which opens a file for writing, creates a new file if it does not exist or truncates the file if it exists. 
For the next lines of code, the mode used was 'a' which is the append mode, where the data is added at the end of the file. [10]

```python
print("Overview of the Fisher's Iris Dat Set", file=open("iris-data-summary.txt", "w"))
print(data.head(), file=open("iris-data-summary.txt", "a"))
print(data.describe(), file=open("iris-data-summary.txt", "a"))
print(data.info(), file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').size(), file=open("iris-data-summary.txt", "a"))
print(data.groupby('class').mean(), file=open("iris-data-summary.txt", "a"))
```

* The "head" method in Pandas is used to display the first few rows of the dataset. It returns the first 5 rows if a number is not specified. [11]
* The "describe" method returns the description of the data columns. It displays the count, mean, standard deviation, minimum and maximum values for each column in the dataset. [12]
<details>
<summary markdown="span">Basic descriptive statistics - click me to expand</summary>

       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
</details>
The variable "counts" represents the number of non-missing values for each feature. In this case, there are 150 values for each feature, which means that there are not missing values in the dataset. [13]
In this table, we can find also the mean value of each feature  or the minimun and maximum value. For example, the minimum value of the sepal length is 4.3 cm and the maximum value is 7.9 cm being one of the highest values within all categories for the different Iris species.
* The "info" method prints information about the data that we are analizing. The information contains the number of columns, column labels (in this case the features), column data types and the number of cells in each column. All the input attributes (0-3) are in float and the output attribute (4) is in object. [14]
* The "groupby" method prints out the number of samples on each class in the dataset. This functions is used to split the data into groups based on some criteria. It helps to aggregate data efficiently. [15]
* The final line of code prints out the mean values categorized by species and uses the "groupby" method to do so, followed by the variable "class" which is the variable name assigned for each iris specie. Doing this, we can get the mean value of the sepal length, sepal width, petal length and petal width for each specie of Iris. [16]
<details>
<summary markdown="span">Mean values categorized by specie - click me to expand</summary>

                 sepal_length  sepal_width  petal_length  petal_width
class                                                                
Iris-setosa             5.006        3.418         1.464        0.244
Iris-versicolor         5.936        2.770         4.260        1.326
Iris-virginica          6.588        2.974         5.552        2.026
</details>
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

![Mean per Specie Histogram](https://github.com/StefaniaVerduga/pands-project/blob/main/Mean-per-class-Histogram.png)

### Code explanation
#### Overview Histogram
To create an overview histogram of all the features of the different species, it is needed to use some Python functions.
```python
data.hist(figsize = (9,6), color = "lightblue")
plt.savefig('Overview-Hist.png')
plt.show()
```
The first line of code creates an histogram of the data stored in the "data" DataFrame. The 'hist()' function is a Panda method that computes and visualizes the distribution of each column in the DataFrame as a histogram. It is possible to customize the appearance of the histogram by supplying the 'hist()' method additional parameters. 
In this case, the parameters used were the 'figsize' parameter which sets the size of the plot and the 'color' parameter which sets the color of the bars in the histogram. [17]
We use the 'plt.savefig()' function to save the figure as a PNG image file on the local machine. [18]
The last line of the code displays the figure on the screen. [19] [20]

#### Features by Species Histogram
This program creates a figure and axes objects using Seaborn 'histplot()' function. 
```python
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(data=data, x="sepal_length", hue="class", bins=10)
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length by Class")
```
A new figure and axes objects is created using the 'subplots()' function from the 'pyplot' module of Matplotlib. The 'fig' object represents the entire figure, while the 'ax' object represents a single subplot or axes within the figure. [21] 
The 'Seaborn' 'histplot()' function was used to create the histogram. The 'data' parameter indicates the DataFrame from which the data will be drawn. The 'x' parameter specifies the column in the DataFrame to be plotted on the x-axis, which is "sepal_length" in this case and will vary in the other histograms depending on the features of the Iris. The 'hue' parameter indicates the column that defines different categories or groups, which is "class" in this case and refers to the different Iris species. The 'bins' parameter indicates the number of bins or intervals to use in the histogram. [22]
The 'plt.ylabel()' and 'plt.title()' functions set the label for the y-axis of the plot, which in this case is "Frequency" and the title respectively. [23] [24]

#### Mean per Species Histogram

```python
grouped_data = data.groupby('class')
averages = grouped_data.mean()

averages.plot(kind='bar', figsize=(9, 7))
plt.xlabel("Class")
plt.ylabel("Average Value")
plt.title("Average Value of Each Feature for Each Class")
plt.xticks(rotation=0)
plt.legend(title="Feature")
plt.savefig('Mean-per-class-Histogram.png')
plt.show()
```
For this histogram, I wanted to show the main values of each feature per specie. In order to do so, I used the the function 'groupby()' which is used for grouping the data according to the categories and applying a function to the categories. It is useful to aggregate data efficiently. In this case I grouped the data by "Class" which refers to the different Iris species. [25]

Once the data is grouped by species, it is needed to calculate the mean of each feature, so in order to do so, I used the method 'mean()' The 'mean()' method is applied to the 'grouped_data' object, resulting in a new DataFrame, averages, that contains the average values for each feature in each class. [26]

The next lines of codes are related the format of the plot of the 'averages' DataFrame using the 'plot()' method, already used for the previous histograms. I set several parameters as 'kind' to specify that a bar plot should be created and the 'figsize' to set the size of the plot.

To set the x-axis and y-axis labels of the plots, it was needed to use the functions 'xlabel()' and 'ylabel()' respectively. The 'xticks()' function sets the rotation of the x-axis tick label to 0 degrees, ensuring that the class names are not rotated. [27]

### Scatter Plots

![ScaterPlotSLSW](https://github.com/StefaniaVerduga/pands-project/blob/main/ScatterPlots/scatplotSLSW.png)

![ScaterPlotPLPW](https://github.com/StefaniaVerduga/pands-project/blob/main/ScatterPlots/scatplotPLPW.png)

### Code explanation

```python
sns.FacetGrid(data, hue ="class", height = 6) \
    .map(plt.scatter,'sepal_length','sepal_width') \
    .add_legend()
plt.title("Sepal Length vs Sepal Width", fontsize=12, fontweight= 'bold')
plt.subplots_adjust(top=0.9)
plt.savefig('scatplotSLSW.png')
```
In order to create a scatter plot for this project, I have used the FaceGrid function. FacetGrid function helps in visualizing distribution of one variable as well as the relationship between multiple variables separately within subsets of your dataset using multiple panels.
The first line of the program creates a FacetGrid object. It takes in the data DataFrame as input and sets the color 'hue' to be determined by the 'class' column. The 'height' parameter specifies the height of each facet in the grid. [28]

In the next line, the 'map()' method is used to specify the type of plot and the variables to be plotted on the x-axis and y-axis within the FacetGrid. The 'scatter()' function is used to create a scatter plot, which displays points on a two-dimensional plane. The 'sepal_length' column will be plotted on the x-axis, and the 'sepal_width' column will be plotted on the y-axis.

A legend was also added to this scatter plot using the 'add_legend()' function, which shows the mapping between the color hue (representing different classes) and the respective class labels.

The next step in this case was adding a title and adjust the subplot layout, specifically setting the top margin of the plot to 0.9. This is useful to make space for the title at the top.

Finally, the 'savefig()' function was used to save this plot into the machine, under the name of 'scatplotSLSW.png' [29] [30]
### Pairplot

![Pairplot](https://github.com/StefaniaVerduga/pands-project/blob/main/pairplot.png)

### Code explanation

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

[17][https://mode.com/example-gallery/python_histogram/]

[18][https://www.educative.io/answers/what-is-the-matplotlibpyplotsavefig-function-in-python]

[19][https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html]

[20][https://www.kaggle.com/code/agilesifaka/step-by-step-iris-ml-project]

[21][https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html]

[22][https://seaborn.pydata.org/generated/seaborn.histplot.html]

[23][https://www.w3schools.com/python/matplotlib_labels.asp]

[24][https://www.educba.com/seaborn-histogram/]

[25][https://www.geeksforgeeks.org/python-pandas-dataframe-groupby/]

[26][https://www.geeksforgeeks.org/pandas-groupby-and-computing-mean/]

[27][https://data-flair.training/blogs/iris-flower-classification/]

[28][https://www.geeksforgeeks.org/python-seaborn-facetgrid-method/]

[29][https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset]

[30][https://www.geeksforgeeks.org/plotting-graph-for-iris-dataset-using-seaborn-and-matplotlib/?ref=rp]
