[TOC]



# Dealing with Missing Data

Tired of working with messy data? Did you know that most of a data scientist's time is spent in finding, cleaning and reorganizing data?! Well turns out you can clean your data in a smart way! In this course Dealing with Missing Data in Python, you'll do just that! You'll learn to address missing values for numerical, and categorical data as well as time-series data. You'll learn to see the patterns the missing data exhibits! While working with air quality and diabetes data, you'll also learn to analyze, impute and evaluate the effects of imputing the data.

## The Problem With Missing Data

Get familiar with missing data and how it impacts your analysis! Learn about different null value operations in your dataset, how to find missing data and summarizing missingness in your data.

The workflow that goes into dealing with missing data is the following: 

1.  Convert missing values to null values. This is important because by doing so you will be able to identify them, replace them or play with them in the following steps. 
2.  Analyze the amount and type of missing values in the data
3.  Decide on how to delete or impute the missing values
4.  Evaluate & Compare the performance of the treated / imputed dataset

There are two types of null values in the data. They are the `None` which is a python-defined null value and the other is `NaN`, which is typically assigned by `np.nan`. Pandas use the latter. Here are few properties of these two: 

| `None`                                          | `np.nan`                                             |
| ----------------------------------------------- | ---------------------------------------------------- |
| `None` or `True` returns `True`                 | `np.nan` or `True` returns `nan`                     |
| `None` does not work with arithmetic operations | `np.nan` returns `nan` for all arithmetic operations |
| Type(None) is of `NoneType`                     | `Type(np.nan)` is float                              |

### Checking with Null Values

When we compare `None` as in `None == None` we get the expected answer `True`. However, this is not the case for `np.nan == np.nan` as null values cannot be equal. However, we can check this equality using `np.isnan(np.nan)` and we get `True` as expected. 

### Assigning missing values with `np.nan`

To try this out, let's load the college data. The missing values are usually filled with `NA`, `-` or `.` etc...

```python
import pandas as pd
college = pd.read_csv('college.csv')
college.head()
```

![image-20201209093455997](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209093455997.png)

We see here that the missing values seem to be filled with a period. We see that all the columns are of type `float` but we see a discrepancy when we do: 

```python
college.info()
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209093622347.png" alt="image-20201209093622347" style="zoom:80%;" />

This makes sense are we have a `.` for missing values for most columns. We can look for different type of missing values that have been filled in using the `.unique()` method in pandas. We, of course, have to do this for all columns one at a time. Let's look at `csat` column: 

```python
csat_unique = college.csat.unique()
np.sort(csat_unique)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209093912563.png" alt="image-20201209093912563" style="zoom:80%;" />

We see that the only character used for missing value is the period. So, this make is easier for us to replace this column. Suppose, we find that the period is the only character used replace the missing value. We can then replace that period with `nan` when we read the file itself: 

```python
college = pd.read_csv('college.csv', na_values = '.')
```

However, if we find that there are different types such as periods, dashes, or even 'NA', we can pass a list instead: 

```python
college = pd.read_csv('college.csv', na_values=['.', '-', 'NA'])
```

### Missing Values in Disguise

Sometimes the missing values are not really replace with strings, periods or dashes. In fact, there are replaced a legitimate value such as `0`. Let's look at another set: 

```python
pima = pd.read_csv('pima-indians-diabetes data.csv')
pima.head()
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209094702106.png" alt="image-20201209094702106" style="zoom:80%;" />

We see that the missing values are correctly replaced as `NaN`. However, we see something strange when we look at the data distribution: 

```python
pima.describe.T
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209094810408.png" alt="image-20201209094810408" style="zoom:80%;" />

We see that the minimum value for `BMI` is `0`. We know that `BMI` cannot be zero. So, it looks like the missing values have been replaced with a `0`. Such type of missing values are tricky as they require some level of domain knowledge. 

We can replace these zero values with `NaN` as follows: 

```python
pima.loc[:, 'BMI'] = pima.loc[:, 'BMI'].replace(np.nan, 0)
```

### Analyse the Amount of Missing Data in a Dataset

 