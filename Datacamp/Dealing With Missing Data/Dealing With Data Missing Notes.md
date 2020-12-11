[TOC]



# Dealing with Missing Data

Tired of working with messy data? Did you know that most of a data scientist's time is spent in finding, cleaning and reorganizing data?! Well turns out you can clean your data in a smart way! In this course Dealing with Missing Data in Python, you'll do just that! You'll learn to address missing values for numerical, and categorical data as well as time-series data. You'll learn to see the patterns the missing data exhibits! While working with air quality and diabetes data, you'll also learn to analyze, impute and evaluate the effects of imputing the data.

## Chapter 1: The Problem With Missing Data

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

The analysis involves finding the total number and the percentage of missing data. Let's look at another dataset: 

```python
df_air = pd.read_csv(data + 'air-quality.csv',
                                    parse_dates=['Date'],
                                    index_col='Date')
df_air.head()
```

 <img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209100228498.png" alt="image-20201209100228498" style="zoom:80%;" />

We next find the nullity of a data frame using either the `.isnull()` or `.isna()` method on the data frame: 

```python
air_null = df_air.isna()
air_null.head()
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209100827307.png" alt="image-20201209100827307" style="zoom:80%;" />

We get a mask of null values in the dataset. We can then find the total number of missing values: 

```python
air_null.sum()
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209101046248.png" alt="image-20201209101046248" style="zoom:80%;" />

We can find the percentage by doing the following: 

```python
air_null.mean() * 100
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209101208870.png" alt="image-20201209101208870" style="zoom:80%;" />

### Graphically Looking at Missing Values

We make use of the `missingno` package for graphical analysis of missing values. 

```python
import missingno as msno
msno.bar(df_air)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209101517243.png" alt="image-20201209101517243" style="zoom:80%;" />

We can also look at the location of missing values in the dataset. This allows us to look for patterns in the missing values. We do this by using the **nullity matrix**:

```python
msno.matrix(df_air)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209101713323.png" alt="image-20201209101713323" style="zoom:80%;" />

As this dataset is a time series dataset, we can use time series keywords to see any patterns in missing dataset: 

```python
# Look at the dataset at the monthly frequency
msno.matrix(df_air, freq='M')
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209102015050.png" alt="image-20201209102015050" style="zoom:80%;" />

We see that a large number of missing values in June. We can next slice to zoom into those missing data: 

```python
msno.matrix(df_air.loc['May-1976': 'Jul-1976'], freq='M')
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209102158128.png" alt="image-20201209102158128" style="zoom:80%;" />

## Chapter 2: Looking for Patterns in Missing Data

Analyzing the type of missingness in your dataset is a very important step towards treating missing values. In this chapter, you'll learn in detail how to establish patterns in your missing and non-missing data, and how to appropriately treat the missingness using simple techniques such as listwise deletion.

There are many reasons why data could be missing. Some of the reasons are: 

*   Values are missing at random instances or intervals for a given variable. We call these 
*   Values missing due to a dependency on another variable. 
*   Values missing due to a specific reason why is often uncovered from domain knowledge. 

We can categorize missing data into three broad types: 

1.  **Missing Completely At Random (MCAR**): Missing values have no relationships between values that are either observed or missing. 
2.  **Missing at Random (MAR)**: There is a systematic relationship between missing values and another variable. Such a variable can be either observed or confounding. For example, a winter class may have missing students that may seem at random but perhaps they are all fallen sick to flu. In this example, there is a weak correlation between seasons and missing students. 
3.  **Missing Not at Random (MNAR)**: There is relationship between missing values and a variable either observed or confounding. For example, a certain number of students are missing in a class. We find that this is because one of the student has thrown a party and all her friends who are also students in the class attend the party. 

Identifying the type of missing data and which category it belongs to helps us follow a methodology to address the issue. 

Let's look at an example from each of the categories. The `diabetes` data shows values missing at random for BMI and Glucose. 

```python
msno.matrix(diabetes)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209125519126.png" alt="image-20201209125519126" style="zoom:80%;" />

We see an example of MCAR for `Glucose` and `BMI`. 

We see an example of MAR in the `Diastolic_BP`. 

We see an example of MNAR between the `Skin_Fold` and `Serum_Insulin`. 

### Finding Patterns in Missing Data

In order to look for patterns in missing data we will use the following tools: 

*   Heatmap - Describes a correlation between variables
*   Dendograms - Groups similar groups in close branches

```python
import missingno as msno
diabetes = pd.read_csv('pima-indians-diabetes data.csv')
msno.heatmap(diabetes)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209131315905.png" alt="image-20201209131315905" style="zoom:80%;" />

We see that there is a high correlation between missing values between certain variables. From the shades of blue we see that the relationships are more positive than negative. 

The dendogram plot can be made in the following way: 

```python
msno.dendrogram(diabetes)
```

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209132831234.png" alt="image-20201209132831234" style="zoom:80%;" />

Here we see the height of the tree. The deeper the node is, higher is the correlation. We see that `Skin_Fold` and `Serum_Insulin` are highly correlated as their value is 12. 

### Visualing Missing Across A Variable

Consider the following plot of `BMI` and `Serum Insulin`. Consider the following plot: 

<img src="Dealing%20With%20Data%20Missing%20Notes.assets/image-20201209164848251.png" alt="image-20201209164848251" style="zoom:80%;" />

We see that the null values span BMI values a lot more than those that span Seum Insulin. We also see that there is no correlation between BMI and Serum Insulin. 

This plot is created in the following way: 

```python
def fill_dummy_values(df, scaling_factor=0.075):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
    # Calculate column range
    col_range = col.max() - col.min()
    # Scale the random values to scaling_factor times col_range
    dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()
    col[col_null] = dummy_values
  return df_dummy
```

When we plot any data, matplotlib removes the null values and only plot the null values. Therefore, in order to force matplotlib, we created the above function. 

```python
# Fill dummy values in diabetes_dummy
diabetes_dummy = fill_dummy_values(diabetes)

# Sum the nullity of Skin_Fold and BMI
nullity = diabetes.Serum_Insulin.isnull() + diabetes.BMI.isnull()

# Create a scatter plot of Skin Fold and BMI 
diabetes_dummy.plot(x='Serum_Insulin', y='BMI', kind='scatter', alpha=0.5, 
                    
                    # Set color to nullity of BMI and Skin_Fold
                    c=nullity, 
                    cmap='rainbow')

plt.show()
```

### When and How to Delete Missing Data

When it comes to deleting the missing data, there are two types: 

*   **Pairwise Deletion** - involves skipping the values for a given variable that is null. 
*   **Listwise Deletion** - involves skipping the row if there is any null value across the columns

>   This deletion process is only used when the values are MCAR

Pairwise deletion is preferred as it does not delete/skip a large amount of data while the listwise deletion does. When the number of values that are missing are small compared to the total amount of data, listwise deletion can be used. 

## Chapter 3: Imputation Techniques

Embark on the world of data imputation! In this chapter, you will apply basic imputation techniques to fill in missing data and visualize your imputations to be able to evaluate your imputations' performance.

The most common imputations is done using the **mean, median** and **mode**. Let's see how this can be performed: 

```python
from sklearn.impute import SimpleImputer
diabetes_mean = diabetes.copy(deep=True)

# Create the mean imputer object
mean_imputer = SimpleImputer(strategy='mean')

# Impute the missing values
diabetes_mean.iloc[:,:] = mean_imputer.fit_transform(diabetes_mean)
```

To use the median, you simply need to set the `strategy` to `'median'` on line 5. For mode, we set the `strategy='most_frequent'. ` For a constant value such as `-999`, you can set the following: `strategy='constant', fill_value=-999`. 

The above plot after imputation using mean looks something like this: 

![image-20201211093950255](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211093950255.png)

### Imputing Time Series Data

For imputation time series data, let's look at the Airquality dataset. We saw that `Ozone` had the most number of missing values. 

To fill the NaN values in this dataset we will explore the `.ffill()` method in pandas. The filling methods are of two types:

*    **backfill:**`bfill` -  The back fill replaces the NaNs with values that come AFTER the NaNs. 

    ```python
    airquality.fillna(method='bfill', inplace=True)
    ```

    ![image-20201211100433423](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211100433423.png)

*   **fowardfill**: `ffill` - The forward fill replaces the NaNs with the values that come BEFORE the NaNs. This is done in the following way: 

    ```python
    airquality.fillna(method='ffill', inplace=True)
    ```

     This is what the forward fill does: 
    ![image-20201211100256021](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211100256021.png)

Those these are great alternatives but when it comes to time series data, the `.interpolate()` method is the most appropriate to fill missing data. The interpolate method has the following strategies to fill NaNs: 

*   **Linear:** This method fits a line between non-missing values and replaces the missing values with values from the fit line. 

    ```python
    df.interpolate(method='linear', inplace=True)
    ```

    ![image-20201211100800102](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211100800102.png)
    And here's how the output looks like: 
    ![image-20201211100843928](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211100843928.png)

*   **Quadratic:** In this method a quadratic function is fit to the data and the missing values are imputed using the values from the quadratic function: 

    ```python
    df.interpolate(method='quadratic', inplace=True)
    ```

    ![image-20201211101040230](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101040230.png)
    ![image-20201211101100596](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101100596.png)

*   **Nearest**: This is a combination of `bfill` and `ffill`. The imputation is done based on the nearest value. 

    ```python
    df.interpolate(method='nearest', inplace=True)
    ```

    ![image-20201211101236645](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101236645.png)
    ![image-20201211101258747](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101258747.png)

### Visualizing Time-series Imputations

Here's how we can visualize: 

```python
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30,5))
```

![image-20201211101623098](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101623098.png)

Let's apply the imputation technique and see how it looks: 

```python
ffill_imp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30,5))
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30,5))
```

Here `fill_imp` is the imputed dataframe of airquality using forward fill. 

![image-20201211101818658](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211101818658.png)

Like this we can visualize decide which imputation technique looks good. We can create multiple plots and see all techniques and decide which looks better: 

```python
# Set nrows to 3 and ncols to 1
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

# Create a dictionary of interpolations
interpolations = {'Linear Interpolation': linear, 'Quadratic Interpolation': quadratic, 
                  'Nearest Interpolation': nearest}

# Loop over axes and interpolations
for ax, df_key in zip(axes, interpolations):
  # Select and also set the title for a DataFrame
  interpolations[df_key].Ozone.plot(color='red', marker='o', 
                                 linestyle='dotted', ax=ax)
  airquality.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)
  
plt.show()
```

![image-20201211102643752](Dealing%20With%20Data%20Missing%20Notes.assets/image-20201211102643752.png)

## Advanced Imputation Techniques

Finally, go beyond simple imputation techniques and make the most of your dataset by using advanced imputation techniques that rely on machine learning models, to be able to accurately impute and evaluate your missing data. You will be using methods such as KNN and MICE in order to get the most out of your missing data!

The **k-Nearest Neighbors (KNN)** imputation technique uses the closest $k$  observations to the null value to determine its imputed value. 

```python
from fancyimpute import KNN
knn_imputer = KNN()
diabetes_knn = diabetes.copy(deep=True)
diabetes_knn.iloc[:, :] = knn.imputer.fit_transform(diabetes_knn)
```

The **Multiple Imputation by Chained Equation (MICE)** performs multiple regressions over random sample of the data. Then it takes the average of the multiple regression values and uses that to fill in the value. MICE is a very robust for imputing data. 

```python
from fancyimpute import IterativeImputer
MICE_imputer = IterativeImputer()
diabetes_MICE = diabetes.copy(deep=True)
diabetes_MICE.iloc[:, :] = MICE_imputer.fit_transform(diabetes_MICE)
```

### Imputing Categorical Values

So far we have imputed numerical values. However, we can do imputation on categorical values too. But we need to first encode the categorical values before they can be imputed. We make use of *one-hot-encoding* to convert categorical to numerical values or use *ordinal encoder*. Next, we impute the NaN value with the "most frequent category". 

```python
from sklearn.preprocessing import OrdinalEncoder

# Instantiate the object: 
ambience_ord_enc = OrdinalEncoder()

# Select non-null values in the ambience dataset
ambience = users['ambience']
ambience_not_null = ambience[ambience.notnull()]
reshaped_vals = ambience_not_null.values.reshape(-1,1)

# Encode the non-null values of ambience
encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)
```





