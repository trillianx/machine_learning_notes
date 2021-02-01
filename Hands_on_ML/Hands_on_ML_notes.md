[TOC]



# Notes on Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

These are my notes on ML from the second edition of this book. The notes also include theory where it is missing in the book. My notes follow each of the chapters in the book as is but also inlcude a summary of the chapter, additional material to support the chapter, and python code that is clearly listed out. 

## Chapter 1: The ML Landscape

Let's start by defining what Machine Learning (ML) is: 

>   Machine learning is the science (and art) of programming computers so they can learn from data. 

This is the simplest definition of ML. An example of a simple ML program is the spam filter. It learns from the data and is then able to identify spam emails by itself without being explicitly programmed. 

A more mathematical definition of ML is: 

>   A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E

A machine learning program makes uses of data to be trained. This data is called a **training set**. A training set is composed of **training instances** or **samples**. The ML program is designed for a task T (e.g. flag spam emails). It learns this from the training set and builds the experience, E from these samples. The program's performance, P is then evaluated through a measure of accuracy metric with the use of another data set called the **test set**. The test set contains **test instances** that the ML program has never seen. 

### Why Use ML? 

Here are reasons why ML is important: 

1.   **ML programs automatically adapting to changing data**. Rather than being explicitly program, ML programs adapt as the training data changes. This helps to make the programs shorter and saves us a lot of time from writing code. 
2.  **ML programs allow us to find insights that we could normally miss**. ML algorithms work with patterns. When a ton of data that the ML programs can go through, they have the ability to find patterns that we, humans, can easily miss.  Moreover, we don't have the time or the ability to go through a large amount of data and look for patterns. The ML program helps us do that for us. 
3.  **ML programs help to solve Complex Problems**. Some problems do not have a direct solution that we can code. In fact, there are a lot of problems that do not have a mathematical solution. In such cases, the ML program help us solve such problems. 

### Types of ML Systems

ML programs can be broadly put into three categories: 

1.  **Systems that require or don't require human interactions** - Systems that require human supervision called are **supervised ** or **unsupervised** ML systems. These include: 
    1.  Supervised
    2.  Semi-supervised
    3.  Unsupervised
    4.  Reinforcement Learning
2.  **Systems that learn incremenatally or on the fly** - Such systems learn as the data flows through them. Such systems include: 
    1.  Online Learning
    2.  Batch Learning
3.  **Systems that use instance-based or model-based learning** - Such systems either use the new information by comparing with what they have already learned or they create a model of the system and then use the new information to make predictions. 

It is to be noted that a ML system will not exclusively fall into these three buckets. Most often they include a combination of one or two buckets. 

Let's look at each of these in greater detail: 

#### Supervised & Unsupervised Learning 

In supervised learning, the samples and associated labels are provided to the ML system. The ML system find patterns between the samples and the labels so that it is able to make predictions of the labels based on only the sample data. If the label is a category, then the supervised ML system is called a **classification** while if the label is numerical, the ML system is called **regression**. Here are some important supervised learning algorithms that we cover: 

*   k-Nearest Neighbors
*   Linear Regression
*   Logistic Regression
*   Support Vector Machines
*   Decision Trees & Random Forest
*   Neural Networks

In the case of unsupervised learning, the training set has samples but no associated labels. The ML system then learns from the data to create categories or make predictions. Some important unsupervised learning algorithms we cover are: 

*   K-Means
*   DBSCAN
*   Hierarchical Cluster Analysis
*   Anomaly Detection
    *   One-class SVM
    *   Isolation Forest
*   Visualization & Dimensionality Reduction
    *   Principle Component Analysis (PCA)
    *   Kernel PCA
    *   Locally Linear Embedding
    *   t-Distributed Stochastic Neighbor Embedding (t-SNE)
*   Association Rule Learning
    *   Apriori
    *   Eclat

#### Semi-Supervised & Reinforcement Learning

**Semi-supervised learning systems** work best when when we have a partially labeled dataset. For example, you may have a large collection of photos. Now a ML system is able to tag you in all of the photos in which you are there if you can tag yourself in 1% or less of the photos. 

Semi-supervised learning is a combination of supervised and unsupervised learning. Some examples are: 

*   Deep Belief Networks (DBNs)
*   Restricted Boltzmann Machines (RBMs)

In **Reinforcement Learning systems** we have an agent. This agent can observe the environment, select and perform actions, and get rewards in return or negative rewards (punishment). The agent must learn by itself and devise a strategy, called a policy, to get most rewards over time. The policy defines what action the agent should choose when it it is in a given situation. 

#### Batch and Online Learning

**Batch learning** is when a system learns from training data that is fed to it in batches. At times if the data is too large for a system to handle due to restricted resources, breaking the data into smaller batches allows the system to continually learn. The disadvantage of batch learning is that when the learning takes place, the system goes into offline mode. This is known as **offline learning**.

If system is required to be continuously online, then **online learning ** is the best option. This sytem learns sequentially when it fed with small groups of data called **mini batches**. Each learning step is fast and cheap and more importantly the system does not need to be offline. The drawback of online learning is that such systems adapt quickly, called **learning rate**, to the data that is fed. So, if the data that is fed to an online system is bad, the system will train on bad data and make bad predictions. So, care must be taken to limit feeding bad data to online systems. 

#### Instance-based vs Model-Based Learning

In **instance-based learning**, the system learns the patterns from the training data and sort of by hearts all the information it has learned. When a new instance is provided, it simply compares the closest instance to that new instance and make predictions. An example of instance-based in KNN. The KNN simply learns what each of the instances in the training examples are labled. When a new instance is given, it find the closest labeled instances and labels the new instance based on majority rule. 

In **model-based learning** the system generalizes the pattern from the training examples and then uses that information to make predictions on the new instance it has been provided. An example of model-based would be SVM, which creates a boundary between classes. The new instance is labeled a category based on where it falls. 

### Main Challenges of ML

There are two things that can go wrong: bad data and bad algorithm. 

#### Insufficient Quantity of Training Data

ML systems require a lot more data than humans to either predict or classify accurately. If we have less data, then the ML systems are as good as the data they are fed. 

#### Nonrepresentative Training Data

Care must be taken to select data that will ultimately help the ML system to accurately make predictions or inference. If the data are nonrepresentative, the ML system will be trained badly and will be a poor system. 

When the training sample is too small, it is said to have high **sampling noise**. Of course, large sample that is not representative can also have sampling noise. Large nonrepresentative data are said to have **sampling bias**.

#### Poor Quality Data

Data that has a lot of missing values across features or outliers is said to be of poor quality. Such poor quality of data will result in poor performance of any ML system. So, care must be taken to clean the data before passing it through any ML system. 

#### Irrelvant Features

A ML system learns best when it is given a good set of features to train on. By good we mean relevant features that will help it to be more accurate. The process of creating good, relevant features that are passed to a ML system is called **feature engineering**. Feature engineering involves the following steps: 

*   **Feature selection** - Selection fo relevant features to be used for training the model
*   **Feature extraction** - Combining existing features to produce a more useful feature. This also helps to reduce the dimensionality of the feature space. 
*   **Creating new Features by acquiring more data** - Once the problem is known and we find that the current features do not aid in good accuracy, we may choose to gather more relevant data with featuresthat will help. 

#### Overfitting the Training Data

It is easy for an ML system to overfit the data. **Overfitting** means when an ML system fits the noise in the data rather than just the data. The symptoms of overfitting is close to 100% accuracy of the model on training dataset but poor accuracy on test dataset. Overfitting can also occur when a more complex model is used instead of a simple model for the given dataset. 

Here are steps taken to avoid overfitting: 

*   Use a simple model
*   Gather more data
*   Reduce noise in the training data
*   Constrain the model through regularization

#### Underfitting the Training Data

Underfitting is the opposite of overfitting. Underfitting occurs when the model used is too simple.   Underfitting results in poor training accuracy. Here are some reasons for why we may have underfitting:

*   Select a more powerful model with more parameters or choose a non-parametric model
*   Feed better features to the learning algorithm
*   Reduce the constrains through regularization so the model is more flexible

### Testing & Validation

Once the model is trained, it 's accuracy is evaluated by running it on a **test dataset**. We also evaluate the error rate of the model on the test set. This error is called **generalization error (or out of sample error** or **test error rate**. Here' s a scenario that can happen: 

*   If the training error is low but the test error rate is high, the model is overfitting on the training data

More often than not, when working on a project, we work on multiple ML systems. We then need to tune each of these model parameters called **hypertuning** and evaluate the best model, which is called, **model selection**.

The hypertuning is used to tune the parameters of a given model so that we can increase the test accuracy or reduce the generalization error. As we will see later, we can construct a list of hypertuning parameters and run of model through the data and find which combination of hypertuning parameters work best. 

Generalization error is determined through a validation set, like a test set. However, a test set is a single instance of the process to get us the result. We cannot be completely sure that the result we get from this is the actual accuracy of the model. Also, at times the validation set could be too small which may doubt our results. Hence, we make use of **cross-validation**. 

The cross-validation is a technique in which a dataset is broken into $k$ groups. One group is used for validation while the $k-1$ groups are used for training. This process is repeated across all $k$ groups. This results in a accuracy score along with the standard deviation of the score. This gives a good picture how the model performs. 

>   When performing validation, it is important that the validation set is representative of the training set. If this is not the case, we have a **mismatch** between the two sets. The model will perform badly on the validation set. 



## Chapter 2: End-to-End Machine Learning Project 

In this chapter we will go through a project from the beginning to the end. This will illustrate how a typical machine learning project works in practice.

Here are the main steps to follow when working on a ML project: 

1.  Look at the big picture
2.  Get the data
3.  Discover and Visualize the data to gain insights
4.  Prepare data for ML algorithms
5.  Select a model and train it
6.  Fine-tune your model
7.  Present your solution
8.  Launch, monitor, and maintain your system

As an example to go through these steps, we will look at the housing prices. 

### Look at the Big Picture

We are given a data of California housing prices. The data contains population, median income, median housing price for each block and other geographical information. **Our goal is to predict the median housing price in any district given all the other metrics.** 

Before we launch into building a model, we need to ask our manager or boss about how this model is going to be used. Is this a stand-alone model or would this be part of a pipeline. We also need to know how accurate this model should be. How critical is the accuracy of the model and how would the prediction from the model be used. All these questions help us decide how we build and test the model. 

We find out that our model is just a piece in the company's pipeline. It would look something like this: 

<img src="Hands_on_ML_notes.assets/image-20210127140738743.png" alt="image-20210127140738743" style="zoom:100%;" />

>   A sequence of data processing components is called a **data pipeline**. 

We have been asked to make predictions on the housing prices at a given location. So, we are looking at a numerical output. So, this is a regression problem as opposed to a classification problem. We know that there are a lot of features that we will use to make the prediction so this is a multivariate problem. So, this is a **multivariate regression problem**. We also know that we have labeled data. So we will be working with a **supervised learning** ML system. 

Next, we find that the data are small enough to fit in memory. So, we are looking at batch learning as opposed to online learning. 

>   If the data are large, you can break it up using MapReduce technique, and then use the batch learning method. 

#### Select Performance Measure

Now that we know the type of ML system to use, we need to decide what metric we should use to evaluate the model performance. Knowing that this is a regression problem, the typical metric to use the **Root Mean Square Error (RMSE)**. The RMSE is given by: 

![image-20210127141627183](Hands_on_ML_notes.assets/image-20210127141627183.png)

Here the $\bold{X}$ is the feature matrix and $h$ is the prediction matrix ($h = \hat{y}^{(i)} = h(\bold{x}^{(i)})$). The other $y^{(i)}$ is the target or the label for a given sample $i$, while $m$ are the total number of samples or observations. 

Ofen in linear regresssion we use the **mean absolute error (MAE)** instead as it is robust against outliers. The MAE is given by, 

![image-20210127142149792](Hands_on_ML_notes.assets/image-20210127142149792.png) 

Both RMSE and MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target values. 

#### Check the Assumptions

When working with any ML system, it is important to know what the assumptions are. There are two types of assumptions that we make:

1.  Assumptions involving the ML system. Each ML algorithm requires the data to be in certain type and format. We need to ensure that the assumptions are satisfied. If not, we will get erroneous decision. We will work on this when we prepare the data for ML algorithms. 
2.  Assumptions regarding what the format of the output of the ML system is and how it is going to be used. We find that the downstream system takes in actual prices and not the price categories. 

### Getting the Data

When working on a project, it is important to create a new environment so that other projects are not affected. So, we begin by creating an environment:

```python
python3 -m pip3 install --user --U virtualenv
virtualenv my_env
source my_env/bin/activate
```

We then install all python packages in that environment. 

#### Take a Quick Look at the Data Structure

We can do a quick EDA to see how the data looks like. This includes:

*    Use of `.head()` to see the data structure
*   Use of of `.info()` to find which features have null values
*   Use of `.value_counts()` to find the categories in a categorical variable
*   Use of `.describe()` to see the descriptive statistics of numerical features
*   Use of `.hist()` to see the distribution of numerical values

#### Creating a Test Set

Now that we know how the data looks like. We first start with a test set. This is important because we do not want to create a test data after we have done EDA on the data. We do not want to be biased by it. 

The test split is creating using the scikit-learn algorithm `train_test_split()`:

```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

The above function randomly selects samples from the `housing` dataset and assigns 20% to test and the remaining to the train. However, there are times when you want to preserve the distribution of features of the original dataset within both the train and test. In such a case, we will do a **stratefied sampling**:

```python
from sklearn.model_selection import StratifiedShuffleSplit
strat_splits = StratifiedShuffleSplit(n_split = 1, 
            	                  test_size=0.2,
                	              random_state=42)
for train_index, test_index in strat_splits.split(housing, 
                                                  housing['income_category']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Finally we remove the `income_cat` attribute: 
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
```

### Discover & Visualize the Data to Gain Insights

To gain insights into the data, we begin by creating multiple plots. 

#### Visualize the Data

Here we begin by plotting the data. 

```python
housing.plot(kind='scatter', x='longitude', y='latitude')
```

![image-20210127145720173](Hands_on_ML_notes.assets/image-20210127145720173.png)

We can also create a heatmap to see where the housing prices are high. These would be expensive places: 

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
```

The result is the following: 

![image-20210127145855954](Hands_on_ML_notes.assets/image-20210127145855954.png)

We see that the housing prices are very high next to the coast but much cheaper inland, as expected. 

#### Looking for Correlation

If the data is not too large, we can create correlation matrix. 

```python
corr_matrix = housing.corr()
```

Lookinag at this matrix, for one of the variable, `median_housing_value`, we find:

```python
print(corr_matrix["median_house_value"].sort_values(ascending=False))

median_house_value    1.000000
median_income         0.687170
total_rooms           0.135231
housing_median_age    0.114220
households            0.064702
total_bedrooms        0.047865
population           -0.026699
longitude            -0.047279
latitude             -0.142826
Name: median_house_value, dtype: float64”

```

 We notice that `median_income` is high correlated with `median_house_value`. The other features are less so. Some are weakly anticorrelated. 

We can also plot some correlations as well, 

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

![image-20210127150351943](Hands_on_ML_notes.assets/image-20210127150351943.png)

We see that some are strongly correlated but we notice something vety interesting: 

![image-20210127150428843](Hands_on_ML_notes.assets/image-20210127150428843.png)

Notice some data are exactly at 500000. This is most likely due to a capping of the data. We may need to remove some district that have these values capped to prevent the algorithm from learning to reproduce these data quirks. 

#### Working with Attribute Combinations

You will notice that some distributions are tail heavy. We may need to transform them so that the distributions are normal. This is important because the multiple linear regression assumes that the attributes are normally distributed. This is where our assumptions about the ML system comes into picture. 

You also want to explore the features to see if there are features that can be removed or which can be modified. We have a total number of rooms per district, which is not quite helpful. Instead, we want total number of rooms per household. Also, the total number of bedrooms are not useful. Instead, we want the total number of bedrooms in terms of the total number of rooms. We make this change

```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```

Looking at the correlation matrix for `median_house_value`, we find the following: 

```python
“corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

median_house_value          1.000000
median_income               0.687160
rooms_per_household         0.146285
total_rooms                 0.135097
housing_median_age          0.114110
households                  0.064506
total_bedrooms              0.047689
population_per_household   -0.021985
population                 -0.026920
longitude                  -0.047432
latitude                   -0.142724
bedrooms_per_room          -0.259984
Name: median_house_value, dtype: float64
```

We see that `rooms_per_household` is much more correlated with `median_house_value` rather simply the `total_rooms`. We also see that `bedrooms_per_room` is strongly anti correlated than simply `total_bedrooms`. 

>   The round of exploration does not have to be rigorous, instead it should be just enough to get started. 

### Prepare Data for ML Algorithms

The data preparation should be written as a pipeline rather than doing it manually. So, it is important to write functions instead. Such functions can then be used in future projects. 

#### Data Cleaning

As we saw earlier, our feature `total_bedrooms` has some missing values. We can either drop rows that have missing values, we can drop the feature entirely, or we can impute the missing values. 

We will use the impute method to impute the missing values. Here's how it is done: 

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
```

Now the imputer requires that the dataframe be purely numerical. So, we will get rid of all categorical features and then pass this to the imputer: 

```python
housing_num = housing.drop('ocean_proximity', axis=1)
```

Now we are ready to impute: 

```python
imputer.fit(housing_num)
X = imputer.transform(housing_num)
```

This gives is X, which is a numpy ndarray. We can put this back into a dataframe: 

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

This takes care of all missing values across features. We have simply used a median value of that feature to be imputed. 

#### Handling Text & Categorical Attributes

For ML systems to use the data, the categorical features need to be numeric. Therefore, we need to transform them. The most common way to convert is to use `OneHotEncoder`:

```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat = cat_encoder.fit_transform(housing_cat)
```

>   The drawback of OneHotEncoding is that the resultant matrix is a sparse matrix. If this is the case, you should think about transforming the categorical feature into a numerical feature. For example, rather than using `ocean_proximity` as a categorical feature, you can use a distance measure from the ocean as a feature. Alternatively, you could replace each category with a learnable, low-dimensional vector called **embedding**. We will see this Chapter 13 and 17. 

#### Feature Scaling



## Chapter 3: Classification

We will begin our exploration of ML algorithms with classification. As an example, we will use the **MNIST** dataset. It has a set of 70,000 small images of digits that were written by highschool students. 

Scikit-learn provides this dataset: 

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# Get the X and y vectors
X, y = mnist['data'], mnist['target']
```

We can see their shapes: 

```python
print(X.shape)
print(y.shape)
```

```python
(70000, 784)
(70000,)
```

We see that there are 70,000 images and each image has `28 x 28` pixels, which amount to 784 features. 

Here's how the MNIST dataset looks like: 

<img src="Hands_on_ML_notes.assets/image-20210201101654656.png" alt="image-20210201101654656" style="zoom:80%;" />

Now let's split the dataset and get started with classification: 

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

### Binary Classifier

We will start with a very simple classifier. It is a binary classifier, in the sense the number we are looking at, say 5, is either true or false. For this we will use the **Stochastic Gradient Descent** classifier. The advantage of SGD is that it can easily handle very large datasets efficiently. This is because SGD deals with instances rather than large batches of data. Think of it as online learning vs batch learning. 

To make things easier, we will create a binary classifier that evaluate whether an image has a '5' in it or not. So, we will train the classifier on all the images of 5: 

```python
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```

This creates our labels, which are booleans. Now we train the classifier:

```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

Looking ath `y_test_5`, we know that the last two digits are False. 

```python
for index in [998, 999]:
    single_sample = X_test.iloc[index].values.reshape(1, -1)
    print(sgd_clf.predict(single_sample))
```

And we get: 

```python
[False]
[False]
```

This is rather painful way of evaluating the classifier. Let's automate the process. 

#### Evaluating the Classifier

We will implement cross-validation to evaluate the classifier. 

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring='accuracy')
```

Here we have done a k-fold cross validation with `k=5`. The result is the following: 

```python
array([0.88083333, 0.88325   , 0.88116667, 0.86625   , 0.8875    ])
```

We see that the accuracy on average is greater than 88%. Now before we get excited about the result, we should know that only 10% of the images are that of '5', so any dumb classifier will get over 90% of accuracy. **This suggests that accuracy is generally not the preferred performance measure for classifiers**, especially when the datasets are skewed. 

#### Confusion Matrix

A much better way to evalute the performance of a classifier is to use the confusion matrix. Given two classes A and B, the confusion matrix counts the number of times the classifier has: 

*   Correctly classifier numbers belonging to class A 
*   Correctly classified the number belonging to class B
*   Incorrectly classified the number belonging to class A
*   Incorrectly classified the number belonging to class B

Here's how we setup our confusion matrix: 

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

Just like `cross_val_score()`, the `cross_val_predict()` makes k-fold cross-validation but rather than returning the scores, it returns the predictions on each test fold. 

Now we are ready to construct our confusion matrix: 

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
```

The output is the following: 

```python
array([[53892,   687],
       [ 1891,  3530]])
```

Few things to know about confusion matrix as we see above: 

*   Each row corresponds to the actual class
*   Each column corresponds to predicted class. 

>   In a confusion matrix, we want higher numbers along the diagonal and smaller numbers along the off-diagonal for the classifier to be good. A perfect classifier will have 0 values in its off-diagonal

The confusion matrix gives us a lot of information but we may be interested in a summary of the result, a more concise metric. There are two such metrics that are often used to evaluate accuracy of a classifier. But before we go into the metrics, let's use some definitions: 

*   **True Positive (TP)** - Observations that are correctly predicted to belong to the positive class. 
*   **True Negative (TN)** - Observations that are correctly predicted to belong to the negative class
*   **False Positive (FP)** - Observations that actually belong to the negative class but are predicted to belong to the positive class
*   **False Negative (FN)** - Observations that actually belong to the positive class but are predicted to belong to the negative class. 

Here's an example to look at: 

<img src="Hands_on_ML_notes.assets/image-20210201114308553.png" alt="image-20210201114308553" style="zoom:150%;" />

With these definitions, let's look at some metrics that are often used to evalutate a classifier: 

*   **Precision** - This is the ratio of the total number of observation that were predicted correctly to be positive over the total number of observations predicted to be positive class (TP + FP). 
    $$
    \text{precision} = \frac{TP}{TP + FP}
    $$
    The drawback of using only precision is that if we make one prediction and it is correct, we get a 100% precision. Therefore, precision is used along another metric. 

*   **Recall** - This is the ratio of the total number of observations that are predicted correctly to be positive over the total number of actual observations that belong to the positive class (TP + FN) 
    $$
    \text{recall} = \frac{TP}{TP + FN}
    $$
    

*   **F1-score** - The F1-score is the harmonic mean of precision and recall. F1-score is particularly useful to compare two or more classifiers. The difference between mean and harmonic mean is that the latter gives more weight to low values. As a result, the classifier will only get a high F1-score if both recall and precision are high.

    $$
    F_1\text{ score} = 2 \times \frac{precision \times recall}{precision + recall}
    $$

Here is how you can do that in sklearn: 

```python
from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision: ", np.round(precision_score(y_train_5, y_train_pred),2))
print("Recall: ", np.round(recall_score(y_train_5, y_train_pred),2))
print("F1-score: ", np.round(f1_score(y_train_5, y_train_pred),2))
```

And here is the result: 

```python
Precision:  0.84
Recall:     0.65
F1-score:   0.73
```

>    The F1-score favors classifiers that have similar precision and recall

There are cases when this is not something we would want. In some cases, you may want higher precision while in other cases, you may want higher recall. Let's think of precision and recall in terms of a car alarm. We are in a bad neighborhood, so we get an alarm in our car in order to prevent it from a carjack. 

*   **When Precision Matters** - What does it mean to have high precision? We can get high precision, if the **False Positives** are smaller. Fewer False positives mean that few number of times the alarm rings when car is shaken by the wind or if a large truck passes close-by. 
*   **When Recall Matters** - What does it mean to have high recall? We can get high recall if the **False Negatives** are smaller. Few False negatives mean that a carjack happens when the alarm does not ring. 

This all boils down to **sensitivity** of our alarm. If we make the alarm less sensitive, we will have less of false positives but it will increase false negatives. In other words, we will not have the alarm go off every now and then, but we risk on it not going off when the carjack happens. On the other hand, if we increase the sensitivity, we will definitely catch the thief but we are also likely to have the alarm go off more times. 

Unfortunately, we cannot have high precision and high recall. This is because increasing precision reduces recall and vice versa. This is known as **precision-recall trade-off**

#### Precision/Recall Trade-off

A classifier makes use of a decision boundary. Based on where the observation is, in terms of the decision boundary, the classifier assigns it to one or the other class. For example, the images are ranked by the classifier score as follows: 

<img src="Hands_on_ML_notes.assets/image-20210201133600845.png" alt="image-20210201133600845" style="zoom:150%;" />

We have three positions of a decision boundary. Based on where the boundary is, we may have higher precision or higher recall. 

sklearn does not let you set the threshold or the decision boundary explicitly but gives you access to the decision scores that it uses to make predictions. Rather than using `.predict()`, if you used, `.decision_function()` method, you will get the score for that instance. 

```python
single_sample = X_test.iloc[996].values.reshape(1, -1)
sgd_clf.decision_function(single_sample)

array([-7861.86923293])
```

We see that the value is lot negative then if we used another value: 

```python
single_sample = X_test.iloc[999].values.reshape(1, -1)
sgd_clf.decision_function(single_sample)

array([-5838.15108052])
```

This is indicative that the decision boundary is set to zero. 

#### Setting up a Threshold

So, how do we decide what threshold to use? We follow these steps: 

1.  We get decision scores of all instances in our training set
2.  We use a `precision_recall_curve()` to compute precision and recall values for all possible thresholds or decision boundaries and not just at zero. 
3.  We plot the result. 

```python
# Get decision scores for all instances
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                            method='decision_function')

# Get precision and recall values for all thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plot to view the curves
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend()
plt.ylabel('Score')
plt.xlabel('Threshold')
```

<img src="Hands_on_ML_notes.assets/image-20210201135358885.png" alt="image-20210201135358885" style="zoom:150%;" />

For a given decision boundary (red) we can get the precision and recall scores. 

>   The precision curve is bumpier than recall curve. This happens because precision can go down even when the threshold is increased. Notice that the precision goes down when we move our threshold in the above numbers image when we go right. However, the recall will always increase or decrease. 

#### Precision - Recall Curve 

We could select a good precision recall value by simply plotting a recall and precision curve. 

```python
plt.plot(recalls, precisions)
plt.ylabel('Precision')
plt.xlabel('Recall')
```

<img src="Hands_on_ML_notes.assets/image-20210201140043691.png" alt="image-20210201140043691" style="zoom:100%;" />

If we want a 90% precision, we need to have a recall of about 40%. Ignore the bump, we see that if we wish to have a precision of 99%, our recall will be 5% or so. 

#### ROC Curve

The **Receiver Operating Characteristic (ROC)** curve is another common tool used for binary classifiers. Rather than using precision, we use  **true positive rate** (also known as **recall**) and **false positive rate**. The false positive rate is defined as `1 - True Negative Rate`. The TNR is the ratio of negative instances that are correctly classified as negative. The TNR is also called **specificity**. 

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
```

![image-20210201140904279](Hands_on_ML_notes.assets/image-20210201140904279.png)

We see again a trade-off. Higher the recall value (TPR), the higher will be the False Positive Rate. 

The False positive rate is calculated as, 
$$
\text{False Positive Rate} = \frac{FP}{FP + TN}
$$

>   The false positive rate is the probability that the true event will be missed by the test. In other words, the FPR is the probability of falsely rejecting the null hypothesis (i.e. making a Type I error). 

#### Area Under the Curve (AUC)

The ROC allows you to compare classifiers graphically. Rather than graphical representations, we can simply compute the area under the curve of the ROC. The closer the AUC is to 1, the better the classifier. 

```python
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_train_5, y_scores)
print(np.round(auc_score, 2))
```

We get a score of 0.96 or 96%. 

#### When to Use which Curve

*   Use the Precision Recall Curve whenever the positive class is rare or when you care more about the FP than the FN. 
*   Use the ROC curve otherwise

### Random Forest Classifier

Let's try using the Random Forest classifier for our problem. Note that Random Forest classifier does not have a `decision_function()` but has a `predict_proba()` method. 

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                   method='predict_proba')
```



