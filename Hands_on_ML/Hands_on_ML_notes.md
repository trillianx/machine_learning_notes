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



