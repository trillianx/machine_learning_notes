[TOC]



# Machine Learning Algorithms

These are my notes on the definition on each of the common machine learning algorithms. 

## Nonlinear Algorithms

Having gone through a list of linear algorithms, we now expand our definition to nonlinear algorithms. 

### Decision Trees

Decision trees are both used for regression and classifications. The classification and regression of trees come under an umbrella called **CART**. 

#### Definition

Decision tree, in the classification setting, is an algorithm that creates decision boundaries (or split the feature space using decision boundaries) such that each region only contains observations that belong to a single class. 

#### Metrics Used

Decision trees make use of one of the two metrics, the **gini impurity** or **entropy**. Both these metrics are high when the a decision tree's has observations that belong to more than 1 class. When the observations belong to a single class, the gini impurity of entropy of that node is 0. 

#### Predictions

The predictions are simply made by following the decisions in a given decision tree. Typically, the leaf node is a decision tree gives the prediction for an given unknown instance.

*   **Classification setting:** the prediction is that of a class in a given leaf node that has the highest probability. 
*   **Regression setting:** the prediction is the mean of all the observations in the leaf node.

#### Weakness

Decision trees tend to overfit the data if regularization is not introduced. The overfitting causes them to have high variance. 



### Naive Bayes



## Metrics in Machine Learning

 Here is a list of metrics that are used in machine learning. 

### Gini Impurity

