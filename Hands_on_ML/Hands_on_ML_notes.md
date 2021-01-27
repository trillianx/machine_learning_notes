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

#### Supervised /Unsupervised Learning 

In supervised learning, the samples and associated labels are provided to the ML system. The ML system find patterns between the samples and the labels so that it is able to make predictions of the labels based on only the sample data. If the label is a category, then the supervised ML system is called a **classification** while if the label is numerical, the ML system is called **regression**. Here are some important supervised learning algorithms that we cover: 

*   k-Nearest Neighbors
*   Linear Regression
*   Logistic Regression
*   Support Vector Machines
*   Decision Trees & Random Forest
*   Neural Networks

In the case of unsupervised learning, the training set has samples but no associated labels. The ML system then has to find 