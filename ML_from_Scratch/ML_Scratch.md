

# ML From Scratch

This is a list of algorithms that can coded without the use of python's Sci-kit learn package. 

## Train-Test Split

**Concept**: The main motivation behind the train-test split is randomize the observations so that the ML model does not find a pattern that does not exist. 

**Actual**: The train-test split makes use of the `random` module that randomly assigns observations to a test dataset. The remaining observations are then assigned to a train dataset. 

**The Code**

```python
def train_test_split(X, y, test_size, random_state=-1):
    """
    The function takes a dataframe and splits it into
    a training set and a test set. The proportion of the split
    is determined by the test_size
    
    Input:
    ------
    	X - pandas observations dataset
    	y - pandas label datase
    	random_state - ensures the results are reproducible
    	
    	test_size - a percentage of data that will be assigned to the test set
    	
   	Output:
   	------
   		X_train - pandas data frame containing training observations
   		y_train - pandas data frame containing training labels
   		y_test -  pandas data frame containing test labels
        X_test -  pandas data frame containing test observations
    """
    indices = X.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)
    return X_train, X_test, y_train, y_test
    
```





## Decision Tree

