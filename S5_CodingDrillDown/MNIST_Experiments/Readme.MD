# MNIST Data set different eperiments

## 1. Basic Setup

### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training  & Test Loop

### Results:
- Parameters: 6.3M
- Best Training Accuracy: 99.93
- Best Test Accuracy: 99.28

### Analysis:
- Extremely Heavy Model for such a problem
- Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step


## 2. Basic Skeleton

### Target:

- Get the basic skeleton interms of convolution and placement of transition blocks (max pooling, 1x1's)
- Reduce the number of parameters as low as possible

### Results:
- Parameters: 48,180
- Best Training Accuracy: 99.50
- Best Test Accuracy: 98.87

### Analysis:
- We have structured our model in a readable way
- The model is lighter with less number of parameters 
- The performace is reduced compared to previous models. Since we have reduced model capacity, this is expected, the model has capability to learn.   
- Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.


## 3. Batch-Normalization

### Target:

- Add Batch-norm to increase model efficiency.

### Results:
-   Parameters: 10970
-   Best Train Accuracy: 99.73%
-   Best Test Accuracy: 99.15%

### Analysis:
-   There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer.
-   Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.


## 4. Dropout and GAP followed by a Fully connected Layer

### Target:

- Add Dropout in each layer , FC layers after Avg GAP layer  to increase model efficiency.

### Results:


-   Parameters: 7496
-   Best Train Accuracy: 99.18%
-   Best Test Accuracy: 99.33%

### Analysis:


-   There is slight decrease in the number of parameters, also accuracy decreased while we add dropout at each layer and used GAP before a convolution layer 
-   Model problem is rectified to an extent we got train accuracy to 99.18% whereas the test accuracy is 99.33%. But, we have not reached the target test accuracy 99.40%.


## 5. Adding Data-Agumentation

### Target:

- To achive the better results with data augmentation techniques.

### Results:

-   Parameters: 7496
-   Best Train Accuracy: 99.01%
-   Best Test Accuracy: 99.37%

### Analysis:

-   The number of parameters remains same but added random rotation to train data.
-   Model overfitting problem is rectified to an extent.
-   Adding learning rate schedulers can help to improve the training, as model update weights with step of a particular learning rate so there is a chance that model may be stuck at the local minima which can be solved by incrersing or decresing the learning rate.
