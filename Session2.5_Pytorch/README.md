# Pytorch 2. Session assignment
## Assignment question

```
1. Write a neural network that can:
    1. take 2 inputs:
        1. an image from the MNIST dataset (say 5), and
        2. a random number between 0 and 9, (say 7)
2. and gives two outputs:
    1. the "number" that was represented by the MNIST image (predict 5), and
    2. the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)
3. you can mix fully connected layers and convolution layers
4. you can use one-hot encoding to represent the random number input as well as the "summed" output.
    1. Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
    2. Sum (13) can be represented as: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0


```



## Data Preparation
```
MNIST Dataset contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels
We transform images to Tensors since pytorch deals with Tensor data
```

## Data Generation Strategy
1. We created a one-hot encoding for random number (0-9) and added to MNIST dataset
2. We created Train and Test data loader with batch size of 256
```
class MNISTRandomDataset(Dataset):
  def __init__(self, MNIST_dataset):
    self.MNIST_dataset = MNIST_dataset

  def __getitem__(self, index):
    image, label = self.MNIST_dataset[index]
    randNum = random.randint(0,9)

    #Creating one hot encoding for random number 
    one_hot_randNumber = F.one_hot(torch.arange(0, 10)) # used from https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function

    #add actual label and random number
    sum_label = label + randNum
    return image, label, one_hot_randNumber[randNum], sum_label

  def __len__(self):
    return len(self.MNIST_dataset)
```

## Custom Neural Network written
The Input of one-hot encoding of Random numbers are connected to FC1 with 20 features are input i.e MNIST ( 10 features) and Random Numbers (0 -9)
In the final output we get two output 
1. MNIST detection output
2. SUM of random number output
```
Network(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (max_pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (max_pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv7): Conv2d(128, 10, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=20, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=19, bias=True)
)
The model has 220,125 trainable parameters
```

## Device Selection 
1. The neural netowrk is train on GPU only
```
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
```
**cuda**

## Why Cross_Entropy loss function is used
**Cross-Entropy Loss Function**
Also called logarithmic loss, log loss or logistic loss. Each predicted class probability is compared to the actual class desired output 0 or 1 and a score/loss is calculated that penalizes the probability based on how far it is from the actual expected value. The penalty is logarithmic in nature yielding a large score for large differences close to 1 and small score for small differences tending to 0.

Cross-entropy loss is used when adjusting model weights during training. The aim is to minimize the loss, i.e, the smaller the loss the better the model. A perfect model has a cross-entropy loss of 0.

[!Cross_entropy]https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e

**Loss = (MNIST Cross_entropy Loss + SUM cross_entropy Loss)/2**

## results you finally got and how did you evaluate your results
-  We are calculating the average loss function for train parameters and test parameters
-  Lower the loss values higher is the accuracy



## Sample Training Log
```
Epoch 1 : 
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Train set: Average loss: 1.0962
Test set: Average loss: 0.004,
 MNIST Accuracy:98.19, 
Addition_Accuracy:42.77

Epoch 2 : 
Train set: Average loss: 0.5098
Test set: Average loss: 0.002,
 MNIST Accuracy:99.11, 
Addition_Accuracy:94.75

Epoch 3 : 
Train set: Average loss: 0.1372
Test set: Average loss: 0.001,
 MNIST Accuracy:99.22, 
Addition_Accuracy:98.87

Epoch 4 : 
Train set: Average loss: 0.0565
Test set: Average loss: 0.000,
 MNIST Accuracy:99.35, 
Addition_Accuracy:99.07

Epoch 5 : 
Train set: Average loss: 0.0401
Test set: Average loss: 0.000,
 MNIST Accuracy:99.46, 
Addition_Accuracy:99.27

```

# Train,test accuracy vs epochs

``` 
Need to add Image of plot

```