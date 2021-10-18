# Training MNIST Dataset with Neural Network with less than 20k parameters and achieve greater than 99.4% accuracy.

MNIST dataset is used to create a NN with less than 20k parameters and achieve 99.4% accuracy under 20 epoches

## Parameters used and Achieved accuracy
```
The model has 19,712 trainable parameters and achieved accurancy 99.4%
```

## Neural Network built

```
Network(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.1, inplace=False)
    (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (5): ReLU()
    (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (8): ReLU()
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Dropout2d(p=0.1, inplace=False)
  )
  (transition_layer): Sequential(
    (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.1, inplace=False)
    (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): ReLU()
    (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout2d(p=0.1, inplace=False)
    (8): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (9): ReLU()
    (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout2d(p=0.1, inplace=False)
    (12): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (13): ReLU()
    (14): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): Dropout2d(p=0.1, inplace=False)
  )
  (conv_final): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (gap): AvgPool2d(kernel_size=5, stride=5, padding=0)
)
The model has 19,712 trainable parameters



```

## Training Log


## Final Review
Total we used 19712 parameters, with model size 1.35 MB and achieved 99.44% at epoch 8


