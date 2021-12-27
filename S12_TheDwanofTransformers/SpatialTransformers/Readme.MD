## Spatial Transformers

``
Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. 
Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model.
For example, it can crop a region of interest, scale and correct the orientation of an image.
It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations
``

![stn-arch](https://user-images.githubusercontent.com/90888045/147482155-373c914d-1133-46c7-b735-e4524d9258bd.png)


## Model Architecture

Spatial transformer networks boils down to three main components :

1. The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
2. The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
3. The sampler uses the parameters of the transformation and applies it to the input image.

## Notebooks Links
1. [Google colab link](https://colab.research.google.com/drive/1YtGMllGPzDFire-n7lqjSE7vKjjHH_3Q?usp=sharing)
2. [Notebook link](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S12_TheDwanofTransformers/SpatialTransformers/SpatialTransformers_CIFAR10.ipynb)

## Network for CIFAR10 and Spatial Transformer

```
Net(
  (conv1): Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=400, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=512, out_features=64, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=64, out_features=6, bias=True)
  )
)
```
## Training Logs
```
Train Epoch: 2 [0/50000 (0%)]	Loss: 1.808350
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.791269

Test set: Average loss: 1.6902, Accuracy: 3934/10000 (39%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.835937
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.658971

Test set: Average loss: 1.5851, Accuracy: 4288/10000 (43%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.712984
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.779117

Test set: Average loss: 1.5338, Accuracy: 4485/10000 (45%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.648149
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.574058

Test set: Average loss: 1.5362, Accuracy: 4533/10000 (45%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.566600
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.445429

Test set: Average loss: 1.4551, Accuracy: 4844/10000 (48%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.544824
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.470032

Test set: Average loss: 1.5145, Accuracy: 4588/10000 (46%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.639204
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.413089

Test set: Average loss: 1.4260, Accuracy: 4940/10000 (49%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.624146
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.538240

Test set: Average loss: 1.4083, Accuracy: 5043/10000 (50%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.454199
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.457172

Test set: Average loss: 1.3811, Accuracy: 5212/10000 (52%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.530526
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.597302

Test set: Average loss: 1.3699, Accuracy: 5339/10000 (53%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.542783
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.254032

Test set: Average loss: 1.3492, Accuracy: 5407/10000 (54%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.398004
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.344200

Test set: Average loss: 1.3481, Accuracy: 5444/10000 (54%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.360715
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.185049

Test set: Average loss: 1.3246, Accuracy: 5531/10000 (55%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.265211
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.452600

Test set: Average loss: 1.4369, Accuracy: 4971/10000 (50%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.479920
Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.176236

Test set: Average loss: 1.3492, Accuracy: 5447/10000 (54%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.415220
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.309864

Test set: Average loss: 1.3248, Accuracy: 5596/10000 (56%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.327458
Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.438377

Test set: Average loss: 1.3196, Accuracy: 5623/10000 (56%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 1.280071
Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.485934

Test set: Average loss: 1.3048, Accuracy: 5631/10000 (56%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 1.161076
Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.245271

Test set: Average loss: 1.3466, Accuracy: 5504/10000 (55%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 1.445119
Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.428747

Test set: Average loss: 1.2960, Accuracy: 5720/10000 (57%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 1.301713
Train Epoch: 22 [32000/50000 (64%)]	Loss: 1.153543

Test set: Average loss: 1.3283, Accuracy: 5558/10000 (56%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 1.211499
Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.467224

Test set: Average loss: 1.2791, Accuracy: 5849/10000 (58%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 1.298214
Train Epoch: 24 [32000/50000 (64%)]	Loss: 1.363718

Test set: Average loss: 1.3126, Accuracy: 5626/10000 (56%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 1.249705
Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.235647

Test set: Average loss: 1.4333, Accuracy: 5174/10000 (52%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 1.551670
Train Epoch: 26 [32000/50000 (64%)]	Loss: 1.227186

Test set: Average loss: 1.3070, Accuracy: 5704/10000 (57%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 1.467292
Train Epoch: 27 [32000/50000 (64%)]	Loss: 1.216343

Test set: Average loss: 1.2558, Accuracy: 5885/10000 (59%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 1.112918
Train Epoch: 28 [32000/50000 (64%)]	Loss: 1.166901

Test set: Average loss: 1.2825, Accuracy: 5806/10000 (58%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 1.265888
Train Epoch: 29 [32000/50000 (64%)]	Loss: 1.091688

Test set: Average loss: 1.2942, Accuracy: 5717/10000 (57%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 1.274439
Train Epoch: 30 [32000/50000 (64%)]	Loss: 1.199701

Test set: Average loss: 1.2784, Accuracy: 5773/10000 (58%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 1.179442
Train Epoch: 31 [32000/50000 (64%)]	Loss: 1.393703

Test set: Average loss: 1.2766, Accuracy: 5787/10000 (58%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 1.247020
Train Epoch: 32 [32000/50000 (64%)]	Loss: 1.166880

Test set: Average loss: 1.3173, Accuracy: 5592/10000 (56%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 1.487950
Train Epoch: 33 [32000/50000 (64%)]	Loss: 1.350139

Test set: Average loss: 1.2512, Accuracy: 5899/10000 (59%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 1.112786
Train Epoch: 34 [32000/50000 (64%)]	Loss: 1.414909

Test set: Average loss: 1.2245, Accuracy: 5963/10000 (60%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 1.140597
Train Epoch: 35 [32000/50000 (64%)]	Loss: 1.284468

Test set: Average loss: 1.2714, Accuracy: 5747/10000 (57%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 1.178042
Train Epoch: 36 [32000/50000 (64%)]	Loss: 1.012157

Test set: Average loss: 1.2618, Accuracy: 5809/10000 (58%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 1.209430
Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.967027

Test set: Average loss: 1.2299, Accuracy: 5934/10000 (59%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 1.311249
Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.267321

Test set: Average loss: 1.2355, Accuracy: 5845/10000 (58%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 1.341726
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.931414

Test set: Average loss: 1.2590, Accuracy: 5705/10000 (57%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 1.217106
Train Epoch: 40 [32000/50000 (64%)]	Loss: 1.335319

Test set: Average loss: 1.2817, Accuracy: 5730/10000 (57%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 1.036455
Train Epoch: 41 [32000/50000 (64%)]	Loss: 1.125590
![output](https://user-images.githubusercontent.com/90888045/147481785-4b2e8d50-f940-4f74-9523-a610d4ff50d2.png)

Test set: Average loss: 1.2208, Accuracy: 5995/10000 (60%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 1.019974
Train Epoch: 42 [32000/50000 (64%)]	Loss: 1.125357

Test set: Average loss: 1.2844, Accuracy: 5587/10000 (56%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 1.342727
Train Epoch: 43 [32000/50000 (64%)]	Loss: 1.099939

Test set: Average loss: 1.2403, Accuracy: 5858/10000 (59%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 0.937416
Train Epoch: 44 [32000/50000 (64%)]	Loss: 1.137147

Test set: Average loss: 1.2225, Accuracy: 5920/10000 (59%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 1.093160
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.922044

Test set: Average loss: 1.3624, Accuracy: 5332/10000 (53%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 1.572492
Train Epoch: 46 [32000/50000 (64%)]	Loss: 1.135669

Test set: Average loss: 1.1855, Accuracy: 6069/10000 (61%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 1.339075
Train Epoch: 47 [32000/50000 (64%)]	Loss: 1.577280

Test set: Average loss: 1.2301, Accuracy: 5902/10000 (59%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 1.148864
Train Epoch: 48 [32000/50000 (64%)]	Loss: 1.060968![output](https://user-images.githubusercontent.com/90888045/147481771-e1b06bed-5013-42aa-ac29-720a70b5a4bb.png)


Test set: Average loss: 1.1848, Accuracy: 6014/10000 (60%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 0.670103
Train Epoch: 49 [32000/50000 (64%)]	Loss: 1.378164

Test set: Average loss: 1.1981, Accuracy: 5943/10000 (59%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 1.342169
Train Epoch: 50 [32000/50000 (64%)]	Loss: 1.274292

Test set: Average loss: 1.1763, Accuracy: 6029/10000 (60%)

```
## Training Results
- Achieved Accuracy of 60% at 50th epoch

## Visualization
![output](https://user-images.githubusercontent.com/90888045/147481798-5aae4cff-9a95-452d-bd9b-1feb0bd64b33.png)


