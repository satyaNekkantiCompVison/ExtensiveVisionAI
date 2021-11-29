# ASSIGNMENT for Session 9 Custome ResNet and Higher Receptive Fields

- [Problem Statement](#problem-statement)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Team Members](#team-members) 

# Problem Statement
1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
	1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
	2. Layer1 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
		R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
		Add(X, R1)
	3. Layer 2 -
		Conv 3x3 [256k]
		MaxPooling2D
		BN
		ReLU
	4. Layer 3 -
		X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
		R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
		Add(X, R2)
	5. MaxPooling with Kernel Size 4
	6. FC Layer 
	7. SoftMax

2.	Uses One Cycle Policy such that:
	Total Epochs = 24
	Max at Epoch = 5
	LRMIN = FIND
	LRMAX = FIND
	NO Annihilation

3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

4. Batch size = 512

5. Target Accuracy: 90% (93% for late submission or double scores). 

# Model Evaluation

-  We can find S9_assignment google colab notebook here [Notebook](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S9_CustomResNet/S9_CustomResnet.ipynb)
1. Augmentation DataTransforms
- As the requirement we created augumented train dataset using the follwing augumentations parameters

```
train_aug = A.Compose(
    {      
     A.Sequential([
                   A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by defaulf
                   A.RandomCrop(32,32)
                   ]),
     A.HorizontalFlip(always_apply=False,p=0.1),
     A.CoarseDropout(1, 8, 8, 1, 8, 8,fill_value=0.473363, mask_fill_value=None, always_apply=False,p=0.1),
     A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
     }
     )

```

2. Custom Resnet
- We created a custom resnet model in which

```
    - 3 convolution layers are created
    - Convolution layer 1 added to Resnetblock
    - Convolution layer 3 added to Resnetblock
    - Followed by Max-pooling
    - FC-layer and Softmax
```
- The network can be found here [Custom Resnet](https://github.com/satyaNekkantiCompVison/pytorch_visionmodels/blob/main/models/custom_resnet.py)

3. Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
              ReLU-2           [-1, 64, 32, 32]               0
       BatchNorm2d-3           [-1, 64, 32, 32]             128
            Conv2d-4          [-1, 128, 32, 32]          73,856
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
       BasicBlock-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 256, 16, 16]         295,168
        MaxPool2d-14            [-1, 256, 8, 8]               0
      BatchNorm2d-15            [-1, 256, 8, 8]             512
             ReLU-16            [-1, 256, 8, 8]               0
           Conv2d-17            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-18            [-1, 512, 4, 4]               0
      BatchNorm2d-19            [-1, 512, 4, 4]           1,024
             ReLU-20            [-1, 512, 4, 4]               0
           Conv2d-21            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-22            [-1, 512, 4, 4]           1,024
           Conv2d-23            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-24            [-1, 512, 4, 4]           1,024
       BasicBlock-25            [-1, 512, 4, 4]               0
        MaxPool2d-26            [-1, 512, 1, 1]               0
           Linear-27                   [-1, 10]           5,130
================================================================
Total params: 6,574,090
Trainable params: 6,574,090
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.13
Params size (MB): 25.08
Estimated Total Size (MB): 31.22
----------------------------------------------------------------

```

4. LR finder 
- LR finder is calculated using the reference python module https://github.com/davidtvs/pytorch-lr-finder
- From 200 iteratins we get max_lr value to be 0.507

5. For OneCycleLR  scheduler we used below following settings

```
epochs = 24
scheduler_type = "OneCycleLR"
scheduler_params = {}
scheduler_params["max_lr"] = max_lr
scheduler_params["div_factor"] = 16
scheduler_params["three_phase"] = False
scheduler_params["anneal_strategy"] = "linear"
scheduler_params["max_epoch"] = 5
```

# Results

1. Training log

```
Epoch 1:
Train Loss=1.3297667503356934 Batch_id=390 LR= 0.12692 Train Accuracy= 30.45: 100%|██████████| 391/391 [01:42<00:00,  3.82it/s]

: Average Test loss: 0.0127, Test Accuracy: 4055/10000 (40.55%)

Epoch 2:
Train Loss=1.289710283279419 Batch_id=390 LR= 0.22212 Train Accuracy= 53.97: 100%|██████████| 391/391 [01:42<00:00,  3.83it/s]

: Average Test loss: 0.0088, Test Accuracy: 6244/10000 (62.44%)

Epoch 3:
Train Loss=0.8635790944099426 Batch_id=390 LR= 0.31732 Train Accuracy= 66.62: 100%|██████████| 391/391 [01:42<00:00,  3.83it/s]

: Average Test loss: 0.0074, Test Accuracy: 6676/10000 (66.76%)

Epoch 4:
Train Loss=0.7883478403091431 Batch_id=390 LR= 0.41252 Train Accuracy= 72.21: 100%|██████████| 391/391 [01:42<00:00,  3.81it/s]

: Average Test loss: 0.0056, Test Accuracy: 7595/10000 (75.95%)

Epoch 5:
Train Loss=0.8041122555732727 Batch_id=390 LR= 0.50741 Train Accuracy= 76.10: 100%|██████████| 391/391 [01:42<00:00,  3.82it/s]

: Average Test loss: 0.0057, Test Accuracy: 7581/10000 (75.81%)

Epoch 6:
Train Loss=0.5966517329216003 Batch_id=390 LR= 0.48070 Train Accuracy= 79.02: 100%|██████████| 391/391 [01:41<00:00,  3.85it/s]

: Average Test loss: 0.0052, Test Accuracy: 7765/10000 (77.65%)

Epoch 7:
Train Loss=0.5180865526199341 Batch_id=390 LR= 0.45399 Train Accuracy= 81.41: 100%|██████████| 391/391 [01:41<00:00,  3.84it/s]

: Average Test loss: 0.0046, Test Accuracy: 8010/10000 (80.10%)

Epoch 8:
Train Loss=0.6507259607315063 Batch_id=390 LR= 0.42728 Train Accuracy= 83.21: 100%|██████████| 391/391 [01:42<00:00,  3.81it/s]

: Average Test loss: 0.0047, Test Accuracy: 8088/10000 (80.88%)

Epoch 9:
Train Loss=0.4668838083744049 Batch_id=390 LR= 0.40057 Train Accuracy= 84.79: 100%|██████████| 391/391 [01:42<00:00,  3.82it/s]

: Average Test loss: 0.0040, Test Accuracy: 8337/10000 (83.37%)

Epoch 10:
Train Loss=0.4808238446712494 Batch_id=390 LR= 0.37387 Train Accuracy= 86.09: 100%|██████████| 391/391 [01:41<00:00,  3.84it/s]

: Average Test loss: 0.0042, Test Accuracy: 8271/10000 (82.71%)

Epoch 11:
Train Loss=0.3691065311431885 Batch_id=390 LR= 0.34716 Train Accuracy= 86.92: 100%|██████████| 391/391 [01:42<00:00,  3.82it/s]

: Average Test loss: 0.0038, Test Accuracy: 8457/10000 (84.57%)

Epoch 12:
Train Loss=0.18067866563796997 Batch_id=390 LR= 0.32045 Train Accuracy= 87.99: 100%|██████████| 391/391 [01:42<00:00,  3.81it/s]

: Average Test loss: 0.0046, Test Accuracy: 8212/10000 (82.12%)

Epoch 13:
Train Loss=0.22203388810157776 Batch_id=390 LR= 0.29374 Train Accuracy= 88.79: 100%|██████████| 391/391 [01:43<00:00,  3.78it/s]

: Average Test loss: 0.0034, Test Accuracy: 8608/10000 (86.08%)

Epoch 14:
Train Loss=0.40563711524009705 Batch_id=390 LR= 0.26703 Train Accuracy= 90.10: 100%|██████████| 391/391 [01:42<00:00,  3.80it/s]

: Average Test loss: 0.0037, Test Accuracy: 8555/10000 (85.55%)

Epoch 15:
Train Loss=0.2313348799943924 Batch_id=390 LR= 0.24032 Train Accuracy= 90.91: 100%|██████████| 391/391 [01:43<00:00,  3.79it/s]

: Average Test loss: 0.0037, Test Accuracy: 8583/10000 (85.83%)

Epoch 16:
Train Loss=0.2126224786043167 Batch_id=390 LR= 0.21361 Train Accuracy= 91.73: 100%|██████████| 391/391 [01:43<00:00,  3.78it/s]

: Average Test loss: 0.0038, Test Accuracy: 8521/10000 (85.21%)

Epoch 17:
Train Loss=0.11997723579406738 Batch_id=390 LR= 0.18690 Train Accuracy= 92.48: 100%|██████████| 391/391 [01:43<00:00,  3.78it/s]

: Average Test loss: 0.0036, Test Accuracy: 8682/10000 (86.82%)

Epoch 18:
Train Loss=0.23041482269763947 Batch_id=390 LR= 0.16019 Train Accuracy= 93.24: 100%|██████████| 391/391 [01:42<00:00,  3.80it/s]

: Average Test loss: 0.0033, Test Accuracy: 8716/10000 (87.16%)

Epoch 19:
Train Loss=0.2567698061466217 Batch_id=390 LR= 0.13348 Train Accuracy= 93.98: 100%|██████████| 391/391 [01:43<00:00,  3.79it/s]

: Average Test loss: 0.0035, Test Accuracy: 8755/10000 (87.55%)

Epoch 20:
Train Loss=0.16470648348331451 Batch_id=390 LR= 0.10677 Train Accuracy= 94.75: 100%|██████████| 391/391 [01:43<00:00,  3.77it/s]

: Average Test loss: 0.0032, Test Accuracy: 8836/10000 (88.36%)

Epoch 21:
Train Loss=0.08445911854505539 Batch_id=390 LR= 0.08006 Train Accuracy= 95.56: 100%|██████████| 391/391 [01:42<00:00,  3.81it/s]

: Average Test loss: 0.0032, Test Accuracy: 8896/10000 (88.96%)

Epoch 22:
Train Loss=0.18544237315654755 Batch_id=390 LR= 0.05335 Train Accuracy= 96.27: 100%|██████████| 391/391 [01:42<00:00,  3.83it/s]

: Average Test loss: 0.0032, Test Accuracy: 8951/10000 (89.51%)

Epoch 23:
Train Loss=0.09381355345249176 Batch_id=390 LR= 0.02664 Train Accuracy= 97.01: 100%|██████████| 391/391 [01:42<00:00,  3.82it/s]

: Average Test loss: 0.0031, Test Accuracy: 8992/10000 (89.92%)

Epoch 24:
Train Loss=0.04978334903717041 Batch_id=390 LR=-0.00007 Train Accuracy= 97.64: 100%|██████████| 391/391 [01:42<00:00,  3.81it/s]

: Average Test loss: 0.0031, Test Accuracy: 8986/10000 (89.86%)

```

- We got **Train Accuracy of 97.01%** where as **Test Accuracy of 89.86%**


2. Plotting Graphs

## Loss Grpahs for Train and Test

![loss_graphs](https://user-images.githubusercontent.com/90888045/143826840-6c055698-c928-4a2b-96f9-322fca0b4bb7.png)


## Accuracy Graph for Train and Test

![Accuracy_graph](https://user-images.githubusercontent.com/90888045/143826871-5f2081dd-1af3-414d-b723-af38d95383b4.png)


3. Plotting Mis-Classified Images

![misclassified_images](https://user-images.githubusercontent.com/90888045/143826889-f380a29f-dceb-4ec3-b1ec-0bde2cd20857.png)


4. Plotting Grad-cam for Mis-classified images

![gradcam_1](https://user-images.githubusercontent.com/90888045/143826909-682ed6c0-2944-4f85-8a35-183ee6f9f103.png)


![gradcam_2](https://user-images.githubusercontent.com/90888045/143826919-e791d76f-0476-4e3f-af9d-dd5fec5462dc.png)


# Team Members

1. Satya Nekkanti
2. Pranabesh Dash
