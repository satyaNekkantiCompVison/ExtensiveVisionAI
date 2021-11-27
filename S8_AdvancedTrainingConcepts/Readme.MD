# Assignment Submission for Session 8 (Late Assignment)

- [Problem Statement](#problem-statement)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Team Members](#team-members) 

# Problem statement
1 . Train for 40 Epochs
2. Plot 20 misclassified images
3. Plot 20 GradCam output on the SAME misclassified images
4. Apply these transforms while training:
	1. RandomCrop(32, padding=4)
    2. CutOut(16x16)
    3.Rotate(±5°)
5. Must use ReduceLROnPlateau
6. Must use LayerNormalization ONLY


# Model-evaluation


## Code Modularity
- The main model and utility functions are written in [Model Repo](https://github.com/satyaNekkantiCompVison/pytorch_visionmodels)
- Its data structure is 
```
├── main.py
├── models
│   └── resnet.py
├── README.md
└── utils
    ├── data_augmentations.py
    ├── fitmodel.py
    ├── gradcam.py
    ├── helper.py
    ├── plot_images.py
    ├── test.py
    └── train.py
```
- Notebook to this Assignment can be found here [Notebook](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S8_AdvancedTrainingConcepts/S8_Advanced_Training_concepts.ipynb)


## Augmentation Strategy Used
```
A.Compose({
  A.Rotate (limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
  A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by default
                A.RandomCrop(32,32)]),
  A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363, mask_fill_value=None),
  A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
})
```

## Resent Model with Layer normalization

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
        GroupNorm-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
        GroupNorm-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
        GroupNorm-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
        GroupNorm-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
        GroupNorm-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
        GroupNorm-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
        GroupNorm-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
        GroupNorm-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
        GroupNorm-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
        GroupNorm-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.50
Params size (MB): 42.63
Estimated Total Size (MB): 56.14
----------------------------------------------------------------
```

## Training and Testing Epochs

```
EPOCH: 1 (LR: 0.001)
Batch_id=390 Loss=15.17206 Accuracy=21.03%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.9889, Accuracy: 2184/10000 (21.84%)
EPOCH: 2 (LR: 0.001)
Batch_id=390 Loss=14.02129 Accuracy=22.06%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.9395, Accuracy: 2444/10000 (24.44%)
EPOCH: 3 (LR: 0.001)
Batch_id=390 Loss=13.97876 Accuracy=22.65%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.9268, Accuracy: 2504/10000 (25.04%)
EPOCH: 4 (LR: 0.001)
Batch_id=390 Loss=13.94990 Accuracy=22.97%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.9339, Accuracy: 2444/10000 (24.44%)
EPOCH: 5 (LR: 0.001)
Batch_id=390 Loss=13.95834 Accuracy=23.19%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.9096, Accuracy: 2542/10000 (25.42%)
EPOCH: 6 (LR: 0.001)
Batch_id=390 Loss=13.97309 Accuracy=23.81%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.8852, Accuracy: 2666/10000 (26.66%)
EPOCH: 7 (LR: 0.001)
Batch_id=390 Loss=13.92728 Accuracy=24.14%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.8720, Accuracy: 2568/10000 (25.68%)
EPOCH: 8 (LR: 0.001)
Batch_id=390 Loss=13.93769 Accuracy=24.98%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.8709, Accuracy: 2621/10000 (26.21%)
EPOCH: 9 (LR: 0.001)
Batch_id=390 Loss=13.94846 Accuracy=25.62%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.8400, Accuracy: 2712/10000 (27.12%)
EPOCH: 10 (LR: 0.001)
Batch_id=390 Loss=13.93786 Accuracy=26.14%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.8635, Accuracy: 2702/10000 (27.02%)
EPOCH: 11 (LR: 0.001)
Batch_id=390 Loss=13.90531 Accuracy=26.37%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.8362, Accuracy: 2718/10000 (27.18%)
EPOCH: 12 (LR: 0.001)
Batch_id=390 Loss=13.92408 Accuracy=26.42%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7916, Accuracy: 2890/10000 (28.90%)
EPOCH: 13 (LR: 0.001)
Batch_id=390 Loss=13.93242 Accuracy=26.68%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.8082, Accuracy: 2854/10000 (28.54%)
EPOCH: 14 (LR: 0.001)
Batch_id=390 Loss=13.90472 Accuracy=27.00%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7973, Accuracy: 2969/10000 (29.69%)
EPOCH: 15 (LR: 0.001)
Batch_id=390 Loss=13.91411 Accuracy=26.76%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.8680, Accuracy: 2692/10000 (26.92%)
EPOCH: 16 (LR: 0.001)
Batch_id=390 Loss=13.93283 Accuracy=27.04%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7930, Accuracy: 2870/10000 (28.70%)
EPOCH: 17 (LR: 0.001)
Batch_id=390 Loss=13.93471 Accuracy=27.38%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7966, Accuracy: 2924/10000 (29.24%)
EPOCH: 18 (LR: 0.001)
Batch_id=390 Loss=13.90856 Accuracy=27.20%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7933, Accuracy: 2897/10000 (28.97%)
EPOCH: 19 (LR: 0.001)
Batch_id=390 Loss=13.91830 Accuracy=27.22%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.8068, Accuracy: 2891/10000 (28.91%)
EPOCH: 20 (LR: 0.001)
Batch_id=390 Loss=13.94282 Accuracy=27.43%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7953, Accuracy: 2867/10000 (28.67%)
EPOCH: 21 (LR: 0.001)
Batch_id=390 Loss=13.92923 Accuracy=27.60%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7833, Accuracy: 2951/10000 (29.51%)
EPOCH: 22 (LR: 0.001)
Batch_id=390 Loss=13.91059 Accuracy=27.58%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7682, Accuracy: 3013/10000 (30.13%)
EPOCH: 23 (LR: 0.001)
Batch_id=390 Loss=13.92411 Accuracy=27.83%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7775, Accuracy: 2978/10000 (29.78%)
EPOCH: 24 (LR: 0.001)
Batch_id=390 Loss=13.95131 Accuracy=27.39%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7671, Accuracy: 3052/10000 (30.52%)
EPOCH: 25 (LR: 0.001)
Batch_id=390 Loss=13.91995 Accuracy=27.59%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7698, Accuracy: 2976/10000 (29.76%)
EPOCH: 26 (LR: 0.001)
Batch_id=390 Loss=13.92556 Accuracy=27.44%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7925, Accuracy: 2958/10000 (29.58%)
EPOCH: 27 (LR: 0.001)
Batch_id=390 Loss=13.94669 Accuracy=27.69%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7686, Accuracy: 3042/10000 (30.42%)
EPOCH: 28 (LR: 0.001)
Batch_id=390 Loss=13.94817 Accuracy=27.85%: 100%|██████████| 391/391 [04:36<00:00,  1.41it/s]

Test set: Average loss: 1.7723, Accuracy: 2942/10000 (29.42%)
EPOCH: 29 (LR: 0.001)
Batch_id=390 Loss=13.91974 Accuracy=27.70%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.7606, Accuracy: 2916/10000 (29.16%)
EPOCH: 30 (LR: 0.001)
Batch_id=390 Loss=13.94285 Accuracy=27.68%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.7519, Accuracy: 3043/10000 (30.43%)
EPOCH: 31 (LR: 0.001)
Batch_id=390 Loss=13.95359 Accuracy=28.03%: 100%|██████████| 391/391 [04:39<00:00,  1.40it/s]

Test set: Average loss: 1.7629, Accuracy: 3080/10000 (30.80%)
EPOCH: 32 (LR: 0.001)
Batch_id=390 Loss=13.93298 Accuracy=27.80%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.7538, Accuracy: 3064/10000 (30.64%)
EPOCH: 33 (LR: 0.001)
Batch_id=390 Loss=13.93510 Accuracy=27.96%: 100%|██████████| 391/391 [04:39<00:00,  1.40it/s]

Test set: Average loss: 1.7478, Accuracy: 3099/10000 (30.99%)
EPOCH: 34 (LR: 0.001)
Batch_id=390 Loss=13.95652 Accuracy=27.87%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.7992, Accuracy: 2946/10000 (29.46%)
EPOCH: 35 (LR: 0.001)
Batch_id=390 Loss=13.95140 Accuracy=28.06%: 100%|██████████| 391/391 [04:38<00:00,  1.40it/s]

Test set: Average loss: 1.7527, Accuracy: 2949/10000 (29.49%)
EPOCH: 36 (LR: 0.001)
Batch_id=390 Loss=13.93441 Accuracy=27.88%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7670, Accuracy: 2951/10000 (29.51%)
EPOCH: 37 (LR: 0.001)
Batch_id=390 Loss=13.94731 Accuracy=27.67%: 100%|██████████| 391/391 [04:38<00:00,  1.41it/s]

Test set: Average loss: 1.7527, Accuracy: 2948/10000 (29.48%)
EPOCH: 38 (LR: 0.001)
Batch_id=390 Loss=13.96255 Accuracy=27.70%: 100%|██████████| 391/391 [04:37<00:00,  1.41it/s]

Test set: Average loss: 1.7632, Accuracy: 3028/10000 (30.28%)
EPOCH: 39 (LR: 0.001)
Batch_id=390 Loss=13.95400 Accuracy=27.96%: 100%|██████████| 391/391 [04:35<00:00,  1.42it/s]

Test set: Average loss: 1.8188, Accuracy: 2915/10000 (29.15%)
EPOCH: 40 (LR: 0.001)
Batch_id=390 Loss=13.94006 Accuracy=27.92%: 100%|██████████| 391/391 [04:35<00:00,  1.42it/s]

Test set: Average loss: 1.7508, Accuracy: 3022/10000 (30.22%)

```

## PLotting Accuracy vs LOSS

![graphs](https://user-images.githubusercontent.com/90888045/143675201-e3030237-77a2-4048-8671-09491345a593.png)


## 20 Misclassified Images

![misclassified](https://user-images.githubusercontent.com/90888045/143675209-27395f79-4052-4f3e-b1b2-d4143ac04823.png)

## GRAD cam on 20 misclassified images

![gradcam_output](https://user-images.githubusercontent.com/90888045/143675214-dfb8acb4-3e3c-4432-898b-0901d3568a58.png)



# Team Members

- Satya Nekkanti
- Pranabesh Dash
