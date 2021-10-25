# Problem Statement

### Assignment:

  Your new target is:
  - 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
  - Less than or equal to 15 Epochs
  - Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
  
 ## Model Experiments
  | Experiment	| Target |	Parameters| 	Training Accuracy| 	Test Accuracy	| Analysis
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MNIST_Code1_BasicSetup](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_Experiments/MNIST_Code1_BasicSetup.ipynb)	|   • Get the set-up right<br>• Set Transforms<br>• Set Data Loader<br>• Set Basic Working Code<br>• Set Basic Training  & Test Loop                                                                                | 6.3M       | 99.93             | 99.28         | •  Extremely Heavy Model for such a problem<br>•  Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step                                                                                                                                                                                                                                                                                                              |
| [MNIST_Code2_BasicSkeleton](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_Experiments/MNIST_Code2_BasicSkeleton.ipynb) |	Get the basic skeleton interms of convolution and placement of transition blocks (max pooling, 1x1's) Reduce the number of parameters as low as possible | 	48180	| 99.5 | 	98.87 | 	We have structured our model in a readable way <br> The model is lighter with less number of parameters The performace is reduced compared to previous models<br> Since we have reduced model capacity, this is expected, the model has capability to learn. <br> Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.|
| [MNIST_Code3_BatchNormalization](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_Experiments/MNIST_Code3_BatchNormalization.ipynb)	| Add Batch-norm to increase model efficiency. |	10970|	99.73|	99.15| 	There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer. Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.|
| [MNIST_Code4_DropOutandGAP](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_Experiments/MNIST_Code4_DropOutandGAP.ipynb)|	Add Dropout in each layer , FC layers after Avg GAP layer to increase model efficiency.|	7496|	99.18|	99.33|	There is slight decrease in the number of parameters, also accuracy decreased while we add dropout at each layer and used GAP before a convolution layer Model problem is rectified to an extent we got train accuracy to 99.18% whereas the test accuracy is 99.33%. But, we have not reached the target test accuracy 99.40%.|
| [MNIST_Code5_Agumentation](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_Experiments/MNIST_Code5_Agumentation.ipynb)	|To achive the better results with data augmentation techniques.|	7496 | 99.01| 99.37 |The model is underfitting in the starting epochs but got stable after few epochs.<br>The best training accuracy shows the gap for future training <br>Adding learning rate schedulers can help to improve the training, as model update weights with step of a particular learning rate so there is a chance that model may be stuck at the local minima which can be solved by incrersing or decresing the learning rate|			
| [MNIST_FinalModel](https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/blob/main/S5_CodingDrillDown/MNIST_FinalModel.ipynb)	|Added capacity (additional FC layer after GAP) to the model and added LR Scheduler|	7496|	99.27|	99.46|	The model is under-fitting. This is fine, as we know we have made our train data harder. LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to have achieve consistent accuracy in last few layers |


## Training Log 
```
EPOCH: 1
Batch_id=468 Loss=0.29607 Accuracy=70.08: 100%|██████████| 469/469 [00:53<00:00,  8.80it/s]

Test set: Average loss: 0.2511, Accuracy: 9612/10000 (96.12%)

EPOCH: 2
Batch_id=468 Loss=0.14628 Accuracy=95.54: 100%|██████████| 469/469 [00:53<00:00,  8.73it/s]

Test set: Average loss: 0.0785, Accuracy: 9786/10000 (97.86%)

EPOCH: 3
Batch_id=468 Loss=0.04647 Accuracy=97.11: 100%|██████████| 469/469 [00:53<00:00,  8.71it/s]

Test set: Average loss: 0.0556, Accuracy: 9837/10000 (98.37%)

EPOCH: 4
Batch_id=468 Loss=0.04252 Accuracy=97.72: 100%|██████████| 469/469 [00:53<00:00,  8.77it/s]

Test set: Average loss: 0.0493, Accuracy: 9851/10000 (98.51%)

EPOCH: 5
Batch_id=468 Loss=0.03376 Accuracy=98.14: 100%|██████████| 469/469 [00:53<00:00,  8.79it/s]

Test set: Average loss: 0.0369, Accuracy: 9892/10000 (98.92%)

EPOCH: 6
Batch_id=468 Loss=0.03412 Accuracy=98.38: 100%|██████████| 469/469 [00:54<00:00,  8.55it/s]

Test set: Average loss: 0.0304, Accuracy: 9902/10000 (99.02%)

EPOCH: 7
Batch_id=468 Loss=0.03012 Accuracy=98.48: 100%|██████████| 469/469 [00:54<00:00,  8.62it/s]

Test set: Average loss: 0.0286, Accuracy: 9912/10000 (99.12%)

EPOCH: 8
Batch_id=468 Loss=0.01021 Accuracy=98.68: 100%|██████████| 469/469 [00:52<00:00,  8.86it/s]

Test set: Average loss: 0.0304, Accuracy: 9909/10000 (99.09%)

EPOCH: 9
Batch_id=468 Loss=0.20190 Accuracy=98.75: 100%|██████████| 469/469 [00:53<00:00,  8.71it/s]

Test set: Average loss: 0.0251, Accuracy: 9921/10000 (99.21%)

EPOCH: 10
Batch_id=468 Loss=0.08117 Accuracy=98.80: 100%|██████████| 469/469 [00:52<00:00,  8.88it/s]

Test set: Average loss: 0.0223, Accuracy: 9930/10000 (99.30%)

EPOCH: 11
Batch_id=468 Loss=0.05942 Accuracy=98.88: 100%|██████████| 469/469 [00:53<00:00,  8.70it/s]

Test set: Average loss: 0.0217, Accuracy: 9939/10000 (99.39%)

EPOCH: 12
Batch_id=468 Loss=0.01466 Accuracy=99.00: 100%|██████████| 469/469 [00:53<00:00,  8.71it/s]

Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)

EPOCH: 13
Batch_id=468 Loss=0.03093 Accuracy=99.04: 100%|██████████| 469/469 [00:52<00:00,  8.90it/s]

Test set: Average loss: 0.0188, Accuracy: 9946/10000 (99.46%)

EPOCH: 14
Batch_id=468 Loss=0.03084 Accuracy=99.17: 100%|██████████| 469/469 [00:54<00:00,  8.64it/s]

Test set: Average loss: 0.0183, Accuracy: 9942/10000 (99.42%)

EPOCH: 15
Batch_id=468 Loss=0.03181 Accuracy=99.27: 100%|██████████| 469/469 [00:53<00:00,  8.79it/s]

Test set: Average loss: 0.0187, Accuracy: 9941/10000 (99.41%)
```

## Accuracy vs Epochs plots


![plots](https://user-images.githubusercontent.com/90888045/138742094-e7d1ed4c-d447-4746-9d80-bab7c4b463fe.png)

## Collaborators
Satya Nekkanti (https://github.com/satyaNekkantiCompVison)
Pranabesh Dash (https://github.com/pranabeshdash/EVA7)
