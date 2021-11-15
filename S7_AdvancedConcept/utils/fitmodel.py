import torch.optim as optim

from utils.train import train
from utils.test import test


def fit_model(model, hardware=None,epochs=20, lambda1=False, lambda2=False,train_loader= None, test_loader=None):
  train_accuracy, train_loss_list, test_accuracy, test_loss_value = [],[],[],[]
  
  training_param = {}
  if lambda2:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
  else:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.017, epochs=epochs, steps_per_epoch=len(train_loader))

  for epoch in range(1,epochs+1):
      print("epoch:", epoch)
      train_acc, train_loss = train(model, hardware, train_loader, optimizer, lambda1, scheduler)
      test_acc, test_loss = test(model, hardware, test_loader)

      train_accuracy.append(train_acc)
      train_loss_list.append(train_loss)
      test_accuracy.append(test_acc)
      test_loss_value.append(test_loss)
  
  
  training_param["train_Acc"] = train_accuracy
  training_param["train_Loss"] = train_loss_list
  training_param["test_Acc"] = test_accuracy
  training_param["test_Loss"] = test_loss_value
  return training_param