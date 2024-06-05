# basic imports
import sys
import os
import csv
import random

# Pytorch libaries
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import ImageFolder

# For loop
from timeit import default_timer as timer
from tqdm.auto import tqdm
from IPython.display import clear_output

# For visualizing and troubleshooting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

# For saving a path and loading
from pathlib import Path
import zipfile

print("Imported successfully")

def modelzoo(model):
  if not isinstance(model, str):
    raise TypeError("specific_variable must be a string")
   
  match model:
    case "resnet18":
      return models.resnet18(weights='IMAGENET1K_V1')
    
    case "resnet34":
      return models.resnet34(weights='IMAGENET1K_V1')
    
    case "resnet50":
      return models.resnet50(weights='IMAGENET1K_V1')
    
    case "resnet101":
      return models.resnet101(weights='IMAGENET1K_V1')
    
    case "resnet152":
      return models.resnet152(weights='IMAGENET1K_V1')
    
    case "vgg11":
      return models.vgg11(pretrained=True)
    
    case "vgg13":
      return models.vgg13(pretrained=True)
    
    case "vgg16":
      return models.vgg16(pretrained=True)
    
    case "vgg19":
      return models.vgg19(pretrained=True)
    
    case "alexnet":
      return models.alexnet(pretrained=True)
    
    case "squeezenet1_0":
      return models.squeezenet1_0(pretrained=True)
    
    case "squeezenet1_1":
      return models.squeezenet1_1(pretrained=True)
    
    case "densenet121":
      return models.densenet121(pretrained=True)
    
    case "densenet169":
      return models.densenet169(pretrained=True)
    
    case "densenet201":
      return models.densenet201(pretrained=True)
    
    case "densenet161":
      return models.densenet161(pretrained=True)
    
    case "inception_v3":
      return models.inception_v3(pretrained=True)
    
    case "googlenet":
      return models.googlenet(pretrained=True)
    
    case "shufflenet_v2_x0_5":
      return models.shufflenet_v2_x0_5(pretrained=True)
    
    case "shufflenet_v2_x1_0":
      return models.shufflenet_v2_x1_0(pretrained=True)
    
    case "shufflenet_v2_x1_5":
      return models.shufflenet_v2_x1_5(pretrained=True)
    
    case "shufflenet_v2_x2_0":
      return models.shufflenet_v2_x2_0(pretrained=True)
    
    case "mobilenet_v2":
      return models.mobilenet_v2(pretrained=True)
    
    case "resnext50_32x4d":
      return models.resnext50_32x4d(pretrained=True)
    
    case "resnext101_32x8d":
      return models.resnext101_32x8d(pretrained=True)
    
    case "wide_resnet50_2":
      return models.wide_resnet50_2(pretrained=True)
    
    case "wide_resnet101_2":
      return models.wide_resnet101_2(pretrained=True)
    
    case "mnasnet0_5":
      return models.mnasnet0_5(pretrained=True)
    
    case "mnasnet0_75":
      return models.mnasnet0_75(pretrained=True)
    
    case "mnasnet1_0":
      return models.mnasnet1_0(pretrained=True)
    
    case "mnasnet1_3":
      return models.mnasnet1_3(pretrained=True)
    
    case "efficientnet_b0":
      return models.efficientnet_b0(pretrained=True)
    
    case "efficientnet_b1":
      return models.efficientnet_b1(pretrained=True)
    
    case "efficientnet_b2":
      return models.efficientnet_b2(pretrained=True)
    
    case "efficientnet_b3":
      return models.efficientnet_b3(pretrained=True)
    
    case "efficientnet_b4":
      return models.efficientnet_b4(pretrained=True)
    
    case "efficientnet_b5":
      return models.efficientnet_b5(pretrained=True)
    
    case "efficientnet_b6":
      return models.efficientnet_b6(pretrained=True)
    
    case "efficientnet_b7":
      return models.efficientnet_b7(pretrained=True)
    
    case _:
      raise ValueError(f"Model {model} is not available in the model zoo")

def recall_fn(y_true, y_pred):
    """Calculates recall between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared.

    Returns:
        float: Recall value between y_true and y_pred.
    """
    # Convert probabilities/logits to binary predictions (if applicable)
    y_pred = y_pred.argmax(dim=1) if y_pred.ndim > 1 else y_pred
    y_true = y_true.type_as(y_pred)

    # Calculate True Positives (TP) and False Negatives (FN)
    TP = torch.logical_and(y_pred == 1, y_true == 1).sum().item()
    FN = torch.logical_and(y_pred == 0, y_true == 1).sum().item()

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # handle division by zero if there are no positives
    return recall * 100

def precision_fn(y_true, y_pred):
    """Calculates precision between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared.

    Returns:
        float: Precision value between y_true and y_pred.
    """
    # Convert probabilities/logits to binary predictions (if applicable)
    y_pred = y_pred.argmax(dim=1) if y_pred.ndim > 1 else y_pred
    y_true = y_true.type_as(y_pred)

    # Calculate True Positives (TP) and False Positives (FP)
    TP = torch.logical_and(y_pred == 1, y_true == 1).sum().item()
    FP = torch.logical_and(y_pred == 1, y_true == 0).sum().item()

    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # handle division by zero if there are no positive predictions
    return precision * 100

def specificity_fn(y_true, y_pred):
    """
    Calculates specificity between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared.

    Returns:
        float: Specificity value between y_true and y_pred.
    """
    # Convert probabilities/logits to binary predictions (if applicable)
    y_pred = y_pred.argmax(dim=1) if y_pred.ndim > 1 else y_pred
    y_true = y_true.type_as(y_pred)

    # Calculate True Negatives (TN) and False Positives (FP)
    TN = torch.logical_and(y_pred == 0, y_true == 0).sum().item()
    FP = torch.logical_and(y_pred == 1, y_true == 0).sum().item()

    # Calculate Specificity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # handle division by zero if there are no negatives
    return specificity * 100

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def train_step_binary(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device):
  train_loss, train_acc, train_recall, train_prec, train_spec = 0, 0, 0, 0, 0

  # For training
  model.train()
  for batch, (image, label) in enumerate(data_loader):

    # Optimizer zero grad
    optimizer.zero_grad()

    # Forward pass
    image = image.to(device)
    label = label.to(device, dtype=torch.float)
    pred = model(image).squeeze()

    # Calculate the loss
    loss = loss_fn(pred, label)
    train_loss += loss

    # Calculate the accuracy
    train_acc += accuracy_fn(y_true = label,
                             y_pred = torch.round(torch.sigmoid(pred)))

     # Recall calculation
    train_recall += recall_fn(label,
                             torch.round(torch.sigmoid(pred)))

      # Precision calculation
    train_prec += precision_fn(label,
                              torch.round(torch.sigmoid(pred)))

      # Specificity calculation
    train_spec += specificity_fn(label,
                                torch.round(torch.sigmoid(pred)))

    # Backprop
    loss.backward()

    # Optimizer step
    optimizer.step()

  # For calculating average trainloss over every batch in each epoch
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  train_recall /= len(data_loader)
  train_prec /= len(data_loader)
  train_spec /= len(data_loader)

  return train_loss, train_acc, train_recall, train_prec, train_spec

def test_step_binary(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device):

  test_loss, test_acc, test_recall, test_prec, test_spec = 0, 0, 0, 0, 0

  # For testing
  model.eval()

  with torch.inference_mode():
    for batch, (image, label) in enumerate(data_loader):
      # Forward pass
      image = image.to(device)
      label = label.to(device, dtype=torch.float)
      pred = model(image).squeeze()

      # Loss calculation
      loss = loss_fn(pred, label)
      test_loss += loss

      # Accuracy calculation
      test_acc += accuracy_fn(label,
                              torch.round(torch.sigmoid(pred)))

      # Recall calculation
      test_recall += recall_fn(label,
                               torch.round(torch.sigmoid(pred)))

      # Precision calculation
      test_prec += precision_fn(label,
                                torch.round(torch.sigmoid(pred)))

      # Specificity calculation
      test_spec += specificity_fn(label,
                                  torch.round(torch.sigmoid(pred)))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    test_recall /= len(data_loader)
    test_prec /= len(data_loader)
    test_spec /= len(data_loader)

  return test_loss, test_acc, test_recall, test_prec, test_spec

def train_step_multi(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device):
  train_loss, train_acc, train_recall, train_prec, train_spec = 0, 0, 0, 0, 0

  # For training
  model.train()
  for batch, (image, label) in enumerate(data_loader):

    # Optimizer zero grad
    optimizer.zero_grad()

    # Forward pass
    image = image.to(device)
    label = label.to(device, dtype=torch.float)
    pred = model(image).squeeze()
    pred1 = pred.argmax(dim=1)

    # Calculate the loss
    loss = loss_fn(pred, label)
    train_loss += loss

    # Calculate the accuracy
    train_acc += accuracy_fn(y_true = label,
                             y_pred = pred1)

     # Recall calculation
    train_recall += recall_fn(label,
                             pred1)

      # Precision calculation
    train_prec += precision_fn(label,
                              pred1)

      # Specificity calculation
    train_spec += specificity_fn(label,
                                pred1)

    # Backprop
    loss.backward()

    # Optimizer step
    optimizer.step()

  # For calculating average trainloss over every batch in each epoch
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  train_recall /= len(data_loader)
  train_prec /= len(data_loader)
  train_spec /= len(data_loader)

  return train_loss, train_acc, train_recall, train_prec, train_spec

def test_step_multi(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device,
                  ):

  test_loss, test_acc, test_recall, test_prec, test_spec = 0, 0, 0, 0, 0

  # For testing
  model.eval()

  with torch.inference_mode():
    for batch, (image, label) in enumerate(data_loader):
      # Forward pass
      image = image.to(device)
      label = label.to(device, dtype=torch.float)
      pred = model(image).squeeze()
      pred1 = pred.argmax(dim=1)

      # Loss calculation
      loss = loss_fn(pred, label)
      test_loss += loss

      # Accuracy calculation
      test_acc += accuracy_fn(label,
                              pred1)

      # Recall calculation
      test_recall += recall_fn(label,
                               pred1)

      # Precision calculation
      test_prec += precision_fn(label,
                                pred1)

      # Specificity calculation
      test_spec += specificity_fn(label,
                                  pred1)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    test_recall /= len(data_loader)
    test_prec /= len(data_loader)
    test_spec /= len(data_loader)

  return test_loss, test_acc, test_recall, test_prec, test_spec

def data_loading(filepath):
  def load_dataset(dataset_path, transform=None):
    return ImageFolder(dataset_path, transform=transform)

  # Define your transformations
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
  ])

  combined_dataset = load_dataset(filepath, transform=transform)

  print(f"Class types: {combined_dataset.classes}")

  # Split the subset into train and test sets
  train_size = int(0.8 * len(combined_dataset))  # 80% for training
  test_size = len(combined_dataset) - train_size  # 20% for testing
  train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

  # Create DataLoaders for train and test sets
  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
  return train_dataloader, test_dataloader

def plot_function(test_dataloader, df, device, model, loss_fn, filepath):
  # Set a consistent style
  sns.set_theme()

  # Create a figure with GridSpec
  fig = plt.figure(figsize=(15, 15))  # Adjust the size as needed
  gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

  # Use the first four plots
  ax1 = plt.subplot(gs[0, 0])
  ax2 = plt.subplot(gs[0, 1])
  ax3 = plt.subplot(gs[1, 0])
  ax4 = plt.subplot(gs[1, 1])
  ax5 = plt.subplot(gs[2, 0])
  ax6 = plt.subplot(gs[2, 1])

  # Plot each metric

  palette = sns.dark_palette("#79C", n_colors=4, reverse=True)
  sns.lineplot(x='Epoch', y='Train Loss', data=df, label='Train Loss', color = palette[0], ax= ax1)
  sns.lineplot(x='Epoch', y='Test Loss', data=df, label='Test Loss', color = palette[1], ax = ax1)
  ax1.set_title('Training vs Test Losses')
  ax1.legend()

  palette = sns.dark_palette("#b285bc", n_colors=4, reverse=True)

  sns.lineplot(x='Epoch', y='Train Accuracy', data=df, label='Train Accuracy', color = palette[0], ax = ax2)
  sns.lineplot(x='Epoch', y='Test Accuracy', data=df, label='Test Accuracy', color = palette[1], ax = ax2)
  ax2.set_title('Training vs Test Accuracies')
  ax2.legend()


  palette = sns.dark_palette("#b285bc", n_colors=4, reverse=True)
  sns.lineplot(x='Epoch', y='Train Recall', data=df, label='Train Recall', color= palette[0], ax = ax3)
  sns.lineplot(x='Epoch', y='Test Recall', data=df, label='Test Recall', color= palette[1], ax = ax3)
  ax3.set_title('Training vs Test Recalls')
  ax3.legend()

  palette = sns.dark_palette("#79C", n_colors=4, reverse=True)
  sns.lineplot(x='Epoch', y='Train Precision', data=df, label='Train Precision', color= palette[0],ax = ax4)
  sns.lineplot(x='Epoch', y='Test Precision', data=df, label='Test Precision', color= palette[1], ax = ax4)
  ax4.set_title('Training vs Test Precision')
  ax4.legend()

  palette = sns.dark_palette("#79C", n_colors=4, reverse=True)
  sns.lineplot(x='Epoch', y='Train Specificity', data=df, label='Train Specificity', color= palette[0], ax = ax5)
  sns.lineplot(x='Epoch', y='Test Specificity', data=df, label='Test Specificity', color= palette[1], ax = ax5)
  ax5.set_title('Training vs Test Speceficity')
  ax5.legend()

  y_pred = []
  y_true = []
  y_pred_numpy = []
  y_true_numpy = []

  model.eval()
  with torch.inference_mode():
      for batch, (image, label) in enumerate(test_dataloader):
        # Forward pass
        image = image.to(device)
        label = label.to(device, dtype=torch.float)
        pred = model(image).squeeze()
        pred = torch.round(torch.sigmoid(pred))

        # Loss calculation
        loss = loss_fn(pred, label)
        test_loss += loss

        # Accuracy calculation
        test_acc += accuracy_fn(y_true = label,
                                y_pred = torch.round(torch.sigmoid(pred)))
        y_pred.append(pred)
        y_true.append(label)

  # Convert lists to NumPy arrays
  for tensor in y_pred:
      # Move the tensor to CPU and convert to numpy
      numpy_array = tensor.to('cpu').numpy()
      y_pred_numpy.append(numpy_array)

  for tensor in y_true:
      # Move the tensor to CPU and convert to numpy
      numpy_array = tensor.to('cpu').numpy()
      y_true_numpy.append(numpy_array)

  # Convert lists to NumPy arrays
  y_pred_numpy = np.hstack(y_pred_numpy)
  y_true_numpy = np.hstack(y_true_numpy)

  # Compute the confusion matrix
  palette = sns.dark_palette("#b285bc", n_colors=4, reverse=True, as_cmap=True)
  cm = confusion_matrix(y_true_numpy, y_pred_numpy)
  sns.heatmap(cm, annot=True, fmt="d", cmap=palette)
  ax6.set_title('Confusion Matrix')
  ax6.set_ylabel('True Label')
  ax6.set_xlabel('Predicted Label')

  # Adjust layout
  plt.tight_layout()

  # Save the entire figure
  graph_save_path = filepath / 'metrics_comparison.png'
  plt.savefig(graph_save_path)

  # Show the plot
  plt.show()

def save_model_info(model, epochs, total_train_time, optimizer, loss_fn, file_path, transform):
  with open(file_path, 'w') as file:
      # Model architecture
      file.write('**Model Architecture:**\n\n')
      file.write(str(model))
      file.write('\n\n')

      # Transformations
      file.write('**Transformations:**\n\n')
      file.write(str(transform))
      file.write('\n\n')

      # Optimizer details
      file.write('**Optimizer Details:**\n\n')
      file.write(str(optimizer))
      file.write('\n\n')

      # Loss function details
      file.write('**Loss Function:**\n\n')
      file.write(str(loss_fn))
      file.write('\n\n')

      # Number of epochs
      file.write('**Number of Epochs:**\n\n')
      file.write(f'{epochs}\n\n')

      # Total training time
      file.write('**Total Training Time:**\n\n')
      file.write(f'{total_train_time:.2f} seconds\n')


def modelsave(model, savepath, model_iter, epochs,
              totalTrainTime, optimizer, loss_fn, 
              modelInfoSavePath, transform):
  

  drive_path = savepath
  drive_path.mkdir(parents=True, exist_ok=True)

  # Create a model save path
  modelName = model_iter + '.pth'
  modelSavePath = drive_path / modelName

  print(f"SAVING MODEL TO: {modelSavePath}")

  # Saving model info to text file
  
  textName = model_iter + '.txt'
  modelInfoSavePath = drive_path / textName
  save_model_info(model, epochs, totalTrainTime, optimizer, loss_fn, modelInfoSavePath, transform)
  print(f"Model information saved to {modelInfoSavePath}")

  # Example model save (replace `model.state_dict()` with your actual model's state dict)
  torch.save(obj=model.state_dict(), f=modelSavePath)

def full_experiment(model_type, optimizer,
                    epochs, loss_fn, photo_folder,
                    save_folder):
  
  model = modelzoo(model_type)
  train_dataloader, test_dataloader = data_loading(photo_folder)
  
  fc