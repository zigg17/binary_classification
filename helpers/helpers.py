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

def test_step_multi(model: torch.nn.Module,
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

