#import libraries 
import torch
import torch.multiprocessing
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torchsummary import summary
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 
import os

#Absoulte paths of the training, validation and testing dataset that we have already splitted, these paths are path of grayscale plotted spectrograms:
new_training_path=""
new_validation_path=""
new_test_path=""

#open data files and transorm them to a tensors
transform=transforms.ToTensor()
new_training_dataset = datasets.ImageFolder(new_training_path, transform=transform)
new_validation_dataset = datasets.ImageFolder(new_validation_path, transform=transform)
new_testing_dataset = datasets.ImageFolder(new_test_path, transform=transform)

#print out 
print(f"mixed_training_dataset: {len(new_training_dataset)}")
print(f"mixed_validation_datase: {len(new_validation_dataset)}")
print(f"mixed_testing_dataset: {len(new_testing_dataset)}")

#Load testing dataset into batches
train_batches = torch.utils.data.DataLoader(new_training_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

#Load testing dataset into batches
validation_batches = torch.utils.data.DataLoader(new_validation_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

#Load testing dataset into batches
test_batches = torch.utils.data.DataLoader(new_testing_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
writer = SummaryWriter()

#define the accuracy cutoff 
ACCURACY_THRESHOLD=0.9

#define the training function 
def train_neural_net(epochs, model, loss_func, optimizer, train_batches, val_batches):
    final_accuracy = 0

    for epoch in range(epochs):
        # Training mode
        model.train()
        total_train_correct = 0
        total_train_samples = 0
        train_loss = 0

        with torch.enable_grad():
            for batch_idx, (images, labels) in enumerate(train_batches):
                predictions = model(images)
                loss = loss_func(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute batch accuracy
                _, preds = torch.max(predictions, 1)
                batch_correct = (preds == labels).sum().item()
                batch_size = labels.size(0)
                batch_accuracy = batch_correct / batch_size

                total_train_correct += batch_correct
                total_train_samples += batch_size
                train_loss += loss.item()

                # Print batch-level info
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_batches)}], "
                      f"Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy:.4f}")

            # Compute epoch training loss and accuracy
            train_loss /= len(train_batches)
            train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0

        writer.add_scalar("training loss", train_loss, epoch)
        writer.add_scalar("training accuracy", train_accuracy, epoch)

        # Validation mode
        model.eval()
        val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        with torch.inference_mode():
            for images, labels in val_batches:
                predictions = model(images)
                loss = loss_func(predictions, labels)

                val_loss += loss.item()

                # Compute batch accuracy
                _, preds = torch.max(predictions, 1)
                batch_correct = (preds == labels).sum().item()
                total_val_correct += batch_correct
                total_val_samples += labels.size(0)

            val_loss /= len(val_batches)
            val_accuracy = total_val_correct / total_val_samples if total_val_samples > 0 else 0.0
            final_accuracy = val_accuracy

        writer.add_scalar("validation loss", val_loss, epoch)
        writer.add_scalar("validation accuracy", val_accuracy, epoch)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if train_accuracy >= ACCURACY_THRESHOLD:
            print(f"Stopping early as validation accuracy reached {val_accuracy:.4f} at epoch {epoch+1}.")
            break

    return final_accuracy


#define test function that will alsoe calculate the R1 score ,precision and recall score and then it will show us the confussion matrix   
def test_neural_net_with_confusion_matrix(model, loss_func, test_batches):
    """
    Evaluate the performance of the trained model on the test dataset and calculate a confusion matrix.
    """
    test_loss, test_accuracy = 0, 0
    all_preds = []
    all_labels = []
    
    model.eval()  # Set the model to evaluation mode
    with torch.inference_mode():  # Disable gradient computation
        for images, labels in test_batches:
            # Get predictions
            predictions = model(images)
            
            # Append predictions and labels for confusion matrix
            all_preds.append(predictions.argmax(dim=1).cpu().numpy())  # Convert to NumPy
            all_labels.append(labels.cpu().numpy())
            
            # Calculate loss
            test_loss += loss_func(predictions, labels).item()
            
            # Calculate accuracy
            test_accuracy += (predictions.argmax(dim=1) == labels).sum().item()
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate the confusion matrix
    class_labels=['bark','growl','whine']
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot()
    plt.show()
    #calculate recall
    recall = recall_score(all_labels, all_preds, average='weighted')
    #calculate precision 
    precision = precision_score(all_labels, all_preds, average='weighted')  
    #calculate f1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Average the loss and accuracy over the entire test dataset
    test_loss /= len(test_batches)
    test_accuracy /= len(test_batches.dataset)  # Divide by total dataset size
    
    # Print results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")
    print(f"recall (weighted):{recall:.4f},precision (weighted): {precision:.4f}, F1_score (weighted):{f1:.4f}")
    
    return test_loss, test_accuracy
