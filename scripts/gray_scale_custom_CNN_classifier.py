#import libraries 
import librosa
import matplotlib.pyplot as plt 
import pandas as pd 
from csv import writer
from prometheus_client import Summary
from tqdm import tqdm 
import torch
import torch.multiprocessing
from torch import nn
from torchsummary import summary
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import torch.optim as optim
import torchaudio
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import TensorDataset
import os

val_folder = "E:\\MO\\bioinformatcis\\First Semster\\python programming\\python training\\augmented_val"
train_folder = "E:\\MO\\bioinformatcis\\First Semster\\python programming\\python training\\augmented_train"
test_folder = "E:\\MO\\bioinformatcis\\First Semster\\python programming\\python training\\preaugmented dataset\\test"


def label_extractor(file_path):
    filename = os.path.basename(file_path)
    if "ba" in filename:
        return "ba"
    elif "gr" in filename:
        return "gr"
    elif "wh" in filename:
        return "wh"
    else:
        return "unknown"

audio_training_data = []
train_labels = []

for filename in os.listdir(train_folder):
    full_path = os.path.join(train_folder, filename)
    f_label = label_extractor(full_path)
    y, sr = librosa.load(full_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    target_length = sr  # 1 second in samples

    # Pad if shorter than 1 second
    if duration < 1:
        pad_len = target_length - len(y)
        noise = np.random.normal(0.01, 0.01, pad_len)
        y = np.concatenate((y, noise))

    # Trim if longer than 1 second
    elif duration > 1:
        y = y[:target_length]  # Take first 1 second

    # Now y is guaranteed to be 1 second long
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.T  # Shape: (timesteps, n_mels)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dim
    
    audio_training_data.append(spectrogram)
    train_labels.append(f_label)


audio_validation_data = []
val_labels = []

for filename in os.listdir(val_folder):
    full_path = os.path.join(val_folder, filename)
    f_label = label_extractor(full_path)
    y, sr = librosa.load(full_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    target_length = sr  # 1 second in samples

    # Pad if shorter than 1 second
    if duration < 1:
        pad_len = target_length - len(y)
        noise = np.random.normal(0.01, 0.01, pad_len)
        y = np.concatenate((y, noise))

    # Trim if longer than 1 second
    elif duration > 1:
        y = y[:target_length]  # Take first 1 second

    # Now y is guaranteed to be 1 second long
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.T  # Shape: (timesteps, n_mels)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dim

    
    audio_validation_data.append(spectrogram)
    val_labels.append(f_label)

audio_test_data = []
test_labels = []

for filename in os.listdir(test_folder):
    full_path = os.path.join(test_folder, filename)
    f_label = label_extractor(full_path)
    y, sr = librosa.load(full_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    target_length = sr  # 1 second in samples

    # Pad if shorter than 1 second
    if duration < 1:
        pad_len = target_length - len(y)
        noise = np.random.normal(0.01, 0.01, pad_len)
        y = np.concatenate((y, noise))

    # Trim if longer than 1 second
    elif duration > 1:
        y = y[:target_length]  # Take first 1 second

    # Now y is guaranteed to be 1 second long
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = spectrogram.T  # Shape: (timesteps, n_mels)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dim

    
    audio_test_data.append(spectrogram)
    test_labels.append(f_label)

#convert the labels into numbers
train_labels_num = []
for i in train_labels:
    if i == "ba":
        train_labels_num.append(0)
    elif i == "gr":
        train_labels_num.append (1)
    elif i == "wh":
        train_labels_num.append(2)

val_labels_num = []
for i in val_labels:
    if i == "ba":
        val_labels_num.append(0)
    elif i == "gr":
        val_labels_num.append (1)
    elif i == "wh":
        val_labels_num.append(2)

test_labels_num = []
for i in test_labels:
    if i == "ba":
        test_labels_num.append(0)
    elif i == "gr":
        test_labels_num.append(1)
    elif i == "wh":
        test_labels_num.append(2)

#converting list into numpy arrays 
train_data_np = np.array(audio_training_data)
train_labels = np.array(train_labels_num)
validation_data_np = np.array(audio_validation_data)
validation_labels = np.array(val_labels_num)
test_data_np = np.array(audio_test_data)
test_labels = np.array(test_labels_num)

# Convert NumPy arrays to PyTorch tensors
train_tensor_data = torch.tensor(train_data_np, dtype=torch.float32)
train_tensor_labels = torch.tensor(train_labels, dtype=torch.long)

validation_tensor_data = torch.tensor(validation_data_np, dtype=torch.float32)
validation_tensor_labels = torch.tensor(validation_labels, dtype=torch.long)

test_tensor_data = torch.tensor(test_data_np, dtype=torch.float32)
test_tensor_labels = torch.tensor(test_labels, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(train_tensor_data, train_tensor_labels)
validation_dataset = TensorDataset(validation_tensor_data, validation_tensor_labels)
test_dataset = TensorDataset(test_tensor_data, test_tensor_labels)

# Create DataLoaders
train_batches = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
validation_batches = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)
test_batches = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

#let's define some helpful parameters 
CHANNEL_COUNT = 1 # 1 channel as an image is a gray_scale image  
ACCURACY_THRESHOLD = 0.90
writer = SummaryWriter()

# Define a neural network class that inherits from PyTorch nn.Module.
class neuralNetworkV1(nn.Module):
    # The __init__ method is used to declare the layers that will be used in the forward pass.
    def __init__(self):
        super().__init__() # required because our class inherit from nn.Module
        # First convolutional layer with 1 input channels for RGB images, 16 outputs (filters).
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # Second convolutional layer with 16 input channels to capture features from the previous layer, 16 outputs (filters).
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        # Third and fourth convolutional layers with 16 and 10 output channels respectively.
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=2, padding=1)
        # Max pooling layer to reduce feature complexity.
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # ReLU activation function for introducing non-linearity.
        self.relu = nn.ReLU()
        # Flatten the 2D output from the convolutional layers for the fully connected layer.
        self.flatten = nn.Flatten()
        # Fully connected layer connecting to 1D neurons, with 3 output features for 3 classes.
        self.linear = nn.Linear(in_features=20, out_features=3)
    
    # define how each data sample will propagate in each layer of the network
    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.relu(self.conv3(x))
        x = self.pooling(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        try:
            x = self.linear(x)
        except Exception as e:
            print(f"Error : Linear block should take support shape of {x.shape} for in_features.")
        return x

our_model = neuralNetworkV1()

print("Model summary : ")
print(summary(our_model, (1, 44, 128)))

#define a new function to print out the training time 
def display_training_time(start, end):
    total_time = end - start
    print(f"Training time : {total_time:.3f} seconds")
    return total_time

#define a new function to display the training details at each iteration 
def display_training_infos(epoch, val_loss, train_loss, accuracy):
    val_loss = round(val_loss.item(), 2)
    train_loss = round(train_loss.item(), 2)
    accuracy = round(accuracy, 2)
    print(f"Epoch : {epoch}, Training loss : {train_loss}, Validation loss : {val_loss}, Accuracy : {accuracy} %")

#define a new function to calculate the accuracy between the true values and predicated values 
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

#function of training 
def train_neural_net(epochs, model, loss_func, optimizer, train_batches, val_batches, writer=None, device='cpu'):
    final_accuracy = 0
    train_accuracies, train_losses = [], []
    val_accuracies, val_losses = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_correct = 0
        total_train_samples = 0
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_batches, desc=f"Training Epoch {epoch+1}/{epochs}")):
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_func(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(predictions, 1)
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += labels.size(0)
            train_loss += loss.item()

        train_loss /= len(train_batches)
        train_accuracy = total_train_correct / total_train_samples
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        if writer:
            writer.add_scalar("training loss", train_loss, epoch)
            writer.add_scalar("training accuracy", train_accuracy, epoch)

        model.eval()
        val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        with torch.inference_mode():
            for images, labels in val_batches:
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)
                loss = loss_func(predictions, labels)

                val_loss += loss.item()
                _, preds = torch.max(predictions, 1)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += labels.size(0)

        val_loss /= len(val_batches)
        val_accuracy = total_val_correct / total_val_samples
        final_accuracy = val_accuracy
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if writer:
            writer.add_scalar("validation loss", val_loss, epoch)
            writer.add_scalar("validation accuracy", val_accuracy, epoch)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if train_accuracy  > 0.9:
            print(f"Stopping early as validation accuracy reached {val_accuracy:.4f} at epoch {epoch+1}.")
            break

   # Plot training and validation accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
    ax.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    ax.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    plt.show()

    return final_accuracy,val_accuracies,val_losses,train_accuracies,train_losses


#parameters
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
GRADIENT_MOMENTUM = 0.90

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(our_model.parameters(), lr=0.001)
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training the model
train_time_start_on_gpu = timer()
train_neural_net(100, our_model, criterion, optimizer, train_batches, validation_batches)
display_training_time(start=train_time_start_on_gpu,
                  end=timer())


#function of testing 
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

#process of testing 
test_loss,test_accuracy=test_neural_net_with_confusion_matrix(our_model,criterion,test_batches)
print(test_loss,test_accuracy)

