#import libraries 
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt 
import pandas as pd 
from csv import writer
from pydub import AudioSegment
from tqdm import tqdm 
from PIL import Image
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
from torchvision.transforms import ToPILImage
import os


#in this model, we crated a CNN network using Pytorch framework, to test the ability of the model to predict the typee of the dog sound
#firstly, we are going to segment the audio files into (0.3 second) audio snnipets to train and test the model:

#craete a new function to segment the audio files
def audio_segmenter(input_file,output_file, format):
    #intrduing the sounds 
    for f in os.listdir(input_file):
        pre_sound=AudioSegment.from_file(f,format=format)
    #introduce snipets, which's going to be half a second 
    interval=300
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_file):
        os.makedirs(output_file)
        
    audio_length=len(pre_sound)
        
    #loop through the sound files:
    for i in range(0, audio_length,interval):
        #we start from second 0 
        start_time=i
        #we still snipping the audio until we reach a point we cannot produce more 5 seconds snippets 
        end_time=min(i+interval,audio_length) 
        #extract snippets 
        snip=pre_sound[start_time:end_time]
        # Export the snippet with a unique filename
        snippet_filename = os.path.join(output_file, f"snippet_{start_time // 1000}-{end_time // 1000}.{format}")
        snip.export(snippet_filename, format=format)
        print(f"Exported {snippet_filename}")


#after segmenting the audio files into snippets, we have to filter out the unmeaningful frequencies from the audios, by applying a low frequency and highy frequency filter:

def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')
  
#applying a low frequency filter from 10Hz to a high frequency filter 10000Hz and then plot the spectrogram in a gray scale figures with (175*174) pixels:
def butter_bandpass_filter(input_file, lowcut, highcut,order,output_fig_path):
    # Loop over each file in the input directory
    for f in os.listdir(input_file):
        # Full path to the audio file
        file_path = os.path.join(input_file, f)

        # Load the audio file
        wave_form, sample_rate = torchaudio.load(file_path)   
        waveform = wave_form.numpy()[0]  # Extract the first channel if stereo

        # Filter features
        b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
        filtered_signal = lfilter(b, a, waveform)
        
        # Create a spectrogram plot
        fig, ax = plt.subplots(1, 1, figsize=(2,2))
        ax.specgram(filtered_signal, Fs=sample_rate, NFFT=1024, cmap="gray")
        ax.axis('off')  # Remove axes for a cleaner look

        # Save the plot to the specified output path
        output_file_path = os.path.join(output_fig_path, f"{os.path.splitext(f)[0]}_spectrogram.png")
        plt.savefig(output_file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)  # Close the plot to free up memory


#now we have to upload our training,validation and testing dataset 
#firstly we have to introduce pytorch transform, that will transform the datasets 

#Raw_dataset path
Dataset_path=""
#path to output paths
train_path=""
validation_path=""
testing_path=""

transform=transforms.ToTensor()
dataset = datasets.ImageFolder(Dataset_path, transform=transform)

#split dataset into test and train sets
#test and validation ratio
val_ratio = 0.1
test_ratio = 0.1
#dataset volume
val_size = int(val_ratio * len(dataset))
test_size=int(test_ratio *len(dataset))
train_size =int(len(dataset) - (val_size+test_size))

# Ensure reproducibility
torch.manual_seed(42)

# Split dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print(f"{train_size} images for training, {val_size} images for validation, {test_size}images for testing")

#after splitting three datasets into three parts (training,testing and validation), we save each set in separate file:
def save_images(dataset, folder):
    os.makedirs(folder, exist_ok=True)  # Create directory if not exists
    to_pil = ToPILImage()  # Convert tensor to PIL image
    
    # Access the image paths from the original dataset
    for i, (img, label) in enumerate(dataset):
        # Extract the original image path (this works because dataset.imgs has the full paths)
        original_image_path = dataset.dataset.imgs[dataset.indices[i]][0]
        filename = os.path.basename(original_image_path)  # Extract the filename from the path

        # Convert tensor to PIL image and save
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(folder, filename))  # Save with the original filename


# Save train, validation, and test datasets
save_images(train_dataset, train_path)
save_images(val_dataset, validation_path)
save_images(test_dataset, testing_path)

#load the images as a 16 tensor images batches 
#Load testing dataset into batches
train_batches = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

#Load testing dataset into batches
validation_batches = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

#Load testing dataset into batches
test_batches = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=4)

#let's define some helpful parameters 
CHANNEL_COUNT = 1 # 1 channel as an image is a gray_scale image  or it can be 3
ACCURACY_THRESHOLD = 0.90
writer = SummaryWriter()

# in the next section, we are going to set up the model which will be consisted of four types of layers:
# 1) Convolution layer (Conv).
# 2) Pooling (pool).
# 3) Fully connected (FC).
# 4) Rectified linear unit (Relu).


# Define a neural network class that inherits from PyTorch nn.Module.
class neuralNetworkV1(nn.Module):
    # The __init__ method is used to declare the layers that will be used in the forward pass.
    def __init__(self):
        super().__init__() # required because our class inherit from nn.Module
        # First convolutional layer with 1 input channels for RGB images, 16 outputs (filters).
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
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
        self.linear = nn.Linear(in_features=90, out_features=3)
    
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

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(our_model.parameters(), lr=0.001)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#training the model
MAX_EPOCHS = 100
LEARNING_RATE = 0.01
GRADIENT_MOMENTUM = 0.90
train_time_start_on_gpu = timer()
train_neural_net(100, our_model, criterion, optimizer, train_batches, validation_batches)
display_training_time(start=train_time_start_on_gpu,
                  end=timer())

#test the model 
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

