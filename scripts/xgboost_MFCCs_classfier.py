#import libraries
import os 
import numpy as np 
import librosa 
import matplotlib.pyplot as plt 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path

#in this model, we are going to use Xgboost model to classify sounds based on their MFCCs features, first we have to extract audio features using librosa library:
def audio_feature_extractor(audio_file):
    y, sr=librosa.load(audio_file)
    MFCCs=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)
    return np.mean(MFCCs.T,axis=0)

#audio files paths
barks_audio_files = Path("")
growls_audio_files= Path("")
whines_audio_files= Path("")

#we are going to extract bark feature then growls and finally whines 
barks_features=[]
#looping thorugh the audio files
for file in os.listdir(barks_audio_files):
    if file.endswith('.wav') or file.endswith('.mp3'):
        # Get the full path of the audio file
        file_path_b = os.path.join(barks_audio_files, file)
        
        # Extract features using your feature extraction function
        FC = audio_feature_extractor(file_path_b)  # Pass full path here
        barks_features.append(FC)

#print out
print(len(barks_features))

growls_features=[]
#looping thorugh the audio files
for g_file in os.listdir(growls_audio_files):
    if g_file.endswith('.wav') or g_file.endswith('.mp3'):
        # Get the full path of the audio file
        file_path_g = os.path.join(growls_audio_files, g_file)
        
        # Extract features using your feature extraction function
        FCG = audio_feature_extractor(file_path_g)  # Pass full path here
        growls_features.append(FCG)

#print out
print(len(growls_features))

whines_features=[]
#looping thorugh the audio files
for w_file in os.listdir(whines_audio_files):
    if w_file.endswith('.wav') or w_file.endswith('.mp3'):
        # Get the full path of the audio file
        file_path_w = os.path.join(whines_audio_files, w_file)
        
        # Extract features using your feature extraction function
        FCW = audio_feature_extractor(file_path_w)  # Pass full path here
        whines_features.append(FCW)

#print out
print(len(whines_features))

#construct a new list for lables, where: (0) stands for bark, (1) for a growl, and (2) for whine
labels= [0]* len(barks_features) + [1] * len(growls_features) + [2] * len(whines_features)
print(len(labels))

#concatenate the three feature lists 
features_lst = barks_features + growls_features + whines_features
print(len(features_lst))

#convert the both lists to numpy array
X=np.array(features_lst)
y=np.array(labels)

#features normalizations
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split the dataset to training and testing 
x_train,x_test,y_train,y_test=train_test_split(X_scaled , y , test_size = 0.2 , random_state=42)

#launch the model 
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the model
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

