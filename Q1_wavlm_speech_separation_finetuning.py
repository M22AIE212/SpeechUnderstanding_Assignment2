# -*- coding: utf-8 -*-
!wget https://indic-asr-public.objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar

pip install pydub

import tarfile
tar = tarfile.open("testkn_audio.tar")
tar.extractall()
tar.close()

"""## Generating Pair data"""

import os
from pydub import AudioSegment
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
from tqdm.notebook import tqdm
tqdm.pandas()
all_files = os.listdir("kb_data_clean_m4a/hindi/test_known/audio")

all_files_pairs = [(all_files[i], all_files[j])  for i in range(len(all_files)) for j in range(i+1, len(all_files)) if i != j]

print("All possible pairs:", len(all_files_pairs))

import random
sample_kb_data = random.sample(all_files_pairs , 1000)

sample_kb_data_label = [(x,y,1) if x.split("-")[-2]  ==  y.split("-")[-2] else (x,y,0) for x,y in sample_kb_data ]

for file1,file2 ,_ in  tqdm(sample_kb_data_label) :
    m4a_file1 = os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',file1)
    m4a_file2 = os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',file2)

    wav1_filename = f'{file1.split(".")[0]}.wav'
    wav2_filename = f'{file2.split(".")[0]}.wav'

    sound1 = AudioSegment.from_file(m4a_file1, format='m4a')
    file_handle1 = sound1.export(os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',wav1_filename), format='wav')

    sound2 = AudioSegment.from_file(m4a_file2, format='m4a')
    file_handle2 = sound2.export(os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',wav2_filename), format='wav')

data = [((os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',x.split(".")[0] + ".wav"),
         os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',y.split(".")[0] + ".wav"))) for (x,y,_) in sample_kb_data_label]
labels = [ _  for (x,y,_) in sample_kb_data_label]

data[0],labels[0]

"""## Dataset"""

import torch
import torchaudio
from transformers import WavLMForSequenceClassification, AdamW
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define constants
SAMPLING_RATE = 16000  # Model's expected sampling rate
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

# Define preprocessing function
def preprocess_audio(audio_path):
    waveform, _ = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=waveform.size(1), new_freq=SAMPLING_RATE)(waveform)
    waveform = waveform / torch.max(torch.abs(waveform))
    return waveform

# Custom dataset class
class SpeakerVerificationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        waveform1 = preprocess_audio(pair[0])
        waveform2 = preprocess_audio(pair[1])
        label = self.labels[idx]
        return waveform1, waveform2, label

# Load model and modify final layer
model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-large")
# Define the reduction layer
reduction_layer = torch.nn.Linear(256, 1)

# Replace the final layer with the reduction layer
model.classifier = reduction_layer

# Define loss function and optimizer
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Prepare dataset and dataloader
dataset = SpeakerVerificationDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset[0][1].shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs1, inputs2, targets = batch

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        outputs1 = model(inputs1.squeeze(1))
        outputs2 = model(inputs2.squeeze(1))
        loss = criterion(outputs1.logits, outputs2.logits, targets.float())  # Calculate loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")

    scheduler.step()

# Save the trained model
model_save_path = "/content/drive/MyDrive/Assignments/SU/speaker_verification_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

"""## EER%"""

print(F"EER for wavlm_ : {EER}")

"""## Gradio"""

pip install gradio

!cp /content/drive/MyDrive/Assignments/SU/speaker_verification_model.pth /content

pwd

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import torch
# import torchaudio
# from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
# import gradio as gr
# 
# # Load the pretrained speaker verification model
# model.load_state_dict(torch.load("/content/drive/MyDrive/Assignments/SU/speaker_verification_model.pth",map_location=torch.device('cpu')))
# 
# # Define preprocessing function
# def preprocess_audio(audio_path):
#     waveform, _ = torchaudio.load(audio_path)
#     waveform = torchaudio.transforms.Resample(orig_freq=waveform.size(1), new_freq=SAMPLING_RATE)(waveform)
#     waveform = waveform / torch.max(torch.abs(waveform))
#     return waveform
# 
# # Define the function to perform speaker verification
# def speaker_verification(audio_file1, audio_file2):
#     # Load and process audio files
#     waveform1 = preprocess_audio(audio_file1)
#     waveform2 = preprocess_audio(audio_file2)
# 
#     # Forward pass through the model
#     with torch.no_grad():
#         outputs1 = model(waveform1.squeeze(1)).logits
#         outputs2 = model(waveform2.squeeze(1)).logits
# 
#     # Calculate the cosine similarity between the embeddings
#     similarity_score = torch.nn.functional.cosine_similarity(outputs1, outputs2).item()
# 
#     return similarity_score
# 
# # Define Gradio interface
# inputs = [
#     gr.Audio(label="Audio File One", type="filepath"),
#     gr.Audio(label="Audio File Two", type="filepath")
# ]
# output = gr.Textbox(label="Similarity Score")
# 
# title = "Speaker Verification"
# description = "Gradio demo for Speaker Verification using a pretrained model."
# article = "<p style='text-align: center'><a href='https://huggingface.co/models' target='_blank'>Hugging Face Model Hub</a></p>"
# 
# gr.Interface(speaker_verification, inputs, output, title=title, description=description, article=article).launch(debug=True)
#