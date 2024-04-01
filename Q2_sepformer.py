# -*- coding: utf-8 -*-

!git clone https://github.com/JorisCos/LibriMix.git

"""## Librimix Data Generation"""

pip install torchmetrics

cd LibriMix

# Commented out IPython magic to ensure Python compatibility.
# 
# %%bash generate_librimix.sh storage_dir

pip install speechbrain

"""## Data Load"""

from google.colab import drive
drive.mount('/content/drive')

pwd

!cp /content/drive/MyDrive/Assignments/SU/mix_clean.zip /content

!unzip /content/mix_clean.zip

!cp /content/drive/MyDrive/Assignments/SU/test-clean.zip /content

!unzip /content/test-clean.zip

import pandas as pd
pd.set_option('display.max_colwidth', None)

df_test_clean_metadata = pd.read_csv("/content/drive/MyDrive/Assignments/SU/mixture_test_mix_clean.csv")

df_test_clean_metadata[df_test_clean_metadata.mixture_ID == "1089-134686-0000_121-127105-0031"]

df_test_clean_metadata.shape

import os
all_files = os.listdir("/content/LibriMix/mix_clean")

for file in all_files :
  if file.endswith(".txt"):
    print(file)

"""## Train Test Split"""

all_files_list = []
for mixture_id in df_test_clean_metadata.mixture_ID.tolist():
  source1_id = mixture_id.split("_")[0]
  source2_id = mixture_id.split("_")[1]

  mixture_path = "/content/LibriMix/mix_clean/" + mixture_id + ".wav"
  source_1_path = "/content/LibriMix/test-clean/" + "/".join(source1_id.split("-")[:-1]) + "/" + source1_id + ".flac"
  source_2_path = "/content/LibriMix/test-clean/" + "/".join(source2_id.split("-")[:-1]) + "/" + source2_id + ".flac"

  all_files_list.append((source_1_path,source_2_path,mixture_path))

from sklearn.model_selection import train_test_split

# Perform 70:30 train-test split
train_data, test_data = train_test_split(all_files_list, test_size=0.3, random_state=42)

# Print the train and test data
print("Train data:", len(train_data))
print("Test data:", len(test_data))

"""## Pretrained Speechformer"""

from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import soundfile as sf
import librosa
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torch import tensor
model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

si_sdr_metric =ScaleInvariantSignalDistortionRatio()
si_snr_metric = ScaleInvariantSignalNoiseRatio()

mixed_signal = tensor([0.2, 0.4, 0.6, 0.8, 1.0])
si_sdr_mixed = si_sdr_metric(mixed_signal.unsqueeze(0), mixed_signal.unsqueeze(0))
si_sdr_mixed

for index,(source_1_path,source_2_path,mixture_path) in enumerate(test_data):
  print()
  print(f"Separating Audio {index + 1} from test set :")
  print()
  # for custom file, change path
  audio_mixed,sr_mix = sf.read(mixture_path)
  audio_mixed = tensor(librosa.resample(audio_mixed, orig_sr=sr_mix, target_sr=8000))

  est_sources = model.separate_file(path=mixture_path)

  predicted_audio_1 = est_sources[:, :, 0].detach().cpu().numpy()[0]
  predicted_audio_2 = est_sources[:, :, 1].detach().cpu().numpy()[0]

  actual_audio1, sr1 = sf.read(source_1_path)
  actual_audio2, sr2 = sf.read(source_2_path)

  actual_audio1_resampled = librosa.resample(actual_audio1, orig_sr=sr1, target_sr=8000)
  actual_audio2_resampled = librosa.resample(actual_audio2, orig_sr=sr2, target_sr=8000)

  min_len = min(len(actual_audio1_resampled), len(actual_audio2_resampled))
  actual_audio1_resampled = actual_audio1_resampled[:min_len]
  actual_audio2_resampled = actual_audio2_resampled[:min_len]

  print("ScaleInvariantSignalDistortionRatio : ")

  # Compute SI-SDR for separated and mixed signals
  si_sdr_separated = si_sdr_metric(tensor(actual_audio1_resampled), tensor(predicted_audio_1) ) + si_sdr_metric(tensor(actual_audio2_resampled), tensor(predicted_audio_2) )
  si_sdr_mixed = si_sdr_metric(tensor(actual_audio1_resampled), tensor(audio_mixed) ) + si_sdr_metric(tensor(actual_audio2_resampled), tensor(audio_mixed) )

  # Compute SI-SDRi
  si_sdr_improvement = si_sdr_separated - si_sdr_mixed

  # Print the SI-SDRi value
  print("SI-SDRi:", si_sdr_improvement.item(), "dB")

  print("ScaleInvariantSignalNoiseRatio : ")
  si_snr_separated = si_snr_metric(tensor(actual_audio1_resampled), tensor(predicted_audio_1) ) + si_snr_metric(tensor(actual_audio2_resampled), tensor(predicted_audio_2) )
  si_snr_mixed = si_snr_metric(tensor(actual_audio1_resampled), tensor(audio_mixed) ) + si_snr_metric(tensor(actual_audio2_resampled), tensor(audio_mixed) )

  # Compute SI-SNRi
  si_snr_improvement = si_snr_separated - si_snr_mixed

  # Print the SI-SNRi value
  print("SI-SNRi:", si_snr_improvement.item(), "dB")

"""## Gradio"""

pip install gradio==3.50

from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import gradio as gr

model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')

def speechbrain(aud):
  est_sources = model.separate_file(path=aud)
  torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
  torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
  return "source1hat.wav", "source2hat.wav"

inputs = gr.Audio(label="Input Audio", type="filepath")
outputs =  [
  gr.Audio(label="Output Audio One", type="filepath"),
  gr.Audio(label="Output Audio Two", type="filepath")
]

title = "Speech Seperation"
description = "Gradio demo for Speech Seperation by SpeechBrain. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.13154' target='_blank'>Attention is All You Need in Speech Separation</a> | <a href='https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix/separation' '_blank'>Github Repo</a></p>"

gr.Interface(speechbrain, inputs, outputs, title=title, description=description, article=article).launch(debug=True)