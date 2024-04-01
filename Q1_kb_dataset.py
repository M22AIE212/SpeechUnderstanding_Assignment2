# -*- coding: utf-8 -*-

import os
from pydub import AudioSegment
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from tqdm.notebook import tqdm
tqdm.pandas()
all_files = os.listdir("kb_data_clean_m4a/hindi/test_known/audio")

pip install pydub

len(all_files)

"""## Generating Pair data"""

all_files_pairs = [(all_files[i], all_files[j])  for i in range(len(all_files)) for j in range(i+1, len(all_files)) if i != j]

print("All possible pairs:", len(all_files_pairs))

"""## Taking a subset of 1000 pairs"""

import random
sample_kb_data = random.sample(all_files_pairs,1000)

sample_kb_data_label = [(x,y,1) if x.split("-")[-2]  ==  y.split("-")[-2] else (x,y,0) for x,y in sample_kb_data ]

len(sample_kb_data_label)

"""## Converting m4a to wav"""

# cd /UniSpeech/downstreams/speaker_verification

for file1,file2 ,_ in  sample_kb_data_label :
    m4a_file1 = os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',file1)
    m4a_file2 = os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',file2)

    wav1_filename = f'{file1.split(".")[0]}.wav'
    wav2_filename = f'{file2.split(".")[0]}.wav'

    sound1 = AudioSegment.from_file(m4a_file1, format='m4a')
    file_handle1 = sound1.export(os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',wav1_filename), format='wav')

    sound2 = AudioSegment.from_file(m4a_file2, format='m4a')
    file_handle2 = sound2.export(os.path.join('kb_data_clean_m4a/hindi/test_known/audio/',wav2_filename), format='wav')

"""## Data"""

df_kb = pd.DataFrame(sample_kb_data_label)
df_kb.columns = ["speaker1","speaker2","test"]

df_kb.head()

df_kb.test.value_counts()

"""## Speech Verification Model"""

cd /UniSpeech/downstreams/speaker_verification

import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import numpy as np
import IPython.display as ipd
import soundfile as sf
import io
from google.cloud import storage

# BUCKET = 'wmt-mlp-p-intlctlg-export-bucket'

# # Create a Cloud Storage client.
# gcs = storage.Client()


# # Get the bucket that the file will be uploaded to.
# bucket = gcs.get_bucket(BUCKET)
MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification(model,  wav1, wav2, use_gpu=True, checkpoint=None):




    # specify a filename
    file_name1 = wav1
    file_name2 = wav2

    wav1, sr1  = sf.read(file_name1)
    wav2, sr2  = sf.read(file_name2)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
#     print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return sim

cd ..

cd ..

cd ..

"""### Model : hubert_large"""

model_name = "hubert_large"
use_gpu=False
checkpoint='HuBERT_large_SV_fixed.th'
model = init_model(model_name, checkpoint)
df_kb["similarity_score"] = df_kb.progress_apply(lambda row : verification(model,os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker1"].split(".")[0] + ".wav"),
                                                            os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker2"].split(".")[0] + ".wav"),use_gpu,checkpoint),axis = 1)
df_kb["similarity_score_val"] = [t[0].item() for t in df_kb.similarity_score.tolist()]
y_pred = df_kb.similarity_score_val.tolist()
y = df_kb.test.tolist()

fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

print(F"EER for {model_name} : {EER}")

"""### Model : wavlm_large"""

model_name = "wavlm_large"
checkpoint = "wavlm_large_nofinetune.pth"
model = init_model(model_name, checkpoint)

df_kb["similarity_score"] = df_kb.progress_apply(lambda row : verification(model,os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker1"].split(".")[0] + ".wav"),
                                                            os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker2"].split(".")[0] + ".wav"),use_gpu,checkpoint),axis = 1)

df_kb["similarity_score_val"] = [t[0].item() for t in df_kb.similarity_score.tolist()]
y_pred = df_kb.similarity_score_val.tolist()
y = df_kb.test.tolist()

fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

print(F"EER for {model_name} : {EER}")

"""### Model : wavlm_base_plus"""

model_name = "wavlm_base_plus"
checkpoint='wavlm_base_plus_nofinetune.pth'
model = init_model(model_name, checkpoint)

df_kb["similarity_score"] = df_kb.progress_apply(lambda row : verification(model,os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker1"].split(".")[0] + ".wav"),
                                                            os.path.join("kb_data_clean_m4a/hindi/test_known/audio",row["speaker2"].split(".")[0] + ".wav"),use_gpu,checkpoint),axis = 1)
df_kb["similarity_score_val"] = [t[0].item() for t in df_kb.similarity_score.tolist()]
y_pred = df_kb.similarity_score_val.tolist()
y = df_kb.test.tolist()

fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

print(F"EER for {model_name} : {EER}")

