# -*- coding: utf-8 -*-

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8, progress_bar=True)
from tqdm.notebook import tqdm
tqdm.pandas()

from torchaudio.datasets.voxceleb1 import VoxCeleb1
import torchaudio

data = VoxCeleb1(root = "test/",download =True)

!git clone https://github.com/microsoft/UniSpeech.git

cd /home/jupyter/tests/UniSpeech/downstreams/speaker_verification

# %%writefile requirements.txt
# SoundFile==0.10.3.post1
# fire==0.4.0
# sentencepiece==0.1.96
# tqdm==4.62.0
# PyYAML==5.4.1
# h5py==3.3.0
# yamlargparse==1.31.1
# sklearn==0.0
# matplotlib==3.4.2
# torchaudio==0.9.0
# s3prl==0.3.1
# torch==1.9.0
# asteroid==0.4.4
# dtw-python==1.1.6
# pip install -U setuptools
# pip install fire
# pip install fairseq
# pip install s3prl
# pip install omegaconf==2.0.6
# pip install soundfile

cd UniSpeech/downstreams/speaker_verification

# model_name = "hubert_large"
# model_name = "wavlm_large"
model_name = "wavlm_base_plus"
wav1 = "test/wav/id10001/1zcIwhmdeo4/00001.wav"
wav2 = "test/wav/id10001/1zcIwhmdeo4/00001.wav"
use_gpu=False
# checkpoint='HuBERT_large_SV_fixed.th'
# checkpoint = "/content/wavlm_large_finetune.pth"
checkpoint = "wavlm_large_nofinetune.pth"

"""## Loading VOXCELEB1 H"""

!wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt

cd ..

import pandas as pd
df = pd.read_csv("./list_test_hard2.txt.2",sep = " ",header = None)

columns = ["test","speaker_1","speaker_2"]
df.columns = columns

df.head()

df.shape

df.head()

# cd UniSpeech/downstreams/speaker_verification

cd ..

"""## Model"""

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



# model_name = "hubert_large"
model_name = "wavlm_large"
# model_name = "wavlm_base_plus"
use_gpu=False
# checkpoint='HuBERT_large_SV_fixed.th'
# checkpoint = "/content/wavlm_large_finetune.pth"

checkpoint = "wavlm_large_nofinetune.pth"
model = init_model(model_name, checkpoint)

# pip install --force-reinstall soundfile

"""## THE TOTAL DATA SIZE WAS AROUND 40 GB. I WAS UNABLE TO EXECUTE SO I TESTED THE MODELS ON SAMPLE DATASET FOR 1000 SAMPLES"""

df_subset = df.sample(1000)

import os
df_subset["similarity_score"] = df_subset.progress_apply(lambda row : verification(model,os.path.join("test/wav",row["speaker_1"]),
                                                            os.path.join("test/wav",row["speaker_2"]),use_gpu,checkpoint),axis = 1)

df_subset["similarity_score_val"] = [t[0].item() for t in df_subset.similarity_score.tolist()]

df_subset.head()

y_pred = df_subset.similarity_score_val.tolist()
y = df_subset.test.tolist()

"""## EER Calculation"""

import numpy as np
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

print(F"EER for wavlm_large : {EER}")

