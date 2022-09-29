# Librairie de récupération du fichier vidéo
import moviepy.editor as mp
from sklearn.preprocessing import StandardScaler
# Extracteur de features
from transformers import Wav2Vec2FeatureExtractor
# Torch
import torch
import torchaudio
# Numpy
import numpy as np
# Gestion fichiers
import os
# Compression
from models import compression

####################################
#### Initialisation des modèles ####
####################################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Torch
#bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
#model = bundle.get_model().to(device)

#HuggingFace feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, 
                                             padding_value=0.0, do_normalize=True, return_attention_mask=False)

#####################################################################
#### Fonction d'extraction de features par Wav2vec2.0 (Facebook) ####
#####################################################################

##variable file : fichier vidéo
##variable dir : emplacement du dossier où sauvegarder les fichiers audio

def audio_featurizer(file, dir):
    #Récupération du clip et extraction de la piste audio
    clip = mp.VideoFileClip(dir+file)
    if not os.path.exists(dir+file[:-4]+'/'+file[:-4]+'.wav'):
        try:
            clip.audio.write_audiofile(dir+file[:-4]+'/'+file[:-4]+'.wav')
        except IOError as e:
            print(e)
    #Récupération du vecteur associé à la piste audio
    waveform, sample_rate = torchaudio.load(dir+file[:-4]+'/'+file[:-4]+'.wav')
    waveform = waveform.to(device)
    features = np.concatenate((np.array(feature_extractor(waveform, sampling_rate=16000).input_values[0][0]), np.array(feature_extractor(waveform, sampling_rate=16000).input_values[0][1])), 0)
    features = compression(np.pad(features, [(0, 4096000-len(features))]), 10)
    """# Harmonisation des sample rates (le modèle Wav2Vec2.0 suppose un certain sampling rate)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    # Extraction des features
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)"""
    return torch.Tensor(features)