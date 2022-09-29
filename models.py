import torch
import torch.nn as nn
from transformers import BartModel
import torch.nn.functional as F
import numpy as np

#################################
#### Fusion Attention Module ####
#################################

#FAM_block : Fusion Attention Module tel que décrit dans le papier "M2FNet: Multi-modal Fusion Network for Emotion Recognition in Conversation"

##video : main_AUs fournit des features de dimensions 825*nbre_frames, qu'on réduit à la taille 512 par un transformers
##audio : features audio, passées dans un transformers et réduites à la taille 512
##text : BERT -> réduction taille 512

class FAM_block(nn.Module):
    def __init__(self, dropout=0.5):
        super(FAM_block, self).__init__()
        #MultiHeadAttention
        self.multiatt = torch.nn.MultiheadAttention(512, 8, dropout=0.5)
        #2 MHA en parallèle = sortie de taille (512*8)*2 -> on veut une feature finale de taille 512
        self.fc_apres_multiatt = nn.Linear(512*3, 512)
        
    def forward(self, video, audio, text):

        # Fusion par MHA

        # Suppression d'une ligne en dimension 0 sur audio et vidéo
        # (sinon pb de dimensions)

        if (int(audio.shape[0]) == 2):
            audio = audio[0]
            audio = audio.unsqueeze(dim=0)

        if (int(video.shape[0]) == 2):
            video = video[0,]
            video = video.unsqueeze(dim=0)

        audiotext, _ = self.multiatt(text, audio, text)
        videotext, _ = self.multiatt(text, video, text)

        # Concaténation des vecteurs de sortie, et passage dans une couche FC (4096*2, 512)

        concat_sortie = torch.cat((audiotext, videotext, text), 1)

        concat_sortie = self.fc_apres_multiatt(concat_sortie)

        return video, audio, concat_sortie

#FAM : plusieurs blocks de Fusion Attention Module, puis concaténation des vecteurs résultants
##

class FAM(nn.Module):
    def __init__(self):
        super(FAM, self).__init__()
        self.FAMblock = FAM_block()
    
    def forward(self, video, audio, text, num_blocks):
        for _ in range(num_blocks):
            video, audio, text = self.FAMblock(video, audio, text)
        x = torch.cat((video, audio, text), 1)
        return x

###############
#### Utils ####
###############

#CompresseurTorch : fonction qui compresse un vecteur torch selon un facteur de réduction donné
##vecteur : vecteur à compresser
##facteur : facteur de réduction de la compression

def CompresseurTorch(vecteur, facteur):
    return torch.mean(torch.reshape(vecteur, (-1, facteur)), 1)

#SwiGLU : fonction d'activation des transformers
##x : hidden state juste après le FFN

class SwiGLU(nn.Module):
    def forward(self, x):
        return F.silu(x) * x

# Fonction de compression des vecteurs Numpy
##variable vecteur : vecteur au format Numpy à compresser
##variable facteur : de combien doit être divisée la taille du vecteur

def compression(vecteur, facteur):
    return vecteur.reshape(-1, facteur).mean(axis=1)

#Reducteur : fonction qui fait passer la sortie compressée du BART (taille (25216,)) dans un FC->gelu pour réduire la taille des features
##x : 1 vecteur de features issues du texte

class Reducteur(nn.Module):
    def __init__(self):
        super(Reducteur, self).__init__()
        self.fc = nn.Linear(403456, 512)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        return x

#FAM_to_prediction : à partir de la sortie du FAM, calcule une prédiction
##x : vecteur final à faire passer dans un FC(?, 1024) -> GELU -> FC(1024, 21) -> GELU

class FAM_to_prediction(nn.Module):
    def __init__(self):
        super(FAM_to_prediction, self).__init__()
        self.fc1 = nn.Linear(512*3, 512)
        self.fc2 = nn.Linear(512, 8)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        return(x)
    
######################
#### Transformers ####
######################

#modalite_transfo : pour avoir une représentation des features extraites pour chaque modalité
#qui tient compte du contexte global du dialogue

##cste num_layers : nombre de blocks de transformers consécutifs
##cste dim_entree : dimensions des vecteurs passés en entrée des blocks de transformers
##var modalite : features extraites pour la modalité "modalite" (suite de vecteurs de tailles variables,
## sans doute nécessité d'appliquer 1 masque pour homogénéiser leur taille)

class Modalite_transfo(nn.Module):
    def __init__(self, dim_entree):
        super(Modalite_transfo, self).__init__()
        #Chaque block de transformers est sous la forme d'encoderlayer
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=dim_entree, nhead=8, activation=SwiGLU(), batch_first=True)
        #Couche FC pour transformer la sortie des transformers en un vecteur de taille 512
        #self.fc_512 = nn.Linear(8*dim_entree, 512)
        self.gelu = nn.GELU()
    def forward(self, x):
        for _ in range(6):
            id = x
            x = self.encoderlayer(x)
            x += id
        x = self.gelu(x)
        return x

##############################
#### Fusion des modalités ####
##############################

# model_fusion : modèle global qui effectue la featurization à l'échelle de l'utterance pour chaque modalité,
# fait passer ces représentations featurisées dans des transformers, puis fusionne les modalités à l'aide d'un
# Fusion Attention Module
# Text -> BART -> 6 transformers encoders ------------------v
# Video -> extract_AUs -> 6 transformers encoders---------> Fusion Attention Module -> FC -> GeLU -> FC -> GeLU -> prédiction (21 labels)
# Audio -> extract_wav_features -> 6 transformers encoders--^

class model_fusion(nn.Module):
    def __init__(self, dropout=0.5):
        super(model_fusion, self).__init__()
        #BART pour le texte
        self.bart = BartModel.from_pretrained('facebook/bart-large')
        #Transformers à appliquer sur les AUs featurisés
        #En paramètre : la dimension du vecteur d'entrée
        self.transfoAU = Modalite_transfo(dim_entree=512)
        self.transfoaudio = Modalite_transfo(dim_entree=512)
        self.transfotext = Modalite_transfo(dim_entree=512)
        #Fusion Attention Module, pour la fusion de modalités
        self.FAM_layer = FAM()
        self.FAM2PRED = FAM_to_prediction()
        #Pour réduire le vecteur de features du texte
        self.reducteur = Reducteur()
        #Pour régulariser le BART
        self.dropout = nn.Dropout(dropout)
        #Pour réduire la taille des sorties
        self.fc_audio1 = nn.Linear(409600, 512)
        self.fc_video1 = nn.Linear(32768, 512)
        #GeLU
        self.gelu = nn.GELU()

    def forward(self, video, audio, input_id, mask):
        
        # Texte : bart -> dropoout(0.5) -> FC(403456, 512) -> GeLU -> 6 couches de transformers -> sortie 8*512
        pooled_output = torch.flatten((self.bart(input_ids= input_id, attention_mask=mask)).last_hidden_state, start_dim=1) #Taille (403456,)

        dropout_output = self.dropout(pooled_output)
        linear_output = self.reducteur(dropout_output) #Taille (512,)
        transfo_output = self.transfotext(linear_output)
        
        # Vidéo : extract_AU_features -> 6 couches de transformers

        video = self.fc_video1(video)
        video = self.gelu(video)
        video = self.transfoAU(video)


        # Audio : audio_featurizer -> 6 couches de transformers

        audio = self.fc_audio1(audio)
        audio = self.gelu(audio)
        audio = self.transfoaudio(audio)

        # Fusion :

        x = self.FAM_layer(video, audio, transfo_output, 10)
        x = self.FAM2PRED(x)

        return x
