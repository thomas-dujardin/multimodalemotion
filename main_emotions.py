##utils
import pandas as pd
import numpy as np
##Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
##modèle pour le texte
from transformers import BartTokenizer, BartModel
##tqdm pour le training
from tqdm import tqdm
##pour rééquilibrer l'échantillon
from sklearn.utils import class_weight
##pour évaluer le score F1
from sklearn.metrics import f1_score
##pour les accès aux dossiers et aux fichiers
import os
##fonctions d'extraction de features pour audio+AUs
from extract_wav_features import audio_featurizer
from main_AUs import extract_AU_features
##modèles
from models import model_fusion
##fait fonctionner urllib (pour télécharger les .pth essentiels au fonctionnement de py-feat)
import os
os.environ['http_proxy']=''
##pour trier les noms des fichiers
from natsort import natsorted

#############################################
#### HYPERPARAMETRES POUR L'ENTRAÎNEMENT ####
#############################################

EPOCHS = 5
model = model_fusion()
LR = 3e-4
bs = 32 #batch_size

############################################################################################
#### Emplacement des dossiers de train, dev, test et .csv avec les textes et les labels ####
############################################################################################

#Version locale
#root = '/Users/dujardinth/Documents/Python/data_meld/MELD.Raw/'
#Version globale
root = '/usr/users/multimodalemotion/tdujardin/MELD.Raw/'

train, dev, test = root+'train/', root+'dev/', root+'test/'

train_text = pd.read_csv(root+'train_sent_emo.csv')
dev_text = pd.read_csv(root+'dev_sent_emo.csv')
test_text = pd.read_csv(root+'test_sent_emo.csv')

#######################
#### Labellisation ####
#######################

# merge_emotion_sentiment : à partir d'un DataFrame df (qui contient les colonnes Emotion et Sentiment),
# renvoie un DataFrame qui contient la colonne 'emotionsentiment' qui résulte de la concaténation
# des colonnes précitées.
##df : DataFrame (train_text, dev_text et test_text)

def merge_emotion_sentiment(df):
    df['emotionsentiment'] = df['Emotion']+df['Sentiment']
    return df

train_text, dev_text, test_text = merge_emotion_sentiment(train_text), merge_emotion_sentiment(dev_text), merge_emotion_sentiment(test_text)

# Dictionnaire des 21 labels utilisés (7 émotions, 3 sentiments -> 21 émotion-sentiment)

labels = {'angernegative':0,
        'disgustnegative':1,
        'fearnegative':2,
        'joypositive':3,
        'neutralneutral':4,
        'sadnessnegative':5,
        'surprisenegative':6,
        'surprisepositive':7
          }

# Combien d'exemples utiliser
pct_retained = 4762

# Pondération de la fonction de perte, afin de tenir compte des disparités entre le nbre d'occurrences des 21 classes

class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes= np.unique(((train_text).head(pct_retained)).emotionsentiment), y= ((train_text).head(pct_retained)).emotionsentiment)
class_weights = torch.tensor(class_weights,dtype=torch.float)

##########################################
#### Fonction qui retourne des batchs ####
##########################################

# Tokenizer pour l'encodage du texte
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

#Dataset : retourne un batch multimodal
##df : DataFrame contenant les textes
##df_videos : DataFrame contenant les AUs encodées
##df_audios : DataFrame contenant les audios encodés

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df_texts, dir):

        self.labels = [labels[label] for label in df_texts['emotionsentiment']]
        self.texts = [(tokenizer(text,
         padding='max_length',
         max_length=394,
          truncation=True,
           return_tensors="pt")) for text in df_texts['Utterance']]
        self.videos = [(extract_AU_features(file, dir, [0.5, 0.5])) for file in natsorted(os.listdir(dir)) if file.endswith('.mp4')]
        self.audios = [(audio_featurizer(file, dir)) for file in natsorted(os.listdir(dir)) if file.endswith('.mp4')]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of text inputs
        return self.texts[idx]

    def get_batch_videos(self, idx):
        # Fetch a batch of video inputs
        return self.videos[idx]

    def get_batch_audios(self, idx):
        # Fetch a batch of audio inputs
        return self.audios[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_videos = self.get_batch_videos(idx)
        batch_audios = self.get_batch_audios(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_videos, batch_audios, batch_y

######################
#### Entraînement ####
######################

def train_model(model, train_data, val_data, learning_rate, epochs, batch_size):

    #Chargement du dataset

    train_set, val_set = Dataset(train_data, train), Dataset(val_data, dev)

    #Utilisation du GPU

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #CrossEntropyLoss pondérée (pour pallier le déséquilibre du dataset)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    #Optimizer Adam avec LR constant

    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input_text, train_input_video, train_input_audio, train_label in tqdm(train_dataloader):
                #Chargement des labels dans le GPU
                train_label = train_label.to(device)
                #Tokens du texte obtenus par BART
                mask = train_input_text['attention_mask'].to(device)
                input_id = train_input_text['input_ids'].squeeze(1).to(device)

                output = model(train_input_video.to(device), train_input_audio.to(device), input_id, mask.squeeze(1))
                
                batch_loss = criterion(output, train_label.long())
                batch_loss = Variable(batch_loss, requires_grad=True)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input_text, val_input_video, val_input_audio, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input_text['attention_mask'].to(device)
                    input_id = val_input_text['input_ids'].squeeze(1).to(device)

                    output = model(val_input_video.to(device), val_input_audio.to(device), input_id, mask.squeeze(1))

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

train_model(model, train_text, dev_text, LR, EPOCHS, bs)

def evaluate(model, test_data, batch_size):

    test_set = Dataset(test_data, test)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    total_f1 = 0
    with torch.no_grad():

        for test_input_text, test_input_video, test_input_audio, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input_text['attention_mask'].to(device)
              input_id = test_input_text['input_ids'].squeeze(1).to(device)
              test_input_video = torch.Tensor(test_input_video)
              test_input_audio = torch.Tensor(test_input_audio)

              output = model(test_input_video.to(device), test_input_audio.to(device), input_id, mask.squeeze(1))

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              f1 = f1_score(torch.Tensor.cpu(test_label), torch.Tensor.cpu(output.argmax(dim=1)), average='weighted')
              total_f1 += f1
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    print(f'Test F1: {total_f1 / len(test_data): .3f}')
    
evaluate(model, test_text, bs)

#Sauvegarde du modèle entraîné, pour une réutilisation

#Version ruche
torch.save(model.state_dict(), '/usr/users/multimodalemotion/tdujardin/model_fusion_FAM.pth')

#Version locale
#torch.save(model.state_dict(), '/Users/dujardinth/Documents/Python/data_meld/MELD.Raw/model_fusion_FAM.pth')