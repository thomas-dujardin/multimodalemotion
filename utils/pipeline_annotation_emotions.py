# Torch
import torch
# Modèle à appliquer 
from models import model_fusion
# Annotation de la vidéo
from moviepy.editor import *
# Suppression erreur
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys, errno
# Extraction features audios
from extract_wav_features import audio_featurizer
# Extraction features texte
from transformers import BartTokenizer
# Extraction features vidéo
from main_AUs import extract_AU_features

################
#### Labels ####
################

labels = ['angernegative', 'disgustnegative', 'fearnegative', 'joypositive', 'neutralneutral', 'sadnessnegative', 'surprisenegative', 'surprisepositive']

####################################################################
#### Chargement modèle, annotation de la vidéo passée en entrée ####
####################################################################

dir = '/Users/dujardinth/Documents/Python/data_meld/train_reste/'
file = 'dia11_utt0.mp4'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

phrase = 'What\'s wrong with you?'

dir = '/Users/dujardinth/Documents/Python/data_meld/train_reste/'
file = 'dia12_utt4.mp4'
cap = VideoFileClip(dir+file)
video_features = extract_AU_features(file, dir, [1/2, 1/2])
zeros = torch.zeros(32768)
video_features = torch.stack((video_features, zeros), dim=0)

zeros2048000 = torch.zeros(2048000)
audio = audio_featurizer(file, dir)
audio = torch.stack((audio, zeros2048000), dim=0)

tokenized = tokenizer(phrase,padding='max_length',max_length=394,truncation=True,return_tensors="pt")

model = model_fusion()
#model.load_state_dict(torch.load('/home/dujardinth/Python/MELD/model_fusion_FAM.pth'), map_location=torch.device('cpu'))
model.load_state_dict(torch.load('/Users/dujardinth/Documents/Python/data_meld/MELD.Raw/model_fusion_FAM.pth', map_location=torch.device('cpu')))

model.eval()

#String contenant le label de la vidéo "video"

output = model(video_features, audio, tokenized['input_ids'], tokenized['attention_mask'])
videolabel = labels[torch.argmax(output[0]).item()]
print(videolabel)
text=TextClip(videolabel, fontsize=40, color="red").set_position(("left","top")).set_duration(cap.duration)

final=CompositeVideoClip([cap, text])

final.write_videofile('/Users/dujardinth/Documents/Python/MOSEI/videoannotee.mp4')












