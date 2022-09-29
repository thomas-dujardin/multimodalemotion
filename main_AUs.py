######################################################
#### Programme principal pour l'extraction d'AUs  ####
######################################################

#Importation des fonctions de hog_AUs.py et de images_split.py
from images_split import split_video
from hog_AUs import csv_writer, acp_features
#Librairie Python permettant la création de dossiers
from pathlib import Path
#Pour ouvrir les .csv
from pandas import read_csv
#Gestion des dossiers
import os

#à partir d'une liste de vidéos, renvoie, pour la vidéo "nom_video", le dossier "nom_video"
#qui contient toutes ses frames (10 par seconde), les .csv contenant les positions des landmarks
#les HOGs, et les features extraites par ACP.

def extract_AU_features(file, dir, learned_weights):
    # Dossier contenant l'utterance numéro i du dialogue
    directory_i = dir+file[:-4]+'/'
    # Création du dossier "directory" contenant les images de la vidéo + les .csv de features
    if not os.path.exists(directory_i):
        Path(directory_i).mkdir(parents=True, exist_ok=True)
        # Séparation de la vidéo en images placées dans le dossier "directory"
        split_video(dir+file, directory_i)
        # Calcul des "pré-features" (features d'apparence par HOG + features géométrique avec les landmark points)
        # ce qui donne les 2 csv "hogs_file.csv" et "new_landmark_files.csv"
        csv_writer(directory_i, read_csv(directory_i+'img_files.csv'), learned_weights)
        # Calcul des features finales par ACP
    return acp_features(directory_i, read_csv(directory_i+'img_files.csv'))