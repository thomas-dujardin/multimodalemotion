## Modules divers
import math
import os
from cv2 import detail_ExposureCompensator
import pandas as pd
import csv
import numpy as np
from tqdm import tqdm
import torch
## Traitement images
import cv2
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly
from skimage import data, exposure
from skimage.feature import hog
##fait fonctionner urllib (pour télécharger les .pth essentiels au fonctionnement de py-feat)
import os
os.environ['http_proxy']=''
## py-feat
from feat import Detector
## ACP
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

##################################################################
#### Alignement visage (normalisation distance interoculaire) ####
##################################################################

# Cela permet d'appliquer les HOGs

def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """
    Adapted from https://github.com/ZhiwenShao/PyTorch-JAANet by Zhiwen Shao, modified by Tiankang Xie
    img: image
    img_land: landmarks 68
    box_enlarge: relative size of face
    img_size = 112
    """

    leftEye0 = (img_land[2 * 36] + img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] +
                img_land[2 * 41]) / 6.0
    leftEye1 = (img_land[2 * 36 + 1] + img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] +
                img_land[2 * 40 + 1] + img_land[2 * 41 + 1]) / 6.0
    rightEye0 = (img_land[2 * 42] + img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] +
                 img_land[2 * 47]) / 6.0
    rightEye1 = (img_land[2 * 42 + 1] + img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] +
                 img_land[2 * 46 + 1] + img_land[2 * 47 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 30], img_land[2 * 30 + 1], 1],
                   [img_land[2 * 48], img_land[2 * 48 + 1], 1], [img_land[2 * 54], img_land[2 * 54 + 1], 1]])
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1
    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:,0], new_land[:,1]))).astype(int)
    return aligned_img, new_land

#########################
#### Extraction HOGs ####
#########################

#extract_hog : A partir de l'image alignée, récupère les HOGs
##image : image issue de la vidéo (emplacement .jpg)
##learned_weights : vecteur de poids pour faire la "moyenne pondérée" des visages sur l'image (array numpy)
##detector : détecteur de visages (objet py-feat) 

def extract_hog(image, learned_weights, detector):
    im = cv2.imread(image)
    detected_faces = np.array(detector.detect_faces(im)[0])
    im = np.asarray(im)
    detected_faces = detected_faces.astype(int)
    points = (np.array(detector.detect_landmarks(np.array(im), [detected_faces])[0])).astype(int)
    aligned_img, points = align_face_68pts(im, points.flatten(), 2.5)
    if (len(detected_faces) >= 2):
        points = (np.floor((points[:68,:]*learned_weights[0] + points[68:136,:]*learned_weights[1])/(np.sum(learned_weights)))).astype(int)
        hull = ConvexHull(points)
        mask = grid_points_in_poly(shape=np.array(aligned_img).shape, 
                                verts= list(zip(points[hull.vertices][:,1], points[hull.vertices][:,0])) # for some reason verts need to be flipped
                                )
        mask[0:np.min([points[0][1], points[16][1]]), points[0][0]:points[16][0]] = True
        aligned_img[~mask] = 0
        resized_face_np = aligned_img

        fd, hog_image = hog(resized_face_np, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
        return fd, points.flatten(), resized_face_np, hog_image
    
    if (len(detected_faces) == 1):
        hull = ConvexHull(points)
        mask = grid_points_in_poly(shape=np.array(aligned_img).shape, 
                                verts= list(zip(points[hull.vertices][:,1], points[hull.vertices][:,0])) # for some reason verts need to be flipped
                                )
        mask[0:np.min([points[0][1], points[16][1]]), points[0][0]:points[16][0]] = True
        aligned_img[~mask] = 0
        resized_face_np = aligned_img

        fd, hog_image = hog(resized_face_np, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

        return fd, points.flatten(), resized_face_np, hog_image

####################################
#### Initialisation d'un modèle ####
####################################

A01 = Detector(face_model='RetinaFace',emotion_model="resmasknet", landmark_model="mobilefacenet", au_model="svm", facepose_model="img2pose") #initialize model

# Make sure that in the master file you have a column (called original_path), which points to the path of the images

##############################
#### Ecriture dans le CSV ####
##############################

#csv_writer : crée deux .csv hogs_file et new_landmarks_files qui contiennent respectivement
#le vecteur HOG pour chaque image de la vidéo et la position des Facial Landmark Points en coord. (x, y)
##write_path : Emplacement dans lequel mettre ces deux .csv
##master_file : Fichier csv dont les entrées pointent vers l'emplacement de chacune des images

import traceback
import sys

def csv_writer(write_path, master_file, learned_weights):

    #write_path = "/Users/dujardinth/Anaconda3/envs/mosei/Lib/site-packages/feat/tests/data/hog/" # This is the path where you store all hog and landmark files
    #master_file = pd.read_csv("/Users/dujardinth/Anaconda3/envs/mosei/Lib/site-packages/feat/tests/data/test_disfa.csv") # Fichier .csv contenant toutes les images à traiter
    master_file["Marked"] = True # This mark column serves to check whether each image can be correctly processed by the algorithm.

    with open(write_path+'hogs_file.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(range(5408)))

    with open(write_path+'new_landmarks_files.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(range(136)))

    for ix in tqdm(range(master_file.shape[0])):
        try:
            imageURL = master_file['original_path'][ix]
            fd, landpts, cropped_img, hogs = extract_hog(imageURL, learned_weights, detector=A01)
            with open(write_path+'hogs_file.csv', "a+", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(fd)
            
            with open(write_path+'new_landmarks_files.csv', "a+", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(landpts.flatten())

        except Exception as e:
            master_file['Marked'][ix] = False
            #print("failed to load",imageURL)
            print(traceback.format_exc())
            continue;
    master_file.to_csv(write_path+"/HOG_master_file.csv")

################################################
#### ACP sur les features obtenues par HOGs ####
################################################

#acp_features : crée un .csv pca_AU_features qui contient, pour chaque frame de la vidéo splitée,
#825 features qui sont la concaténation des HOGs réduits par ACP (à travers toutes les images) et des
#Facial Landmark Points.
##write_path : Emplacement dans lequel mettre ce .csv
##master_file : Fichier csv dont les entrées pointent vers l'emplacement de chacune des images
def acp_features(write_path, master_file):
    hogs_vals = pd.read_csv(write_path+"hogs_file.csv")
    landmarks_ts = pd.read_csv(write_path+"new_landmarks_files.csv")
    if (len(hogs_vals)>0):
        scaler = StandardScaler()
        hogs_cbd_std = scaler.fit_transform(hogs_vals)
        pca = PCA(n_components=0.95, svd_solver = 'full')
        hogs_transformed = pca.fit_transform(hogs_cbd_std)
        x_features = np.concatenate((hogs_transformed,landmarks_ts),1).flatten()
    else:
        x_features = np.concatenate((hogs_vals, landmarks_ts), 1).flatten()
    x_features = np.pad(x_features, [(0, 32768-len(x_features))])
    try:
        return torch.Tensor(x_features)
    except:
        pass

    """with open(write_path+'pca_AU_features.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(list(range(825)))

    for ix in tqdm(range(master_file.shape[0])):
        try:
            with open(write_path+'pca_AU_features.csv', "a+", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(x_features[ix])
        except:
            master_file['Marked'][ix] = False
            print("failed to load",x_features[ix])
            continue;"""
    #out = pd.read_csv(write_path+'pca_AU_features.csv')
    #out = out.dropna(axis=1)
    #return out