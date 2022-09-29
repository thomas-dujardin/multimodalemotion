# multimodalemotion

Implémentation du modèle "M2FNet" [https://arxiv.org/abs/2206.02187] sur Python avec le framework PyTorch.

Le modèle est entraîné avec un LR constant de 3e-4 sur 5 epochs, avec des batchs de taille 32, et sur huit labels (neutralneutral, angernegative, disgustnegative, fearnegative, joypositive, sadnessnegative, surprisenegative, surprisepositive).

main_emotions.py, le fichier à exécuter, effectue la featurisation des différentes modalités (~20 h de temps de calcul avec le mésocentre des élèves de Centrale), puis l'entraînement du modèle. Il suffit simplement de modifier la ligne 48 du code, pour y indiquer le bon chemin d'accès vers le dossier contenant le dataset. Ce dossier doit être composé de trois sous-dossiers "train", "dev" et "test".
 
Le code est organisé de la manière suivante :

![image](https://user-images.githubusercontent.com/93575161/193032629-aa6ad92b-0dce-40c3-b264-ac32268cd6d4.png)

Le dossier "utils" contient deux fonctions :
- cleaner.py, qui permet de supprimer les phrases en trop dans le .csv (les fichiers .csv contiennent les dialogues de tous les extraits vidéos de MELD, il est donc nécessaire d'en supprimer certaines entrées si l'on veut entraîner le modèle sur une portion du dataset) ;
- pipeline_annotation_emotions.py, qui prend en entrée une vidéo, et qui retourne cette même vidéo captionnée avec l'émotion prédominante.
