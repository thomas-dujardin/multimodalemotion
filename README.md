# multimodalemotion

Implémentation du modèle "M2FNet" [https://arxiv.org/abs/2206.02187] sur Python avec le framework PyTorch.

main_emotions.py, le fichier à exécuter, effectue la featurisation des différentes modalités (~20 h de temps de calcul avec le mésocentre des élèves de Centrale), puis l'entraînement du modèle. Il suffit simplement de modifier la ligne 48 du code, pour y indiquer le bon chemin d'accès vers le dossier contenant le dataset.
 
Le code est organisé de la manière suivante :

![image](https://user-images.githubusercontent.com/93575161/193032629-aa6ad92b-0dce-40c3-b264-ac32268cd6d4.png)
