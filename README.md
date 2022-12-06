# multimodalemotion

Implementation of a model inspired by the "M2FNet" model [https://arxiv.org/abs/2206.02187] on Python using PyTorch.

The training, validation and testing data used come from the MELD dataset [https://affective-meld.github.io/]

The model is trained with a constant LR of 3e-4 on 5 epochs, with a batch size of 32, and on eight labels (neutralneutral, angernegative, disgustnegative, fearnegative, joypositive, sadnessnegative, surprisenegative, surprisepositive).

main_emotions.py, the file to be executed, performs the featurization of the different modalities (~20h of computation time using the computation center of CentraleSupelec which specs can be found here: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/01_cluster_overview/), then the training of the model (~1h of computation). It is enough to modify line 48 of the code, to indicate the correct access path to the folder containing the dataset. This dataset folder must be composed of three subfolders "train", "dev" and "test".
 
The code is organized as follows:

![image](https://user-images.githubusercontent.com/93575161/193032629-aa6ad92b-0dce-40c3-b264-ac32268cd6d4.png)

The "utils" folder contains a function :

- pipeline_annotation_emotions.py, which takes a short video as input (an "utterance" as defined in the M2FNet paper), and returns the same video captioned with the predominant emotion.
