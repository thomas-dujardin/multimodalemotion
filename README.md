# multimodalemotion

An attempt at implementing a model inspired by "M2FNet" [https://arxiv.org/abs/2206.02187] using PyTorch.

A detailed report (in French) and summaries in French and English can be found in the "reports_summaries" folder.

The training, validation and testing data come from the MELD dataset [https://affective-meld.github.io/]

The model is trained with a constant LR of 3e-4 on 5 epochs, with a batch size of 32, and on eight labels (neutralneutral, angernegative, disgustnegative, fearnegative, joypositive, sadnessnegative, surprisenegative, surprisepositive).
 
The code is organized as follows:

![image](https://user-images.githubusercontent.com/93575161/193032629-aa6ad92b-0dce-40c3-b264-ac32268cd6d4.png)

- main_AUs extracts the Action Units (https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/), by splitting the videos in images using images_split.py, and by detecting the Action Units on them using Baltrusaitis's HOGs algorithm (https://www.researchgate.net/publication/308864223_Cross-dataset_learning_and_person-specific_normalisation_for_automatic_Action_Unit_detection) in hog_AUs.py;

- extract_wav_features.py uses Facebook's Wav2vec2.0 (https://arxiv.org/abs/2006.11477) in order to extract features from the video's audio;

- models.py contains M2FNet's architecture (with slight modifications), and extracts features from the text transcription of the video using HuggingFace's pre-trained BART checkpoint ;

- main_emotions.py, the file to be executed, performs the featurization of the different modalities using the aforementioned .py files (~20h of computation time using the computation center of CentraleSupelec which specs can be found here: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/01_cluster_overview/), and the training of the model (~1h of computation). Line 48 must be modified, in order to indicate the correct access path to the folder containing the dataset. This folder must be composed of three subfolders "train", "dev" and "test".

The "utils" folder contains a function :

- pipeline_annotation_emotions.py, which takes a short video as input (an "utterance" as defined in the M2FNet paper), and returns the same video captioned with the predominant emotion.

Due to the short duration of the internship, and due to the fact that I am still learning PyTorch, the performances are not great (~35% accuracy). This might be explained by some approximations and errors in my implementation.
