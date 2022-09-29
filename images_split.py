from math import ceil
##Video Clips Utils
from moviepy.editor import VideoFileClip
##Utils
import numpy as np
import pandas as pd
import os

# On ne garde que 10 images par seconde
SAVING_FRAMES_PER_SECOND = 5

#split_video : découpe la vidéo en images qui seront analysées séparément.
#Les images sont sauvegardées dans un dossier qui porte le nom de la vidéo découpée
##video_file : emplacement du fichier vidéo à découper
##writepath : emplacement où sont stockées les dossiers, images et csv afférents à l'extraction d'AUs.
def split_video(video_file, writepath):
    # load the video clip
    video_clip = VideoFileClip(video_file)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    i = 0
    for current_duration in np.arange(0, video_clip.duration, step):
        # format the file name and save it
        frame_filename = writepath+'frame'+str(i)+'.jpg'
        # save the frame with the current duration
        video_clip.save_frame(frame_filename, current_duration)
        i = i + 1
    filenames_list = [filenames(j, writepath) for j in range(i)]
    sujet001 = pd.DataFrame({'original_path':filenames_list})
    sujet001.to_csv(writepath+'/img_files.csv')

def filenames(i, writepath_img):
    return writepath_img+'frame'+str(i)+'.jpg'

#def filenames(i):
    #return '/Users/dujardinth/Anaconda3/envs/mosei/Lib/site-packages/feat/tests/data/RightVideoSN001_Comp-moviepy/frame'+str(i)+'.jpg'

#if __name__ == "__main__":
    #video_file = "/Users/dujardinth/Anaconda3/envs/mosei/Lib/site-packages/feat/tests/data/RightVideoSN001_Comp.avi"
    #split_video(video_file)