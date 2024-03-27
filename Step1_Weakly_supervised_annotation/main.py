import sys

sys.path.append("Step1_Weakly_supervised_annotation")

from create_wordvideos import *

# Args
clustering = True
viz_clusters = True
seq_threshold = 3
L_threshold = 0.5
sim_threshold = 0.6
n_max = 100  # nb of videos we take for similarity

path_to_dict = "Step1_Weakly_supervised_annotation/Dictionnaire/"
df_path = "mediapi_train_clean1.csv"
feature_path = "../data/Mediapi/features_mediapi/swin/"
L_word_names = ["7'JIM"]
Lwords = [["7'JIM"]]
L_noword = [[]]
L_word_neg = [[]]

create_word_videos_signer(
    L_word_names,
    Lwords,
    L_noword,
    L_word_neg,
    seq_threshold=seq_threshold,
    sim_threshold=sim_threshold,
    L_threshold=L_threshold,
    path_to_dict=path_to_dict,
    feature_path=feature_path,
    n_max=n_max,
    clustering=clustering,
    viz_clusters=viz_clusters,
    video_path="../data/Mediapi/video_crops_train/",
    df_path="mediapi_train_clean1.csv",
)
