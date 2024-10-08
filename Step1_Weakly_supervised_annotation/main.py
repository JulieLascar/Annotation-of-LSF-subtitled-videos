import sys

sys.path.append("Step1_Weakly_supervised_annotation")

from create_wordvideos import *

# Args
clustering = True
viz_clusters = True
seq_threshold = 3  # We will mmake a clip when we get at leat seq_threshold frames above L_threshold.
L_threshold = 0.5
sim_threshold = 0.5
n_max = 100  # nb of videos we take for similarity
video_path = "../data/Mediapi/video_crops_train/"  # path to the continuous videos
path_to_dict = "Step1_Weakly_supervised_annotation/Dictionnaire/"  # path to the bilingual lexixon
df_path = "mediapi_train_clean1.csv"  # csv that contains videos ids with signer ids
feature_path = "../data/Mediapi/features_mediapi/swin/"  # path to the video features

L_word_names = ["800"]  # list of labels (name of the folder)
Lwords = [
    [" 800 ", " huit cent"],
]  # we will select videos which subtitle contain at least one of the words of Lwords
L_noword = [[]]  # we will not select videos which subtitles contain one of the words of L nowords
L_word_neg = [["million", " mille "]]  # for negative examples and computing L-
method = "signer"  # choice to sort videos ('signer' or 'base' or 'semantic')

create_word = {"base": create_word_videos_base, "signer": create_word_videos_signer, "semantic": create_word_videos_semantic}
create_word[method](
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
    video_path=video_path,
    df_path=df_path,
)
