import numpy as np
import pickle
import os
from os import walk
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import shutil


def make_useful_dataDictionnaries(dict_path):
    """
    Using the bilingal lexicon glose-videos (dict_path) to make useful pickle files that are described below.

    Args :
        dict_path (str) : path to the lexicon glose-videos

    Returns :
        d_gloses2Gid (dict) : {glose : gloseId}
        d_Gid2gloses (dict) : {gloseId : glose}
        d_Gid2Vid (dict) : {gloseId : [videos Id that contain this glose]}
        d_Vid2Labels (dict) : {Vid : [0011003333333000000]} annotation of each v_id
        L_videos (list) : list of all annotated videos
        ex_to_look (list): examples with gloses that overlap and require manual inspection.
    """

    df = pd.read_csv("../data/Mediapi/mediapi_train_clean1.csv")  # load global csv with meta datas of Mediapi-rgb
    gloses_names = next(walk(dict_path), (None, None, []))[1]  # load list of gloses names

    # Initialize
    d_gloses2Gid, d_Gid2gloses, d_Gid2Vid, d_Vid2Labels = dict(), dict(), dict(), dict()
    L_videos, ex_to_look = [], []

    i = 0
    for glose in gloses_names:
        L_files = os.listdir(dict_path + glose)  # in L_files : videos +  one csv
        if len(L_files) > 5:  # we're taking gloses that have at least 5 videos instances
            i += 1
            d_gloses2Gid[glose] = i  # define id to each glose
            d_Gid2gloses[i] = glose

            d_Gid2Vid[i] = []  # initialize list of vids that contain this glose

            for f in L_files:
                if "mp4" in f:
                    v_id = "_".join(f.split("_")[:-1])  # entire video
                    d_Gid2Vid[i].append(v_id)
                    v_capture_id = f.split(".")[0]  # video capture of the sign

                    if v_id not in L_videos:
                        L_videos.append(v_id)
                        frames_nb = df[df.video_id == v_id].frames.item()
                        d_Vid2Labels[v_id] = np.zeros(frames_nb - 15)  # initialize labels

                    labels = d_Vid2Labels[v_id]
                    df_glose = pd.read_csv(f"{dict_path}/{glose}/confidence_scores.csv")  # load csv of the glose

                    try:
                        start = df_glose[df_glose.video_ids == v_capture_id].start_frame.item()  # start frame of the sign
                        end = df_glose[df_glose.video_ids == v_capture_id].end_frame.item()  # end frame of the sign

                        # in case of gloses overlapping
                        if labels[start:end].tolist() != np.zeros_like(labels[start:end]).tolist():
                            ex_to_look.append((glose, v_capture_id))  # save v_id to have a look
                            print("conflict between words ", labels[start:end], i)

                        else:
                            labels[start : end + 1] = i  # add annotation
                            d_Vid2Labels[v_id] = labels

                    except:
                        print("problem with", glose, v_id)  # check csv if there is a problem
                        pass

    d_gloses2Gid["neutre"] = 0
    d_Gid2gloses[0] = "neutre"

    return d_gloses2Gid, d_Gid2gloses, d_Gid2Vid, d_Vid2Labels, L_videos, ex_to_look


def Vid2Gids(d_Vid2Labels, v_id):
    """
    Gives the list of gloses of a v_id.

    Args :
        d_Vid2Labels (dict) : {Vid : [0011003333333000000]} annotation of each v_id
        v_id (str) : video Id

    Returns :
        Gids (list) : list of gloses of the v_id
    """
    Gids = np.unique(d_Vid2Labels[v_id]).tolist()
    Gids.remove(0)
    return Gids


def clean_csv(dict_path):
    """
    Clean csv files for each glose of the bilingal lexicon (dict_path) :
    - remove each row of the csv where the video is not present in the directory
    - check all videos are present in the csv

    Args:
        dict_path (str) : path to the lexicon glose-videos
    """
    gloses_names = next(walk(dict_path), (None, None, []))[1]  # list of gloses of the lexicon
    for glose in gloses_names:
        L_files = os.listdir(dict_path + glose)
        L_videos = []

        for f in L_files:
            # just keep the videos
            if ".mp4" in f:
                f = f.split(".")[0]
                L_videos.append(f)

        df = pd.read_csv(open(dict_path + glose + "/confidence_scores.csv", "rb"))
        results = df.copy()
        for i in range(len(df)):
            if df.video_ids.iloc[i] not in L_videos:
                results = results.drop(index=i)

        if len(results) != len(L_videos):
            print("need to check ", glose)
            s = set(L_videos) - set(results.video_ids)
            print(s)
        results.to_csv(dict_path + glose + "/confidence_scores.csv", index=False)


def make_traintest(L_videos, d_Gid2Vid, test_size=0.2, force_video_test=[]):
    """
    Split the data into training and testing datasets and ensure that both datasets contain all classes in the same proportion.

    Args :
        L_videos (list) : list of all videos Id
        d_Gid2Vid (dict) : {gloseId : [videos Id that contain this glose]}
        test_size (float) : size of the test set
        force_video_test (list) : Ids of the videos we want to include in the test set

    Returns :
        DTrain (list)
        DTest (list)
    """
    L = set(L_videos.copy())

    # We first sort the classes by the number of videos in ascending order.
    d = OrderedDict(sorted(d_Gid2Vid.copy().items(), key=lambda x: len(x[1]), reverse=False))
    C = list(d.keys())  # sorted classes by the number of videos

    DTrain = []
    DTest = force_video_test
    L = L - set(DTest)

    for c in C:
        try:
            Lc = list(set(d_Gid2Vid[c]) & set(L))  # videos whose annotation contains class C and
            # which have not yet been assigned to Dtrain or Dtest
            DTrain_c, DTest_c = train_test_split(Lc, test_size=test_size, random_state=42)
            DTrain += DTrain_c
            DTest += DTest_c
            L = L - set(Lc)
        except:
            print("problem with class", c)
    random.shuffle(DTrain)
    random.shuffle(DTest)

    return DTrain, DTest


def data_viz(dataset_path, legend=True):
    """
    Make histogram to visualize train val test datas.

    Args :
        dataset_path (str) : path to the dataset folder
        legend (bool) : if True, add a legend to the plot
    """
    path = os.path.join(dataset_path, "saved_files/")

    # Load datas
    d_Gid2Vid = pickle.load(open(path + "d_Gid2Vid.pkl", "rb"))
    d_Gid2gloses = pickle.load(open(path + "d_Gid2gloses.pkl", "rb"))
    DTrain = pickle.load(open(path + "DTrain.pkl", "rb"))
    DVal = pickle.load(open(path + "DVal.pkl", "rb"))
    DTest = pickle.load(open(path + "DTest.pkl", "rb"))

    # Sort classes by number of videos
    d = OrderedDict(sorted(d_Gid2Vid.copy().items(), key=lambda x: len(x[1]), reverse=True))
    C = list(d.keys())

    # Make train val test dictionnary : {glose :  nb of videos with that glose}
    d_Train = dict()
    d_Test = dict()
    d_Val = dict()

    for c in C:
        Lc = d_Gid2Vid[c]
        S_Train_c = set(Lc) & set(DTrain)
        S_Test_c = set(Lc) & set(DTest)
        S_Val_c = set(Lc) & set(DVal)

        k = d_Gid2gloses[c]
        d_Train[k] = len(S_Train_c)
        d_Test[k] = len(S_Test_c)
        d_Val[k] = len(S_Val_c)

    # Make histogram
    sorted_keys1 = sorted(d_Train, key=lambda x: d_Train[x], reverse=False)
    sorted_values1 = np.array([d_Train[key] for key in sorted_keys1])
    sorted_values2 = np.array([d_Val[key] for key in sorted_keys1])
    sorted_values3 = np.array([d_Test[key] for key in sorted_keys1])

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.barh(sorted_keys1, sorted_values1, color="purple", label="train")
    ax.barh(sorted_keys1, sorted_values2, color="green", label="val", left=sorted_values1)
    ax.barh(sorted_keys1, sorted_values3, color="blue", label="test", left=sorted_values1 + sorted_values2)

    ax.set_ylabel("Classes")
    ax.set_xlabel("Number of videos")
    ax.legend()
    if legend == False:
        plt.gca().set_yticklabels([])
    plt.savefig(dataset_path + "data_viz")


def clean_videos(glose, d_Vid2Gid, d_gloses2Gid, d_Vid2Labels, d_Gid2Vid):
    """
    OPTIONAL FUNCTION
    To use for gloses that have numerous videos instances.
    Select videos that contains at least 2 gloses, delete videos that contain only one glose. Update dictionnaries.

    Args :
        glose (str) : glose that contains numerous videos
        d_Vid2Gid (dict) : {video Id : [gloses Id]}
        d_gloses2Gid (dict) : {gloses Id : gloses Id}
        d_Vid2Labels (dict) : {video Id : [gloses Id]}
        d_Gid2Vid (dict) : {glose Id : [videos Id that contain this glose]}

    Returns :
        d_Vid2Gid (dict)
        d_Vid2Labels (dict)
        d_Gid2Vid (dict)
        L_videos (list)
    """
    cpt = 0
    todelete = 0
    id_glose = d_gloses2Gid[glose]
    d = d_Vid2Gid.copy()
    print(f"{glose} :", id_glose)
    for vid, labels in d.items():
        if id_glose in labels and len(labels) > 1:
            cpt += 1
        elif id_glose in labels and len(labels) == 1:
            todelete += 1
            del d_Vid2Gid[vid]
            del d_Vid2Labels[vid]
    print(f"nb of videos with at least 2 occurences of {glose} : {cpt}")
    print(f"nb of videos with 1 occurence of {glose} : {todelete}")
    cpt = 0
    L_delete = []
    d = d_Gid2Vid.copy()
    for vid in d[id_glose]:
        if vid not in list(d_Vid2Gid.keys()):
            L_delete.append(vid)
            cpt += 1
    for vid in L_delete:
        d_Gid2Vid[id_glose].remove(vid)
    L_videos = list(d_Vid2Labels.keys())

    return d_Vid2Gid, d_Vid2Labels, d_Gid2Vid, L_videos


# For expert dictionary only
def build_expert_dict(
    dict_linguist_path="../data/Mediapi/dictionnaire_plateforme_linguist/",
    dict_path="../data/Mediapi/dictionnaire_expert/",
):
    """
    Create a new folder that contain the expert dictionnary with perfect and nearly perfect videos (code 1 and 2).
    The expert dictionnary is a copy of the base dictionnary containing only the videos with good segmentation.

    Args :
        dict_linguist_path (str) : path to the base bilingual dictionnary (containing modified csv by linguists)
        dict_path (str) : path to the expert dictionnary
    """
    gloses_names = next(walk(dict_linguist_path), (None, None, []))[1]  # list of gloses
    os.makedirs(dict_path, exist_ok=True)  # Make a folder for the expert dictionnary

    for glose in gloses_names:
        print(glose)
        df = pd.read_csv(open(os.path.join(dict_linguist_path, glose, "confidence_scores.csv"), "rb"))
        df_exp = df.loc[df["segm"].isin([1, 2])]  # filter videos with perfect (seg=1) or not too bad segmentation (seg=2)
        print("Number of videos :", len(df_exp))

        df_exp.fillna({"var": 0}, inplace=True)  # replace nan by o in var column
        df_exp["var"] = df_exp["var"].replace(["!", "/", "b"], 0)
        L_variantes = set(df_exp["var"].unique())  # get the variantes
        if len(L_variantes) == 1:  # if only one variant, copy selected videos in the glose folder
            os.makedirs(dict_path + glose, exist_ok=True)
            df_exp.to_csv(dict_path + glose + "/confidence_scores.csv", index=False)
            for v_id in df_exp.video_ids:
                try:
                    shutil.copyfile(dict_linguist_path + glose + "/" + v_id + ".mp4", dict_path + glose + "/" + v_id + ".mp4")

                except:
                    print("pb with ", v_id)

        else:  # if several variants, copy selected videos in the glose_0, glose_1 ... folders
            for v in L_variantes:
                variant_path = dict_path + glose + "_" + str(v)
                os.makedirs(variant_path, exist_ok=True)
                df_exp.to_csv(variant_path + "/confidence_scores.csv", index=False)
            for vid in df_exp.video_ids:
                v = df_exp.loc[df_exp.video_ids == vid, "var"].item()

                try:
                    shutil.copyfile(
                        dict_linguist_path + glose + "/" + vid + ".mp4",
                        dict_path + glose + "_" + str(v) + "/" + vid + ".mp4",
                    )
                except:
                    print("pb with ", vid)

    print("done")
