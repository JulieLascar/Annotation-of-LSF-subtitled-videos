import numpy as np
import torch
from itertools import groupby, count
from pathlib import Path
from torch.utils.data import DataLoader
import pickle
from datasets import Videofeatures_Dataset
from models import *


def smoothing(L):
    """
    Smoothing function : given a list of integers, remove the isolated numbers.

    Args :
        L(list): list of integers

    Returns :
        L(list): list of integers

    Example :
    >>> L = [0, 1, 1, 1, 0, 1, 1, 0, 0 ,0, 3, 0, 0, 0, 0] ---> L = [0, 1, 1, 1, 1, 1, 1, 0, 0 ,0, 0, 0, 0, 0, 0]
    """
    co = count()
    for i in range(2, len(L) - 2):
        if L[i] == 0 and L[i + 1] != 0 and L[i - 1] == L[i + 1] and (L[i - 2] == L[i + 1] or L[i + 2] == L[i + 1]):
            L[i] = L[i + 1]
        elif L[i - 1] == 0 and L[i - 2] != L[i] and L[i + 1] == 0 and L[i + 2] != L[i] and L[i] != 0:
            L[i] = 0

    if L[0] != 0 and L[1] == 0 and L[2] != L[0]:
        L[0] = 0
    elif L[1] != 0 and L[0] == 0 and L[2] != L[0]:
        L[1] = 0
    elif L[len(L) - 2] != 0 and L[len(L) - 3] == 0 and L[len(L) - 1] != L[len(L) - 2]:
        L[len(L) - 2] = 0
    elif L[len(L) - 1] != 0 and L[len(L) - 2] == 0 and L[len(L) - 3] != L[len(L) - 1]:
        L[len(L) - 1] = 0

    L = np.array(L)
    C = list(np.unique(L))
    C.remove(0)
    for c in C:
        idx = np.where(L == c)[0]
        if len(idx) != 0:
            L1 = [list(g) for _, g in groupby(idx, lambda x: x - next(co))]
            for L_temp in L1:
                if len(L_temp) < 3:
                    L[L_temp] = 0
    return list(L)


def get_vids_from_words(L_words, L_subtitles, L_vids):
    """
    Select videos Ids which subtitles contain the chosen words.

    Args:
        L_words (list): list of words that should be present in the subtitles
        L_subtitles (list): list of subtitles (associated with videos Ids)
        L_vids (list): list of videos Ids

    Returns:
        L_vids_from_words (list): list of videos Ids which contain the chosen words
    """
    L_vids_from_words = []
    for i in range(len(L_subtitles)):
        vid = L_vids[i]
        cpt = 0
        for w in L_words:
            if w in L_subtitles[i]:
                cpt += 1
            else:
                pass
            if cpt == len(L_words):
                L_vids_from_words.append(vid)
    return L_vids_from_words


def inference(L_videos, dataset_name, model_name, device):
    """
    Given a list of video paths, and a model : do inference of the model

    Args :
        L_videos (list): list of video paths
        dataset_name (string): name of the dataset
        model_name (string): name of the model
        device (string): name of the device

    Return :
        d_Vid2pred (dict) : {Vid (string) : model prediction (Tensor)}
    """
    dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"

    # load model
    savepath = Path(f"{dataset_path}models/{model_name}.pch")
    with savepath.open("rb") as fp:
        state = torch.load(fp)
    model = state.model
    model.to(device)
    model.eval()

    dataset = Videofeatures_Dataset(L_videos)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    try:
        d_Vid2pred = pickle.load(open(dataset_path + f"d_Vid2pred_{model_name}.pkl", "rb"))
    except:
        d_Vid2pred = dict()

    for inputs, v_id in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)

            predicted = torch.argmax(outputs, dim=1).tolist()[0]
            predicted = torch.tensor(smoothing(predicted)).unsqueeze(0).to(device)  # smooth predictions
        d_Vid2pred[v_id[0]] = predicted

    return d_Vid2pred


def analyse_pred(d_Vid2pred, vid, dataset_name, dico=pickle.load(open("dico_cropId2subtitle.pkl", "rb"))):
    """
    For a video which has been infered with inference function:
        - print the predicted gloses of the video
        - print GT gloses if any

    Args:
        d_Vid2pred (dict) : {Vid (string) : model prediction (Tensor)}
        vid (string): video Id
        dataset_name (string): name of the dataset
        dico (dict): {Vid (string) : subtitle (string)}
    """
    dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"

    print(dataset_name)

    d_Vid2Labels = pickle.load(open(dataset_path + "saved_files/d_Vid2Labels.pkl", "rb"))
    d_Gid2gloses = pickle.load(open(dataset_path + "saved_files/d_Gid2gloses.pkl", "rb"))

    id_gloses = torch.unique_consecutive(d_Vid2pred[vid])  # gloses Id predicted in the video

    print(dico[vid])  # print subtitle of the video

    # print predicted gloses in the video
    print("Predictions :")
    for g in id_gloses.tolist():
        if g != 0:
            print(g, d_Gid2gloses[g])
    print("-----------------")

    # print ground truth if any
    print("Ground truth")
    try:
        GT = d_Vid2Labels[vid]
        id_gloses = torch.unique_consecutive(torch.from_numpy(GT))
        for g in id_gloses.tolist():
            if g != 0:
                print(int(g), d_Gid2gloses[g])
    except:
        print("no GT found")
