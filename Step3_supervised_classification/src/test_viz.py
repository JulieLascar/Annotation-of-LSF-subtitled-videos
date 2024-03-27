import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from utils import get_vids_from_words, inference

""" 
Visualization of videos predictions in the test set or specific videos which subtitles contain specific words.
"""


def viz_preds(vid, model_name, dataset_name, gt=True):
    """do visualisazion of predictions for a video with ot without ground truth.

    Args:
        vid (str)
        model_name (str)
        dataset_name (str)
        gt (bool) : if ground truth
    """
    os.makedirs("Step3_supervised_annotation/Visualisations", exist_ok=True)
    os.makedirs(f"Step3_supervised_annotation/Visualisations/viz_{dataset_name}_{model_name}", exist_ok=True)
    dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"

    dico = pickle.load(open("dico_cropId2subtitle.pkl", "rb"))
    labels = pickle.load(open(f"{dataset_path}saved_files/d_Gid2gloses.pkl", "rb"))
    prediction = pickle.load(open(f"{dataset_path}d_Vid2pred_{model_name}.pkl", "rb"))[vid].cpu().numpy().reshape(-1)
    _, ax = plt.subplots(figsize=(15, 4))
    band_width = 0.2

    def add_annotation(ax, data, y, labels):
        current_color = None
        for i, value in enumerate(data):
            if value != 0:
                color = labels[value]
                if color != current_color:
                    ax.annotate(color, xy=(i + 2, y), ha="center", va="center", color="black", fontsize=8, rotation=90)
                    current_color = color
            else:
                current_color = None

    ax.imshow(
        prediction.reshape(1, -1),
        cmap="viridis",
        aspect="auto",
        extent=[0, len(prediction), 0.8 - band_width / 2, 0.8 + band_width / 2],
        alpha=0.7,
    )
    add_annotation(ax, prediction, 0.8, labels)
    ax.set_yticks([0.7, 0.8])
    ax.set_xlim(0, len(prediction))
    plt.gca().set_yticklabels([])

    d_Vid2Labels = pickle.load(open(f"{dataset_path}saved_files/d_Vid2Labels.pkl", "rb"))
    if gt and vid in list(d_Vid2Labels.keys()):
        gt_annotation = d_Vid2Labels[vid]

        ax.imshow(
            gt_annotation.reshape(1, -1),
            cmap="viridis",
            aspect="auto",
            extent=[0, len(gt_annotation), 0.5 - band_width / 2, 0.5 + band_width / 2],
            alpha=0.7,
        )
        add_annotation(ax, gt_annotation, 0.5, labels)

        ax.set_yticks([0.4, 0.5, 0.8, 0.9])
        ax.set_xlim(0, len(prediction))
        ax.set_yticklabels(["", "ground truth", "prediction", ""])
    plt.title(dico[vid])
    plt.savefig(f"Step3_supervised_annotation/Visualisations/viz_{dataset_name}_{model_name}/{vid}.png")


if __name__ == "__main__":
    model_name = "MLP2_0"
    dataset_name = "Mediapi_Expert"
    L_videos = [
        "997907db82_0014",
        "0a7338124f_0018",
        "a1c9365f5d_0025",
        "f64715269a_0000",
        "a003934c85_0069",
        "689abe06d3_0002",
        "f6cbb0c0ae_0016",
        "2a68bb548d_0003",
        "97008f55fe_0013",
        "c3605137f0_0040",
        "6eb626c240_0014",
        "4433edab36_0003",
        "1c624d4dcb_0040",
        "d830f826dc_0029",
        "6eb626c240_0014",
        "bb2d91ce2e_0058",
        "fa95139292_0030",
    ]
    L_words = []  # in case we want to visualize videos that contains specific words

    dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"
    if L_words == [] and L_videos == []:
        L_videos = pickle.load(open(f"{dataset_path}/saved_files/DTest.pkl", "rb"))

    elif L_words != []:
        df = pd.read_csv("mediapi_train_clean1.csv")
        dico_cropId2subtitle = pickle.load(open("dico_cropId2subtitle.pkl", "rb"))
        train_crop_video_id = df["video_id"].values.tolist()
        L_train_subtitles = []

        for crop_id in train_crop_video_id:
            subtitle = dico_cropId2subtitle[crop_id]
            L_train_subtitles.append(subtitle)

        L_videos = get_vids_from_words(L_words, L_subtitles=L_train_subtitles, L_vids=train_crop_video_id)
        print(L_videos)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d_Vid2pred = inference(L_videos, dataset_name, model_name, device)
        pickle.dump(d_Vid2pred, open(dataset_path + f"d_Vid2pred_{model_name}.pkl", "wb"))

    for vid in L_videos:
        try:
            viz_preds(vid, model_name, dataset_name, gt=True)
            print("ok")
        except:
            print(f"Erreur vid : {vid}")
