import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def viz_2preds(vid, model_name1, model_name2, dataset_name1, dataset_name2, gt=False):
    """
    do visualisazion of 2 predictions for a video with or without gt.
    """
    os.makedirs("Step3_supervised_classification/Visualisations", exist_ok=True)
    os.makedirs(f"Step3_supervised_classification/Visualisations/viz_{dataset_name1}_{dataset_name2}", exist_ok=True)
    dataset_path1 = "Step3_supervised_classification/" + dataset_name1 + "/"
    dataset_path2 = "Step3_supervised_classification/" + dataset_name2 + "/"

    # Load predictions (need to do inference before)
    pred1 = pickle.load(open(f"{dataset_path1}d_Vid2pred_{model_name1}.pkl", "rb"))[vid].cpu().numpy().reshape(-1)
    pred2 = pickle.load(open(f"{dataset_path2}d_Vid2pred_{model_name2}.pkl", "rb"))[vid].cpu().numpy().reshape(-1)
    if gt:
        GT = pickle.load(open(f"Step3_supervised_classification/Visualisations/{vid}_gt.pkl", "rb"))

    dico = pickle.load(open("dico_cropId2subtitle.pkl", "rb"))

    labels1 = pickle.load(open(f"{dataset_path1}saved_files/d_Gid2gloses.pkl", "rb"))
    labels2 = pickle.load(open(f"{dataset_path2}saved_files/d_Gid2gloses.pkl", "rb"))
    _, ax = plt.subplots(figsize=(15, 5))

    band_width = 0.2

    def add_annotation(ax, data, y, labels):
        current_glose = None
        for i, value in enumerate(data):
            if value != 0:
                glose = labels[value]
                if glose != current_glose:
                    ax.annotate(
                        glose,
                        xy=(i + 3, y),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=11,
                        rotation=90,
                    )
                    current_glose = glose
            else:
                current_glose = None

    mask = pred2.copy()
    for i, c in enumerate(mask):
        if c != 0 and (mask[i - 1] != c or mask[i + 1] != c) and mask[i - 1] != 800:
            mask[i] = 800
    mask = [900 if l not in [0, 800] else l for l in mask]
    mask = np.array(mask)
    ax.imshow(
        mask.reshape(1, -1),
        cmap="tab20c",
        aspect="auto",
        extent=[0, len(pred2), 0.8 - band_width / 2, 0.8 + band_width / 2],
        alpha=0.7,
    )
    print(pred2)
    add_annotation(ax, pred2, 0.8, labels2)

    if gt:
        mask = GT.copy()
        for i, c in enumerate(mask):
            if c != 0 and (mask[i - 1] != c or mask[i + 1] != c) and mask[i - 1] != 800:
                mask[i] = 800
        mask = [900 if l not in [0, 800] else l for l in mask]
        mask = np.array(mask)
        print(mask)
        ax.imshow(
            mask.reshape(1, -1),
            cmap="tab20c",
            aspect="auto",
            extent=[0, len(GT), 0.5 - band_width / 2, 0.5 + band_width / 2],
            alpha=0.7,
        )
        print(GT)
        add_annotation(ax, GT, 0.5, labels2)

    mask = pred1.copy()
    for i, c in enumerate(mask):
        if c != 0 and (mask[i - 1] != c or mask[i + 1] != c) and mask[i - 1] != 800:
            mask[i] = 800
    mask = [900 if l not in [0, 800] else l for l in mask]
    mask = np.array(mask)
    ax.imshow(
        mask.reshape(1, -1),
        cmap="tab20c",
        aspect="auto",
        extent=[0, len(pred1), 1.1 - band_width / 2, 1.1 + band_width / 2],
        alpha=0.7,
    )
    add_annotation(ax, pred1, 1.1, labels1)
    if gt:
        ax.set_yticks(
            [0.4, 0.5, 0.8, 0.85, 1.1, 1.15, 1.2],
        )
        ax.set_yticklabels(["", "Ground truth", "expertise", " With ", "expertise", "Without", ""])
    else:
        ax.set_yticks(
            [0.7, 0.8, 0.85, 1.1, 1.15, 1.2],
        )
        # ax.set_yticklabels(["", "expertise", " With ", "expertise", "Without", ""])
        ax.set_yticklabels(["", "363 classes", " With ", "44 classes", "With", ""])
    print("ok2")
    plt.title(f"{dico[vid]}\n", loc="center", fontsize=12)
    plt.savefig(f"Step3_supervised_classification/Visualisations/viz_{dataset_name1}_{dataset_name2}/{vid}.png")


if __name__ == "__main__":
    model_name1 = "MLP2_1"
    model_name2 = "MLP2_1"

    dataset_name1 = "Mediapi_Expert"
    dataset_name2 = "Mediapi_363"

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

    for vid in L_videos:
        try:
            viz_2preds(vid, model_name1, model_name2, dataset_name1, dataset_name2, gt=False)
            print("ok")
        except:
            print(f"Erreur vid : {vid}")
