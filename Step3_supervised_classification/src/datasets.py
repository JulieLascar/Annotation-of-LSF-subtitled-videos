import torch
import numpy as np


class Videofeatures_Labels_Dataset(torch.utils.data.Dataset):
    """Dataset class for training"""

    def __init__(self, d_Vid2Labels, L_videos):
        self.v_ids = L_videos
        self.labels = []
        for v_id in L_videos:
            self.labels.append(d_Vid2Labels[v_id])

    def __getitem__(self, index):
        v_id = self.v_ids[index]
        features_path = "../data/Mediapi/features_mediapi/swin/" + v_id + ".npy"
        x = np.load(features_path)
        y = self.labels[index]
        y = torch.tensor(y).to(dtype=torch.long)
        return x, y, v_id

    def __len__(self):
        return len(self.labels)


class Videofeatures_Dataset(torch.utils.data.Dataset):
    """Dataset class for inferencing"""

    def __init__(self, L_videos):
        self.v_ids = L_videos

    def __getitem__(self, index):
        v_id = self.v_ids[index]
        features_path = "../data/Mediapi/features_mediapi/swin/" + v_id + ".npy"
        x = np.load(features_path)
        return x, v_id

    def __len__(self):
        return len(self.v_ids)
