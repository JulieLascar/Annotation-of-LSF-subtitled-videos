from pathlib import Path
import torch
import pickle
from torch.utils.data import DataLoader
from datasets import Videofeatures_Labels_Dataset
import os

"""
Print scores for the test dataset.
"""

# Args :
dataset_name = "Mediapi_Expert"
model_name = "MLP2_2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Computes scores on test set :
dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"

savepath = Path(os.path.join(dataset_name, "models", model_name + ".pch"))
with savepath.open("rb") as fp:
    state = torch.load(fp)
model = state.model
model.to(device)
model.eval()

save_files_path = os.path.join(dataset_path, "saved_files")
d_Vid2Labels = pickle.load(open(os.path.join(save_files_path, "d_Vid2Labels.pkl"), "rb"))
d_Gid2gloses = pickle.load(open(os.path.join(save_files_path, "d_Gid2gloses.pkl"), "rb"))

DTest = pickle.load(open(os.path.join(save_files_path, "DTest.pkl"), "rb"))
Test_dataset = Videofeatures_Labels_Dataset(d_Vid2Labels, DTest)
Test_dataloader = DataLoader(Test_dataset, batch_size=1, shuffle=False)

class_nb = len(d_Gid2gloses)

_, acc, IoU, P, R, F1 = eval(Test_dataloader, device, model, None, class_nb)

print(f"acc :{round(acc,3)}  R :{round(R,3)} F1 :{round(F1,3)}")
