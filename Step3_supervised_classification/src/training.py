import torch
import pandas as pd
import pickle
import torch.nn as nn
from os import walk
from pathlib import Path
import os
from torch.utils.data import DataLoader
from datasets import Videofeatures_Labels_Dataset
from training_utils import State, weight1, init_weights_xavier
from models import MLP1, MLP2, Lstm, BiLSTM
from evaluation import eval

"""
Train a model and fix optimal nb of epochs.
Save best model (optimize val recall) in folder trained models.
Save parameters and scores in training_results.csv
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Args
path_dict_gloses_videos = "../data/Mediapi/dictionnaire_DJ"
dataset_name = "Mediapi_Expert"
epochs = 10
lr = 1e-4
model_name = "MLP2"
layers_nb = 1
hidden_size = 200
save_models = True
write_results = True
load_model = False
model_nb = 2


dataset_path = os.path.join("Step3_supervised_classification", dataset_name)

os.makedirs(os.path.join(dataset_path, "trained_models"), exist_ok=True)

# Load results
if os.path.exists(os.path.join(dataset_path, "training_results.csv")):
    df_results = pd.read_csv(os.path.join(dataset_path, "training_results.csv"))
else:
    df_results = pd.DataFrame(
        columns=[
            "model",
            "class_nb",
            "lr",
            "layer_nb",
            "hidden_size",
            "epoch",
            "val_acc",
            "val IoU",
            "val P",
            "val R",
            "val_F1",
            "train_acc",
            "train_F1",
            "saved_model",
        ]
    )

# Load utils files
df = pd.read_csv("mediapi_train_clean1.csv")  # load global csv with meta datas
gloses_names = next(walk(path_dict_gloses_videos), (None, None, []))[1]  # load list of gloses names
save_files_path = os.path.join(dataset_path, "saved_files")

d_Vid2Labels = pickle.load(open(os.path.join(save_files_path, "d_Vid2Labels.pkl"), "rb"))
d_Gid2gloses = pickle.load(open(os.path.join(save_files_path, "d_Gid2gloses.pkl"), "rb"))
d_gloses2Gid = pickle.load(open(os.path.join(save_files_path, "d_gloses2Gid.pkl"), "rb"))
d_Gid2Vid = pickle.load(open(os.path.join(save_files_path, "d_Gid2Vid.pkl"), "rb"))
d_Vid2Gid = pickle.load(open(os.path.join(save_files_path, "d_Vid2Gid.pkl"), "rb"))
L_videos = pickle.load(open(os.path.join(save_files_path, "L_videos.pkl"), "rb"))
DTrain = pickle.load(open(os.path.join(save_files_path, "DTrain.pkl"), "rb"))
DVal = pickle.load(open(os.path.join(save_files_path, "DVal.pkl"), "rb"))

# Make Datasets
Train_dataset = Videofeatures_Labels_Dataset(d_Vid2Labels, DTrain)
Val_dataset = Videofeatures_Labels_Dataset(d_Vid2Labels, DVal)
dim_emb = Train_dataset[0][0].shape[1]  # dim video feature (768 for video Swin Transformer)
class_nb = len(d_gloses2Gid)  # number of classes (included 0)
train_dataloader = DataLoader(Train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(Val_dataset, batch_size=1, shuffle=False)

# Models
models = {"MLP1": MLP1, "MLP2": MLP2, "LSTM": Lstm, "biLSTM": BiLSTM}

# Load models
if load_model:
    savepath = Path(os.path.join(dataset_path, "trained_models", f"{model_name}_{model_nb}.pch"))

    with savepath.open("rb") as fp:
        state = torch.load(fp)
    model = state.model
    state.epoch = state.epoch + 1
    optimizer = state.optimizer
    print("Continue the training at epoch", state.epoch, f"during {epochs} epochs")

else:
    model = models[model_name](dim_emb, class_nb, hidden_size, layers_nb)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Initialize optimizer
    state = State(model, optimizer)  # Initialize state
    model.apply(init_weights_xavier)  # Init the neural network weights with Xavier initialization

savepath = Path(os.path.join(dataset_path, "trained_models", f"{model_name}_{len(df_results)}.pch"))

# Define weights in loss function
weight = weight1(Train_dataset, class_nb)

# Define loss function
loss_function = nn.CrossEntropyLoss(torch.tensor(weight).to(device))


######################################Training#########################################
model.to(device)

Best_R, Best_acc = 0, 0

# Run the training loop for defined numbmodel_nber of epochs
for epoch in range(state.epoch, state.epoch + epochs):
    print(f"------Epoch {epoch}------")
    model.train()
    
    # Iterate over the DataLoader for training data
    for inputs, targets, _ in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, val_IoU, val_P, val_R, val_F1 = eval(val_dataloader, device, model, loss_function, class_nb)
    train_loss, train_acc, train_IoU, train_P, train_R, train_F1 = eval(train_dataloader, device, model, loss_function, class_nb)
    print(f"loss -------- train : {round(train_loss/len(Train_dataset),3)}   val : {round(val_loss/len(Val_dataset),3)} ")
    print(f"accuracy  --- train : {round(train_acc,3)}   val : {round(val_acc,3)}")
    print(f"F1  --------- train : {round(train_F1,3)}   val : {round(val_F1,3)}")
    print(f"R  ---------- train : {round(train_R,3)}   val : {round(val_R,3)}")

    if Best_R <= round(val_R, 3):
        Best_acc, Best_IoU, Best_P, Best_R, Best_F1 = val_acc, val_IoU, val_P, val_R, val_F1
        btrain_acc, btrain_IoU, btrain_P, btrain_R, btrain_F1 = (
            train_acc,
            train_IoU,
            train_P,
            train_R,
            train_F1,
        )
        if save_models:
            with savepath.open("wb") as fp:
                state.epoch = epoch
                state.model = model
                state.optimizer = optimizer
                torch.save(state, fp)
                print("--->  save model")
    print("\n")

if write_results:
    df_temp = pd.DataFrame(
        {
            "model": [model_name],
            "class_nb": [class_nb],
            "lr": [lr],
            "layer_nb": [layers_nb],
            "hidden_size": [hidden_size],
            "epoch": [state.epoch],
            "val_acc": [round(Best_acc, 3)],
            "val IoU": [round(Best_IoU, 3)],
            "val P": [round(Best_P, 3)],
            "val R": [round(Best_R, 3)],
            "val_F1": [round(Best_F1, 3)],
            "train_acc": [round(btrain_acc, 3)],
            "train_F1": [round(btrain_F1, 3)],
            "saved_model": [f"{model_name}_{len(df_results)}"],
        }
    )

    if "df_results" in globals():
        df_results = pd.concat([df_results, df_temp])
    else:
        df_results = df_temp

    df_results.to_csv(os.path.join(dataset_path, "training_results.csv"), index=False)
