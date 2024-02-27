import pickle
import torch
from utils import inference, analyse_pred

"""
1. Do inference of a model in the test set and save predictions in d_Vid2pred_{model_name}.pkl.
2. For each video in the test set, print the predicted gloses of the video and compare to the ground truth if any. (Optional)
"""

# Args
dataset_name = "Mediapi_NonExpert"
model_name = "MLP2_1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
analyse = False

# Inference on Test set
dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"
DTest = pickle.load(open(dataset_path + "saved_files/DTest.pkl", "rb"))
d_Vid2pred = inference(DTest, dataset_name, model_name, device)
pickle.dump(d_Vid2pred, open(dataset_path + f"d_Vid2pred_{model_name}.pkl", "wb"))  # Save predictions

if analyse:
    # Print results :
    for vid in DTest:
        print(vid)
        analyse_pred(d_Vid2pred, vid, dataset_name)
        print("------------------------------------------------------- \n")
