"""Make video demos of automatic aanotation"""

import pickle
import cv2
import matplotlib.pyplot as plt
import os
from unidecode import unidecode

# Args
dataset_name = "Mediapi_363"
path = "../data/Mediapi/video_crops_train/"  # path to the continuous videos
model = "MLP2_1"
v_ids = ["c3605137f0_0040", "97008f55fe_0013", "f6cbb0c0ae_0016"]
video_fps = 10

# load files
dico_vId_pred = pickle.load(open(os.path.join("Step3_supervised_classification", dataset_name, f"d_Vid2pred_{model}.pkl"), "rb"))
d_Gid2gloses = pickle.load(
    open(os.path.join("Step3_supervised_classification", dataset_name, "saved_files", "d_Gid2gloses.pkl"), "rb")
)

demo_path = os.path.join("Step3_supervised_classification", dataset_name, "demos")
os.makedirs(demo_path, exist_ok=True)

for v_id in v_ids:
    video_path = path + v_id + ".mp4"
    pred = dico_vId_pred[v_id]
    L = pred[0].tolist()
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    path_out = os.path.join(demo_path, v_id + f"_{model}.mp4")
    video = cv2.VideoWriter(path_out, fourcc, video_fps, (width, height), True)

    for i in range(7):
        ret, frame = cap.read()
        video.write(frame)
    font = cv2.FONT_HERSHEY_TRIPLEX

    for i in range(len(L)):
        ret, frame = cap.read()
        if L[i] == 0:
            """cv2.putText(frame, 
            unidecode(dico_cropId2subtitle[v_id]),
            (50, frame.shape[0] - 50),
            font, 0.3,
            ( 0, 0, 0),
            1,
            cv2.LINE_AA )"""
            video.write(frame)
        else:
            # print(i, L[i])
            cv2.putText(frame, unidecode(d_Gid2gloses[L[i]]), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

            """ cv2.putText(frame, 
                        unidecode(dico_cropId2subtitle[v_id]),
                        (50, frame.shape[0] - 50),
                        font, 0.3,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA )"""

            video.write(frame)
            # if L[i] == 34:
            #   plt.imshow(frame)
            #  plt.show()

    cap.release()
    video.release()
    cv2.destroyAllWindows()

print("end")
