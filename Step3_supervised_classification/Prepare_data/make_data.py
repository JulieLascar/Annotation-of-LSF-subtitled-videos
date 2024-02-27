"""
Prepare datas for training.
--> Do the steps one after the other.

1. make useful dictionaries, and create data annotation with dictionary.
2. correct annotations (ex to look).
3. make train val test datas.
4. Visualize datas.
"""

from data_utils import *

#### Args #####
dataset_name = "Mediapi_363"  # name of the folder

dict_path = "../data/Mediapi/dictionnaire_DJ/"  # path to the bilingual dictionary

step1 = True  # make useful files, and create data annotation with the bilingual dictionary.
step2 = False  # correct annotations (ex to look) --> go to correct_annotation.ipynb
step3 = False  # make train val test datas.
step4 = False  # Visualize datas.

legend = False  # for visualization ; write classes names in y axis of the histogram plot.

force_video_test = [
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
]  # some selected videos ids we want in the test set.


##### Prepare the data #####
dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"
save_path = dataset_path + "saved_files/"

# 1. make useful files, and create data annotation with the bilingual dictionary
if step1:
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Make & save d_gloses2Gid, d_Gid2gloses, d_Gid2Vid, d_Vid2Labels, L_videos, ex_to_look
    (
        d_gloses2Gid,
        d_Gid2gloses,
        d_Gid2Vid,
        d_Vid2Labels,
        L_videos,
        ex_to_look,
    ) = make_useful_dataDictionnaries(dict_path=dict_path)

    pickle.dump(d_gloses2Gid, open(save_path + "d_gloses2Gid.pkl", "wb"))
    pickle.dump(d_Gid2gloses, open(save_path + "d_Gid2gloses.pkl", "wb"))
    pickle.dump(d_Gid2Vid, open(save_path + "d_Gid2Vid.pkl", "wb"))
    pickle.dump(d_Vid2Labels, open(save_path + "d_Vid2Labels.pkl", "wb"))
    pickle.dump(L_videos, open(save_path + "L_videos.pkl", "wb"))
    pickle.dump(ex_to_look, open(save_path + "ex_to_look.pkl", "wb"))

    # Make  & save d_Vid2Gid
    d_Vid2Gid = dict()  # dictionnary {Vid : [1,3]} list of annotated gloses of each v_id
    for v_id in L_videos:
        d_Vid2Gid[v_id] = Vid2Gids(d_Vid2Labels, v_id)
    pickle.dump(d_Vid2Gid, open(save_path + "d_Vid2Gid.pkl", "wb"))

# 2. correct annotations (ex to look)
# --> go to correct_annotation.ipynb

# 3. Make & save DTrain, DVal, DTest
if step3:
    clean_csv(dict_path)

    d_Gid2Vid = pickle.load(open(save_path + "d_Gid2Vid.pkl", "rb"))
    L_videos = pickle.load(open(save_path + "L_videos.pkl", "rb"))

    for vid in force_video_test.copy():
        if vid not in L_videos:
            force_video_test.remove(vid)
    DTrain = list(set(L_videos) - set(force_video_test))
    DTrain, DTest = make_traintest(DTrain.copy(), d_Gid2Vid, test_size=0.12, force_video_test=force_video_test)
    DTrain, DVal = make_traintest(DTrain.copy(), d_Gid2Vid, test_size=0.15, force_video_test=[])

    pickle.dump(DTrain, open(save_path + "DTrain.pkl", "wb"))
    pickle.dump(DTest, open(save_path + "DTest.pkl", "wb"))
    pickle.dump(DVal, open(save_path + "DVal.pkl", "wb"))

# 4. Visualize datas
if step4:
    data_viz(dataset_path, legend=legend)
