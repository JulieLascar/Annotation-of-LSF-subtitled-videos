"""
Prepare expert and non expert datas for training.
--> Do the steps one after the other.

0. build expert dictionary from linguists annotation. (only for expert data)
1. make useful dictionaries, and create data annotation with expert dictionary.
2. correct annotations (ex to look).
3. make train val test datas.
4. Visualize datas.

"""
from data_utils import *
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=(SettingWithCopyWarning))

#### Args #####
expert = False  # True if expert, False otherwise.

if expert:
    dict_linguist_path = "../data/Mediapi/dictionnaire_plateforme_linguist/"  # path to the dictionary annotated by linguists
    step0 = False  # build expert dictionary from linguists annotation.

else:
    dict_path = "../data/Mediapi/dictionnaire_NonExpert/"  # path to he bilingual dictionary

# for expert and non expert data :
step1 = True  # make useful files, and create data annotation with the bilingual dictionary.
step2 = False  # correct annotations (ex to look) --> go to correct_annotation.ipynb
step3 = True  # make train val test datas.
step4 = True  # visualize datas.

legend = True  # for visualization ; write classes names in y axis of the histogram plot.

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
if expert:
    dataset_name = "Mediapi_Expert"
    dict_path = "../data/Mediapi/dictionnaire_expert/"  # path to dictionary

else:
    dataset_name = "Mediapi_NonExpert"

dataset_path = "Step3_supervised_annotation/" + dataset_name + "/"

# 0. build expert dictionary from expert annotation.
if expert and step0:
    build_expert_dict(dict_linguist_path, dict_path)

# 1. make useful files, and create data annotation with the bilingual dictionary.
if step1:
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(dataset_path + "saved_files/", exist_ok=True)

    # Make & save d_gloses2Gid, d_Gid2gloses, d_Gid2Vid, d_Vid2Labels, L_videos, ex_to_look
    (
        d_gloses2Gid,
        d_Gid2gloses,
        d_Gid2Vid,
        d_Vid2Labels,
        L_videos,
        ex_to_look,
    ) = make_useful_dataDictionnaries(dict_path=dict_path)

    pickle.dump(d_gloses2Gid, open(dataset_path + "saved_files/d_gloses2Gid.pkl", "wb"))
    pickle.dump(d_Gid2gloses, open(dataset_path + "saved_files/d_Gid2gloses.pkl", "wb"))
    pickle.dump(d_Gid2Vid, open(dataset_path + "saved_files/d_Gid2Vid.pkl", "wb"))
    pickle.dump(d_Vid2Labels, open(dataset_path + "saved_files/d_Vid2Labels.pkl", "wb"))
    pickle.dump(L_videos, open(dataset_path + "saved_files/L_videos.pkl", "wb"))
    pickle.dump(ex_to_look, open(dataset_path + "saved_files/ex_to_look.pkl", "wb"))

    # Make  & save d_Vid2Gid
    d_Vid2Gid = dict()  # dictionnary {Vid : [1,3]} list of annotated gloses of each v_id
    for v_id in L_videos:
        d_Vid2Gid[v_id] = Vid2Gids(d_Vid2Labels, v_id)
    pickle.dump(d_Vid2Gid, open(dataset_path + "saved_files/d_Vid2Gid.pkl", "wb"))

# 2. correct annotations (ex to look).
# --> go to correct_annotation.ipynb

# 3. make train val test.
if step3:
    clean_csv(dict_path)

    d_Gid2Vid = pickle.load(open(dataset_path + "saved_files/d_Gid2Vid.pkl", "rb"))
    L_videos = pickle.load(open(dataset_path + "saved_files/L_videos.pkl", "rb"))

    if expert:
        for vid in force_video_test.copy():
            if vid not in L_videos:
                force_video_test.remove(vid)
        DTrain = list(set(L_videos) - set(force_video_test))
        DTrain, DTest = make_traintest(DTrain.copy(), d_Gid2Vid, test_size=0.12, force_video_test=force_video_test)
        DTrain, DVal = make_traintest(DTrain.copy(), d_Gid2Vid, test_size=0.15, force_video_test=[])

    else:
        # We take the same DTest and DVal as in Mediapi_Expert
        DTest = pickle.load(open("Step3_supervised_annotation/Mediapi_Expert/saved_files/DTest.pkl", "rb"))
        DVal = pickle.load(open("Step3_supervised_annotation/Mediapi_Expert/saved_files/DVal.pkl", "rb"))

        DTrain = list(set(L_videos) - set(DVal))
        DTrain = list(set(DTrain) - set(DTest))

    pickle.dump(DTrain, open(dataset_path + "saved_files/DTrain.pkl", "wb"))
    pickle.dump(DTest, open(dataset_path + "saved_files/DTest.pkl", "wb"))
    pickle.dump(DVal, open(dataset_path + "saved_files/DVal.pkl", "wb"))

# 4. Visualize datas.
if step4:
    data_viz(dataset_path, legend=legend)
