import os
import sys

sys.path.append("Step1_Weakly_supervised_annotation")

from utils import *
from similarity import *
import pandas as pd
import pickle
import warnings
from os import walk
import shutil
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

warnings.simplefilter(action="ignore", category=FutureWarning)


def create_word_videos_base(
    L_word_names,
    Lwords,
    L_noword,
    L_word_neg,
    seq_threshold=3,
    sim_threshold=0.6,
    L_threshold=0.5,
    path_to_dict="Step1_Weakly_supervised_annotation/Dictionnaire5/",
    feature_path="../data/Mediapi/features_mediapi/swin/",
    n_max=100,
    clustering=True,
    viz_clusters=True,
    video_path="../data/Mediapi/video_crops_train/",
    df_path="mediapi_train_clean1.csv",
):
    """Create a dictionnary with folders for each sign.
    Each folder contains :
        - captured videos of the sign (possibly sort in Kmeans folders if clustering = True)
        - confidence-scores.csv : informations about the captured signs (video id, signer id, nb of frames, start frame, end frame, mid frame )
        - d_Lplus : dictionnary {vid : L+}
        - d_Lmoins : dictionnary {vid : L-}

    Args:
        L_word_names (list): Names of words (to make a folder)
        Lwords (list): we will select videos which subtitle contain at least one of the words of Lwords
        L_noword (list): we will not select videos which subtitles contain one of the words of L nowords
        L_word_neg (list): for L- computing (see paper)
        seq_threshold (int, optional): minimal sequence lenght . Defaults to 3.
        sim_threshold (float, optional): similarity threshold. Defaults to 0.6.
        L_threshold (float, optional): L score threshold. Defaults to 0.5.
        path_to_dict (str, optional): path to the dictionnary folder (to save signs). Defaults to "Step1_Weakly_supervised_annotation/Dictionnaire5/".
        feature_path (str, optional): path to the video feaures. Defaults to "../data/Mediapi/features_mediapi/swin/".
        n_max (int, optional): maximal number of positive examples. Defaults to 100.
        clustering (bool, optional): if we want to use clustering. Defaults to True.
        viz_clusters (bool, optional): if we want to visualize the datas. Defaults to True.
        video_path (str, optional): path to the continous videos. Defaults to "../data/Mediapi/video_crops_train/".
        df_path (str, optional): path to the metadatas csv. Defaults to "mediapi_train_clean1.csv".
    """
    os.makedirs(path_to_dict, exist_ok=True)  # create Dictionnary folder

    dico_cropId2subtitle = pickle.load(open("dico_cropId2subtitle.pkl", "rb"))  # load {vid : subtitle} dictionnary

    df_train = pd.read_csv(df_path)  # load metadata csv from Mediapi-RGB

    for i, word in enumerate(L_word_names):
        path_word = path_to_dict + word
        os.makedirs(path_word, exist_ok=True)  # create word folder to save captured signs

        sim = Similarity(
            L_video_id=list(df_train.video_id),
            word=word,
            L_word=Lwords[i],
            dico=dico_cropId2subtitle,
            feature_path=feature_path,
            pr=True,
            L_nowords=L_noword[i],
            n_max=n_max,
            L_word_neg=L_word_neg[i],
        )
        if sim.Video_nb > 1:
            d_L, d_plus, d_moins = sim.collect_L(sim_threshold=sim_threshold, Negative_ex=True)  # list of scores vectors
            L_positive_videos_id = sim.L_positive_video_id  # list of selected videos ids in which we look for the sign

            for v_id in L_positive_videos_id:
                L_frames = sim.track_frames(
                    d_L[v_id], L_threshold=L_threshold, seq_threshold=seq_threshold
                )  # list of frames sequences
                vid_path = video_path + v_id + ".mp4"
                s_id = int(df_train.loc[df_train["video_id"] == v_id, "signer_id_deepface"].item())

                for l in range(len(L_frames)):
                    if len(L_frames[l]) <= 16:
                        path_out = path_word + "/" + v_id + "_" + str(l) + ".mp4"
                        frames2video(vid_path, path_out, L_frames[l])

                        mid = int(len(L_frames[l]) / 2)
                        mid_frame = L_frames[l][mid]
                        start_frame = L_frames[l][0]
                        end_frame = L_frames[l][-1]
                        df1 = pd.DataFrame(
                            {
                                "video_ids": [v_id + "_" + str(l)],
                                "signer_id": [s_id],
                                "frames_nb": [len(L_frames[l])],
                                "start_frame": [start_frame],
                                "end_frame": [end_frame],
                                "mid_frame": [mid_frame],
                            }
                        )
                        if "results" in locals():
                            results = pd.concat([results, df1])
                        else:
                            results = df1

                    else:  # if the seq has more than 16 frames, we cut it to 16 frames
                        path_out = path_word + "/" + v_id + "_" + str(l) + ".mp4"
                        mid = int(len(L_frames[l]) / 2)
                        mid_frame = L_frames[l][mid]

                        frames2video(vid_path, path_out, L_frames[l][mid - 8 : mid + 7])

                        start_frame = L_frames[l][mid - 8]
                        end_frame = L_frames[l][mid + 7]
                        df1 = pd.DataFrame(
                            {
                                "video_ids": [v_id + "_" + str(l)],
                                "signer_id": [s_id],
                                "frames_nb": [16],
                                "start_frame": [start_frame],
                                "end_frame": [end_frame],
                                "mid_frame": [mid_frame],
                            }
                        )
                        if "results" in locals():
                            results = pd.concat([results, df1])
                        else:
                            results = df1

            # save files
            pickle.dump(d_plus, open(path_word + "/d_Lplus.pkl", "wb"))
            pickle.dump(d_moins, open(path_word + "/d_Lmoins.pkl", "wb"))
            if "results" in locals():
                results.to_csv(path_word + "/confidence_scores.csv", index=False)
        if "results" in locals():
            results.to_csv(path_word + "/confidence_scores.csv", index=False)

        # Clustering
        if clustering is True:
            cluster(path_word, results, feature_path, viz_clusters)

        # delete empty folders
        folders_names = next(walk(path_word), (None, None, []))[1]
        try:
            for folder in folders_names:
                folder_path = path_word + "/" + folder
                if len(os.listdir(folder_path)) == 0:  # Check if the folder is empty
                    print("remove ", folder)
                    shutil.rmtree(folder_path)
        except:
            pass


def create_word_videos_signer(
    L_word_names,
    Lwords,
    L_noword,
    L_word_neg,
    seq_threshold=3,
    sim_threshold=0.6,
    L_threshold=0.5,
    path_to_dict="Step1_Weakly_supervised_annotation/Dictionnaire5/",
    feature_path="../data/Mediapi/features_mediapi/swin/",
    n_max=100,
    clustering=True,
    viz_clusters=True,
    video_path="../data/Mediapi/video_crops_train/",
    df_path="mediapi_train_clean1.csv",
):
    """Create a dictionnary with folders for each sign.
    Each folder contains :
        - captured videos of the sign sorted by signer  and possibly sort in Kmeans folders (if clustering = True)
        - confidence-scores.csv : informations about the captured signs (video id, signer id, nb of frames, start frame, end frame, mid frame )
        - d_Lplus : dictionnary {vid : L+}
        - d_Lmoins : dictionnary {vid : L-}

    Args:
        L_word_names (list): Names of words (to make a folder)
        Lwords (list): we will select videos which subtitle contain at least one of the words of Lwords
        L_noword (list): we will not select videos which subtitles contain one of the words of L nowords
        L_word_neg (list): for L- computing (see paper)
        seq_threshold (int, optional): minimal sequence lenght . Defaults to 3.
        sim_threshold (float, optional): similarity threshold. Defaults to 0.6.
        L_threshold (float, optional): L score threshold. Defaults to 0.5.
        path_to_dict (str, optional): path to the dictionnary folder (to save signs). Defaults to "Step1_Weakly_supervised_annotation/Dictionnaire5/".
        feature_path (str, optional): path to the video feaures. Defaults to "../data/Mediapi/features_mediapi/swin/".
        n_max (int, optional): maximal number of positive examples. Defaults to 100.
        clustering (bool, optional): if we want to use clustering. Defaults to True.
        viz_clusters (bool, optional): if we want to visualize the datas. Defaults to True.
        video_path (str, optional): path to the continous videos. Defaults to "../data/Mediapi/video_crops_train/".
        df_path (str, optional): path to the metadatas csv. Defaults to "mediapi_train_clean1.csv".
    """

    os.makedirs(path_to_dict, exist_ok=True)  # create Dictionnary folder

    dico_cropId2subtitle = pickle.load(open("dico_cropId2subtitle.pkl", "rb"))  # load {vid : subtitle} dictionnary

    df_train = pd.read_csv(df_path)  # load metadata csv from Mediapi-RGB
    L_signer_ids = list(np.unique(df_train["signer_id_deepface"].values.tolist()))  # list of signers ids

    for i, word in enumerate(L_word_names):
        path_word = path_to_dict + word
        os.makedirs(path_word, exist_ok=True)  # create word folder to save captured signs

        for s_id in L_signer_ids:
            print(f"similarity with signer {s_id}")
            sim = Similarity(
                L_video_id=list(df_train[df_train.signer_id_deepface == s_id].video_id),
                word=word,
                L_word=Lwords[i],
                dico=dico_cropId2subtitle,
                feature_path=feature_path,
                pr=True,
                L_nowords=L_noword[i],
                n_max=n_max,
                L_word_neg=L_word_neg[i],
            )
            if sim.Video_nb > 1:
                d_L, d_plus, d_moins = sim.collect_L(sim_threshold=sim_threshold, Negative_ex=True)  # list of scores vectors
                L_positive_videos_id = sim.L_positive_video_id  # list of selected videos ids in which we look for the sign

                for v_id in L_positive_videos_id:
                    L_frames = sim.track_frames(
                        d_L[v_id], L_threshold=L_threshold, seq_threshold=seq_threshold
                    )  # list of frames sequences
                    vid_path = video_path + v_id + ".mp4"
                    os.makedirs(path_word + "/c" + str(int(s_id)), exist_ok=True)  # create a folder for each signer

                    for l in range(len(L_frames)):
                        if len(L_frames[l]) <= 16:
                            path_out = path_word + "/c" + str(int(s_id)) + "/" + v_id + "_" + str(l) + ".mp4"
                            frames2video(vid_path, path_out, L_frames[l])

                            mid = int(len(L_frames[l]) / 2)
                            mid_frame = L_frames[l][mid]
                            start_frame = L_frames[l][0]
                            end_frame = L_frames[l][-1]
                            df1 = pd.DataFrame(
                                {
                                    "video_ids": [v_id + "_" + str(l)],
                                    "signer_id": [s_id],
                                    "frames_nb": [len(L_frames[l])],
                                    "start_frame": [start_frame],
                                    "end_frame": [end_frame],
                                    "mid_frame": [mid_frame],
                                }
                            )
                            if "results" in locals():
                                results = pd.concat([results, df1])
                            else:
                                results = df1

                        else:  # if the seq has more than 16 frames, we cut it to 16 frames
                            path_out = path_word + "/c" + str(int(s_id)) + "/" + v_id + "_" + str(l) + ".mp4"
                            mid = int(len(L_frames[l]) / 2)
                            mid_frame = L_frames[l][mid]

                            frames2video(vid_path, path_out, L_frames[l][mid - 8 : mid + 7])

                            start_frame = L_frames[l][mid - 8]
                            end_frame = L_frames[l][mid + 7]
                            df1 = pd.DataFrame(
                                {
                                    "video_ids": [v_id + "_" + str(l)],
                                    "signer_id": [s_id],
                                    "frames_nb": [16],
                                    "start_frame": [start_frame],
                                    "end_frame": [end_frame],
                                    "mid_frame": [mid_frame],
                                }
                            )
                            if "results" in locals():
                                results = pd.concat([results, df1])
                            else:
                                results = df1

                # save files
                pickle.dump(d_plus, open(path_word + "/c" + str(int(s_id)) + "/d_Lplus.pkl", "wb"))
                pickle.dump(d_moins, open(path_word + "/c" + str(int(s_id)) + "/d_Lmoins.pkl", "wb"))
                if "results" in locals():
                    results.to_csv(path_word + "/confidence_scores.csv", index=False)
        if "results" in locals():
            results.to_csv(path_word + "/confidence_scores.csv", index=False)

        # Clustering
        if clustering is True:
            cluster(path_word, results, feature_path, viz_clusters)

        # delete empty folders
        folders_names = next(walk(path_word), (None, None, []))[1]
        try:
            for folder in folders_names:
                folder_path = path_word + "/" + folder
                if len(os.listdir(folder_path)) == 0:  # Check if the folder is empty
                    print("remove ", folder)
                    shutil.rmtree(folder_path)
        except:
            pass


def cluster(path_word, results, feature_path, viz_clusters):
    """
    Do clustering with captured videos from a word folder.

    Args:
        path_word (str): The path to the word folder
        results (pd.DataFrame): The dataframe containing the results (// confidence_score.csv)
        feature_path (str): The path to the features folder
        viz_clusters (bool): Whether to visualize the clusters or not
    """

    folders_names = next(walk(path_word), (None, None, []))[1]

    if len(folders_names) == 0:
        folders_names = [""]

    if len(results) > 3:  # we only do clustering if there are more then 3 results
        for i in range(len(results)):
            video_id = results.iloc[i].video_ids
            video_crop = "_".join(video_id.split("_")[:-1])
            video_feature_path = feature_path + video_crop + ".npy"
            mid_frame = results.iloc[i].mid_frame
            features = np.load(video_feature_path)
            feature = features[mid_frame, :].reshape(1, -1)
            feature = feature / norm(feature)

            if "Embs" not in locals():
                Embs = np.zeros((len(results), feature.shape[-1]))
            Embs[i] = feature

        # Visualization
        if viz_clusters is True:
            # Reduce data dimension with pca
            n_components = 3
            pca = PCA(n_components=n_components).fit(Embs)
            reduced_data = PCA(n_components=n_components).fit_transform(Embs)
            print(f" pca explained ratio : {pca.explained_variance_ratio_.sum()}")
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color="blue")
            plt.savefig(path_word + "/clusters.png")
            plt.show()

        # First compute the variance to decide wether we need to cluster
        kmeans = KMeans(n_clusters=1, random_state=0).fit(Embs)
        cluster_centers = kmeans.cluster_centers_
        v = norm(Embs - cluster_centers[0], axis=1).mean()

        os.makedirs(path_word + "/Kmeans0", exist_ok=True)

        if v < 0.35:
            print(f"Var = {v}, we don't need to cluster.")
            for fn in folders_names:
                files_names = os.listdir(path_word + "/" + fn)
                for file in files_names:
                    vid_path = path_word + "/" + fn + "/" + file
                    video_path_final = path_word + "/Kmeans0/" + file
                    shutil.copy(vid_path, video_path_final)
        else:
            # Using silhouette method to choose the number of clusters
            range_n_clusters = [2, 3, 4]
            L_silhouette = []
            for n_clusters in range_n_clusters:
                if len(Embs) > n_clusters:
                    clusterer = KMeans(n_clusters=n_clusters)
                    cluster_labels = clusterer.fit_predict(Embs)
                    silhouette_avg = silhouette_score(Embs, cluster_labels)
                    L_silhouette.append(silhouette_avg)
                    print(
                        "For n_clusters =",
                        n_clusters,
                        "The average silhouette_score is :",
                        silhouette_avg,
                    )

            if max(L_silhouette) < 0.18:  # if silhouette score is too low, we don't need to cluster
                print(f"max silhouette score = {max(L_silhouette)}. We don't need to cluster")
                for fn in folders_names:
                    files_names = os.listdir(path_word + "/" + fn)
                    for file in files_names:
                        vid_path = path_word + "/" + fn + "/" + file
                        video_path_final = path_word + "/Kmeans0" + "/" + file
                        shutil.copy(vid_path, video_path_final)
            else:
                n_clusters = np.argmax(L_silhouette) + 2

                # Clustering
                print(f"Var = {v} and max silhouette score = {max(L_silhouette)}, we need to cluster.")
                print(f"{n_clusters} clustering")
                kmeans = KMeans(n_clusters=n_clusters).fit(Embs)
                cluster_centers = kmeans.cluster_centers_

                # Create repositories
                p_kmeans = path_word + "/Kmeans"
                for i in range(n_clusters):
                    p1 = p_kmeans + str(i)
                    os.makedirs(p1, exist_ok=True)

                # Sorting videos in clusters
                for i, video_id in enumerate(results.video_ids):
                    video_path_final = p_kmeans + str(kmeans.labels_[i].item()) + "/" + video_id + ".mp4"
                    if "c" in folders_names[0]:
                        signer_cluster = int(results.loc[results["video_ids"] == video_id, "signer_id"].item())
                        vid_path = path_word + "/c" + str(signer_cluster) + "/" + video_id + ".mp4"
                        shutil.copy(vid_path, video_path_final)
                    else:
                        vid_path = path_word + "/" + video_id + ".mp4"
                        shutil.move(vid_path, video_path_final)
