import sys
import os
import random
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from itertools import groupby, count
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append("Similarity_method/")


class Similarity:
    """
    Similarity class

    """

    def __init__(
        self,
        L_video_id: list,  # a list of video ids
        word: str,  # name of the list of words
        L_word: list,  # list of words
        dico: dict,  # a dictionnary that maps each video id with its subtitle
        feature_path: str = "../data/Mediapi/features_mediapi/swin/",
        pr: bool = True,  # gives infos while working
        L_nowords: list = [],  # list of words prohibited un subtitles
        L_word_neg: list = [],
        n_max=100,  # maximum nb of positives videos
    ):
        self.L_video_id = L_video_id
        self.word = word
        self.L_word = L_word
        self.L_nowords = L_nowords
        self.dico = dico
        self.feature_path = feature_path
        self.pr = pr
        self.L_word_neg = L_word_neg
        self.n_max = n_max
        self.L_video_id_with_word = self.get_videos_id_with_word()  # list of all the video ids that contain that special word
        self.L_video_id_without_word = self.get_videos_id_without_word()
        self.Video_nb = len(self.L_video_id_with_word)
        self.Features = self.set_Features()  # list of features of each video from L_video_id_with_word

        print(f"There are {self.Video_nb} videos with the words of the list {self.word}.\n")

    def get_videos_id_with_word(self):
        """
        Given a list of words, retrieve all the videos which subtitles contain one of these words and no words from L_nowords.
        """
        L = []
        for video_id in self.L_video_id:
            subtitle = self.dico[video_id]
            for word in self.L_word:
                if (video_id not in L and word in subtitle) and (
                    self.L_nowords == [] or (True not in [w in subtitle for w in self.L_nowords])
                ):
                    if self.pr is True:
                        print(subtitle, " : ", video_id)
                    L.append(video_id)
        return L

    def get_videos_id_without_word(self):
        """
        Given a word, retrieve all the videos which subtitles don't contain this word.
        """
        L = []
        for video_id in self.L_video_id:
            subtitle = self.dico[video_id]
            if self.L_word_neg != []:
                for w in self.L_word_neg:
                    if self.word not in subtitle and (w in subtitle) and (video_id not in L):
                        L.append(video_id)
            else:
                if self.word not in subtitle:
                    L.append(video_id)
        return L

    def get_feature(self, i: int, list_videos=None):
        """
        Return the normalized features of the ith video of list_videos (default : list L_video_id_with_word).
        """
        if list_videos is None:
            list_videos = self.L_video_id_with_word
        if "swin" in self.feature_path:
            features_path = self.feature_path + list_videos[i] + ".npy"
            features = np.load(features_path)
        elif "I3D" in self.feature_path:
            features_path = self.feature_path + list_videos[i] + ".mat"
            features = sio.loadmat(features_path)["preds"]
        features_norm = norm(features, axis=1).reshape(features.shape[0], 1)
        return features / features_norm

    def set_Features(self):
        """
        Return the list of the embeddings of each video of the list L_video_id_with_word.
        """
        Features = []
        for k in range(self.Video_nb):
            Features.append(self.get_feature(k))
        return Features

    def sim_mat(self, i: int, j: int, viz=False):
        """
        Return the similarity matrix of the ith ant the jth video of the list L_video_id_with_word.
        """
        if i >= self.Video_nb:
            print(f"\nFirst index is out of range. Choose an index smaller than {self.Video_nb}.")
            return None

        if j >= self.Video_nb:
            print(f"\nSecond index is out of range. Choose an index smaller than {self.Video_nb}.")
            return None

        sim_mat = self.Features[i] @ self.Features[j].T

        # Visualization of the similarity matrix.
        if viz is True:
            fig = plt.figure(figsize=(16, 14))
            sns.heatmap(sim_mat)
            plt.ylabel(self.dico[self.L_video_id_with_word[i]], fontsize=16)
            plt.xlabel(self.dico[self.L_video_id_with_word[j]], fontsize=16)
            plt.show()
            if "I3D" in self.feature_path:
                feature_name = "i3d"
            elif "swin_base" in self.feature_path:
                feature_name = "swin_base"
            else:
                feature_name = "swin"
            os.makedirs("Step1_Weakly_supervised_annotation/Sim_images/", exist_ok=True)
            fig.savefig(f"Step1_Weakly_supervised_annotation/Sim_images/{self.word}{i}{j}_{feature_name}.png")

        return sim_mat

    def collect_L(self, sim_threshold=0.6, Negative_ex=True):
        """
        for each video of self.L_video_id_with_word, collect scores vectors L, L+ and L-.

        Args :
            sim_threshold (float):  threshold (default: 0.6)
            Negative_ex (bool): if True, compute  L-. If False, L- = 0  (default: True)

        Returns :
            d_L (dict) : dictionnary of final scores vectors {vid : L}
            d_L+ (dict): dictionnary of scores vectors for positive examples {vid : L+}
            d_L- (dict): dictionnary of scores vectors for negative examples {vid : L-}
        """
        # if there are too many videos, we randomly choose n positives examples
        if self.Video_nb > self.n_max:
            random.seed(42)  # fix seed for reproductibility
            L_positive_videos_idx = random.sample(list(range(0, self.Video_nb)), self.n_max)
            self.L_positive_video_id = [self.L_video_id_with_word[k] for k in L_positive_videos_idx]

        else:
            L_positive_videos_idx = list(range(0, self.Video_nb))  # self.L_positive_videos are the index of positive examples
            self.L_positive_video_id = [self.L_video_id_with_word[k] for k in L_positive_videos_idx]

        d_L = dict()
        d_Lplus = dict()
        d_Lmoins = dict()

        for r, k in enumerate(L_positive_videos_idx):
            vid = self.L_positive_video_id[r]
            d_Lplus[vid] = np.zeros(self.Features[k].shape[0])
            d_Lmoins[vid] = np.zeros(self.Features[k].shape[0])

        for r, i in enumerate(L_positive_videos_idx[:-1]):
            for j in L_positive_videos_idx[r + 1 :]:
                sim_mat = self.sim_mat(i, j)
                max1 = np.amax(sim_mat, axis=1)
                max2 = np.amax(sim_mat, axis=0)

                d_Lplus[self.L_video_id_with_word[i]] += np.where(max1 < sim_threshold, 0, 1)
                d_Lplus[self.L_video_id_with_word[j]] += np.where(max2 < sim_threshold, 0, 1)

        for k in range(len(L_positive_videos_idx)):
            vid = self.L_positive_video_id[k]
            d_Lplus[vid] = d_Lplus[vid] / (len(L_positive_videos_idx) - 1)

        if Negative_ex is True:
            n_moins = 3 * len(L_positive_videos_idx)
            random.seed(42)

            if n_moins < len(self.L_video_id_without_word):
                L_negative_videos_idx = random.sample(list(range(0, len(self.L_video_id_without_word))), n_moins)

            else:
                L_negative_videos_idx = list(range(0, len(self.L_video_id_without_word)))

            for i in L_positive_videos_idx:
                vid = self.L_video_id_with_word[i]
                for j in L_negative_videos_idx:
                    sim_mat = self.Features[i] @ self.get_feature(j, self.L_video_id_without_word).T
                    max1 = np.amax(sim_mat, axis=1)
                    d_Lmoins[vid] += np.where(max1 < sim_threshold, 0, 1)

            for k in L_positive_videos_idx:
                vid = self.L_video_id_with_word[k]
                d_Lmoins[vid] = d_Lmoins[vid] / (len(L_negative_videos_idx) - 1)

        for k in L_positive_videos_idx:
            vid = self.L_video_id_with_word[k]
            d_L[vid] = d_Lplus[vid] - d_Lmoins[vid]

        return d_L, d_Lplus, d_Lmoins

    def track_frames(self, L, L_threshold=0.1, seq_threshold=3):  # original paper threshold = 0
        """
        For d_L obtained with collect_L, compute sequences with frames which scores are above the threshold.

        Args :  d_L (dict)
                L_threshold (float)
                seq_threshold (int) : sequences must have a lenght above seq_threshold to be considered

        Returns :
            L_frames (list) : list of sequences with frames which scores are above the threshold

        """
        idx = np.where(L > L_threshold)[0]
        L_frames = []
        L_tmp = list(idx)
        L_tmp.sort()
        c = count()
        try:
            while len(L_tmp) >= seq_threshold:
                L1 = max((list(g) for _, g in groupby(L_tmp, lambda x: x - next(c))), key=len)
                if len(L1) >= seq_threshold:
                    L_frames.append(L1)
                L_tmp = list(set(L_tmp) - set(L1))
                L_tmp.sort()
        except:
            print("no success")
        return L_frames
