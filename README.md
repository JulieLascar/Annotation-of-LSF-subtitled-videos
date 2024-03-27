# Annotation of LSF subtitled videos without a pre-existing dictionary (LREC-Coling Workshop 2024)

This repository is the official implementation of **Annotation of LSF subtitled videos without a pre-existing dictionary**.
The aim of our current project is to contribute in developing resources and automatic analysis of LSF videos.
We propose a three stages approach for the automatic annotation of lexical units in LSF videos, using a subtitled corpus without annotation.

Step 1 : weakly supervised segmentation of specific signs in the videos, without use of any isolated example.  
Step 2 : expert reviewing of the segmented signs.  
Step 3 : supervised classification.  

The figure below is a comparison between the predictions of 2 classifiers which have been trained with automatic annotated datas previously checked non-experts (top) and experts (middle), and a ground truth (bottom) on a test video with the subtitle : "But the G7 countries -
Canada, France, Germany, Italy, Japan, the United Kingdom and the United States - reached an agreement on Saturday"

 ![Comparison between the predictions of the non-expert (top), the expert (middle) classifiers and a ground truth (bottom) on a test video](images/g7_new.png)

Coming soon :  
- link to get Mediapi Swin Video features  
- link to our paper