# Annotation of LSF subtitled videos without a pre-existing dictionary (LREC-Coling Workshop 2024)

This repository is the official implementation of **Annotation of LSF subtitled videos without a pre-existing dictionary**.  
The aim of our current project is to contribute in developing resources and automatic analysis of LSF videos.  
We propose a three stages approach for the automatic annotation of lexical units in LSF videos, using a subtitled corpus without annotation.  

### [Step 1 : weakly supervised segmentation of specific signs in the videos, without use of any isolated example](Step1_Weakly_supervised_annotation)
![gif1](https://github.com/JulieLascar/Annotation-of-LSF-subtitled-videos/assets/97949668/b861d7af-c5a7-46b2-933e-a800b289e327)

In this section, we construct a dictionary of signs captured from continuous subtitled videos of Mediapi-RGB Dataset. 

This method is based on similarity calculation.

Content
-----------------------

├── similarity.py   ---------------  _Similarity class to calculate L vectors for a given label_  
├── create_wordvideos.py -----  _Functions to create a bilingual lexicon {label : LSF videos}_   
├── main.py ------------------------  _File to run_  
├── video_utils.py ---------------  _Make videos from list of frames, change fps_  


### Step 2 : expert reviewing of the segmented signs.
- Define the quality level of each sign.
- Identify the variants that were not clustered during Step 1 (as for friday)
 ![friday](https://github.com/JulieLascar/Annotation-of-LSF-subtitled-videos/assets/97949668/0ecabffb-7aa0-4693-81af-40193c7baf89)
 
### [Step 3 : supervised classification](Step3_supervised_classification)

The figure below is a comparison between the predictions of 2 classifiers which have been trained with automatic annotated datas previously checked non-experts (top) and experts (middle), and a ground truth (bottom) on a test video with the subtitle : "But the G7 countries -
Canada, France, Germany, Italy, Japan, the United Kingdom and the United States - reached an agreement on Saturday"

 ![Comparison between the predictions of the non-expert (top), the expert (middle) classifiers and a ground truth (bottom) on a test video](images/g7_new.png)

Coming soon :  
- link to get Mediapi Swin Video features  
- link to our paper
/home/jlascar/Documents/Annotation-of-LSF-subtitled-videos/Step1_Weakly_supervised_annotation
