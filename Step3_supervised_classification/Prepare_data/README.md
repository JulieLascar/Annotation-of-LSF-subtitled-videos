# Prepare data
Code to prepare datas as presented in the paper (sections 6.1 and 6.2).  

├── correct_annotations.ipynb   --------  _To manually correct the annotations (for overlapping cases)_  
├── data_utils.py ----------------------  _Utils fonctions to prepare the data_   
├── make_data.py -----------------------  _Prepare the data (need a bilingual dictionnary)_  
├── make_expVSnonexp_data.py -----------  _Prepare the data when comparing expert and non expert (need 2 bilingual dictionnaries)_  

### 1. First experiment : prepare expert and non expert datas
**run make_expVSnonexp_data.py** to prepare the data (do the steps one after the other) :  
    (i) make expert data  
    (ii) make non expert data  

In both cases :  
--> a new folder will be created (named for example 'Mediapi_Expert' and 'Mediapi_NonExpert') containing :  
- **saved_files folder** :  
d_gloses2Gid (dict) : {glose : gloseId}  
d_Gid2gloses (dict) : {gloseId : glose}  
d_Gid2Vid (dict) : {gloseId : [videos Id that contain this glose]}  
d_Vid2Labels (dict) : {Vid : [0011003333333000000]} annotation of each v_id  
L_videos (list) : list of all annotated videos  
ex_to_look (list): examples with signs that overlap and require manual inspection  
DTrain (list) : list of train annotated videos  
DVal (list) : list of val annotated videos  
DTest (list) : list of test annotated videos  

- **data_viz** : Visualisation of train, val, test datas
  

### 2. Second experiment : with a much larger lexicon  
**run make_data.py** to prepare the data (do the steps one after the other) :  
--> a new folder will be created (named for example 'Mediapi_363') containing :  
- **saved_files folder**  
- **data_viz**  