# Src
Main scripts to train, infer and visualize the models' predictions on sign language videos.

├── datasets.py ------------- _Dataset classes for training and inferencing_  
├── evaluation.py ----------- _Evaluate models (compute loss and metrics)_  
├── models.py --------------- _Models used for classification_  
├── test_inference.py ------- _Inference of a trained model on test set and print detected gloses_  
├── test_score.py ----------- _Compute score on test set_  
├── test_viz_2preds.py ------ _Visualize sign segmentation and compare 2 models_  
├── test_viz.py ------------- _Visualize sign segmentation_  
├── training.py ------------- _Train models_  
├── training_utils.py ------- _Utils fonctions for training_  
└── utils.py ---------------- _Misc fonctions_  

### 1. Train the model 
run training.py (scores are saved in training_results.csv, best model is saved in trained_models)  
### 2. Choose one trained model and do inference on the test set 
run test_inference.py
### 3. Compute test score 
run test_score.py
### 4. Visualize model predictions
run test_viz.py or test_viz_2preds.py
