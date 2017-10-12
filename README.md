This repository contains source code corresponding to the various kernels I ran on Kaggle( https://www.kaggle.com/nirajvermafcb )and some of my deep learning pet projects. 
The dataset used for various kernels can be found on "datasets" folder and some links mentioned below.

# Image Captioning using Tensorflow

Image Embedding containing 4096 dimensionl feature vector from VGG-16 model was used for training
the model using Transfer learning. 
Another embedding layer was utilised to map 4096 dimensional image
features into the space of 256 dimensional textual features.
Multi-layer Long Short Term Memory model
was built .
Masking technique was used to handle variable length input.

### Additional Downloads:
1) you will need VGG-16 image embeddings for the Flickr-30K dataset. These image embeddings are available on Google drive
( https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view ).

2)Additionally, you will need the corresponding captions for these images (results_20130124.token), which can also be downloaded Google Drive.
( https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view )

Mention proper path pointing to the datasets at the start of running the model.

# Comparison and Analysis of various Supervised classification ML models
Performance comparison of various data science models like Logistic Regression, SVM, Random Forest, Decision Trees, Neural Network (MLP Classifier) & Gaussian Naïve Bayes on the basis of Precision, Recall, F-1-Score, ROC-AUC curve using Mushrooms classification dataset. THe dataset can be found on datasets folder named "Mushroom Claasification"

# Exploring Principal Component Analysis
Detailed step-by-step study of PCA without using Scikit-learn using dataset of Human Resources Analytics. Basic concepts such as covariance matrix, Eigen values and Eigen vectors was analysed.The dataset used was "Human Resources Analytics" which can be found on datasets folder.

# Applying Principal component analysis with Scikit Learn

This notebook contains the application of Principal component analysis on the given dataset using Scikit-learn and the dimensions(also known as components) with maximum variance(where the data is spread out)was found out.Features with little variance in the data are then projected into new lower dimension. Then the models are trained on transformed dataset to apply machine learning models.Then I have applied Random forest Regressor on old and the transformed datasets and compared them. The dataset used was "crowdness at the campus gym" which can be found in dataset folder

# Detail-analysis-of-Support-Vector-Machine
Detail study of SVM using dataset of Gender Recognition by voice, by comparing the default model with further tuned model. Various hyper-parameters such as kernel, C & gamma were tuned. The dataset used was "VoiceGender" which can be found on datasets folder.

# Data Visualisation of IPL statistics
Visualization of important stats were produced using Matplotlib and Seaborn library to analyse the trend and generate insights. Important stats included highest run getters, Toss-Win factor, total matches played-won factor, Most wins by big margin(Greater than 50 runs or more than 8 wickets)by teams etc. THe dataset used is no longer available.You can find the notebook with details of the dataset here: ( https://www.kaggle.com/nirajvermafcb/data-visualisation-for-ipl-datasets-1 ) 

# Detail Analysis of various Hospital Factors
Detail study on hospital dataset was done and interesting insights is done by means of visualisation. The dataset used was "Hospital General Information" which can be found in dataset folder

