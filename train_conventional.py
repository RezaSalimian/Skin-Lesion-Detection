import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import feature_extraction as feature_extraction
import pickle
import warnings
import argparse
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

def getPredictionScore(y_lab, y_pred):   
    M_ACCURACY = 'accuracy'
    M_F1_SCORE = 'f1-score'
    M_COHEN_KAPPA = 'Cohen kappa'
    M_CONFUSION_MATRIX = 'Confusion Matrix'
   
    scores = {}
    scores[M_ACCURACY] = accuracy_score(y_lab, y_pred)
    scores[M_F1_SCORE] = f1_score(y_lab, y_pred, labels=None, average='macro', sample_weight=None)
    scores[M_COHEN_KAPPA] = cohen_kappa_score(y_lab, y_pred)
    scores[M_CONFUSION_MATRIX] = confusion_matrix(y_lab, y_pred)
    
    return scores

def imageResize(image, imgSize): 
    # pad image to the max dimension
    top = (max(image.shape[:2]) - image.shape[0])//2
    bottom = max(image.shape[:2]) - image.shape[0] - top
    left = (max(image.shape[:2]) - image.shape[1])//2
    right = max(image.shape[:2]) - image.shape[1] - left
    imagePadded = cv2.copyMakeBorder(image, top, bottom, left, right,cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # resize image
    imageResize = cv2.resize(imagePadded, (imgSize, imgSize))
    return imageResize

def dataPrepare(input_path, imgSize=None, keypoint_detector=None):
    features = []
    labels = []
    folder_list = os.listdir(input_path)
    
    for folder in folder_list:
        print('Processing: ' + folder)       
        folder_path = os.path.join(input_path, folder)
        file_list = os.listdir(folder_path)
        for filename in file_list:
            image = cv2.imread(os.path.join(folder_path, filename))[:, :, :3] # ignore alpha channel
            if imgSize is not None:
                image = imageResize(image, imgSize)
            feature = feature_extraction.extract_features(image)
            features.append(feature)
            labels.append(folder)
            
    return np.array(features), np.array(labels) 

def trainModel(model, X_tr, y_tr, parameters, n_splits=3):
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X=X_tr, y=y_tr)
    clf = GridSearchCV(model, parameters, cv=splits, scoring=make_scorer(accuracy_score))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') 
        clf.fit(X_tr, y_tr)

    return clf.best_estimator_, clf.best_score_, clf.best_params_


def evaluateModel(model, X_tr, y_tr, X_test, y_test):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  
        model.fit(X_tr, y_tr)
        
    # Evaluate on test set
    y_pred = model.predict(X_test)
    scores = None
    if y_test is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            scores = getPredictionScore(y_test, y_pred)

    return model, y_pred, scores

def buildModels(X_train, y_train):
    models = []
    model_params = []
    model_names = []
    
    # Random forest 
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            models.append(RandomForestClassifier(max_features='sqrt', class_weight='balanced', random_state=0))
            model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth]})
            model_names.append('Random Forest')   
    
    # GB
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            for learning_rate in [0.01, 0.1]:
                models.append(GradientBoostingClassifier(subsample=0.7, max_features='sqrt', random_state=0))
                model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth], 'learning_rate':[learning_rate]})
                model_names.append('Gradient Boosting Machine')
    
    # SVM
    for kernel in ['linear', 'rbf']:
        for C in [1.0, 10.0, 100.0, 1000.0]:
            models.append(SVC(probability=True, gamma='auto', tol=0.001, cache_size=200, class_weight='balanced',
                              random_state=0,
                              decision_function_shape='ovr'))
            model_params.append({'kernel':[kernel], 'C':[C]})
            model_names.append('Support Vector Machine')
    
    # LR
    for penalty in ['l1', 'l2']:
        for C in [1.0, 10.0, 100.0, 1000.0]:
            models.append(linear_model.LogisticRegression(max_iter=500, solver='liblinear', multi_class='ovr',
                                                          class_weight='balanced', random_state=0))
            model_params.append({'penalty':[penalty], 'C':[C]})
            model_names.append('Logistic Regression')
        
    # KNN
    for n_neighbors in [5, 10, 15]:
        for weights in ['uniform', 'distance']:
            models.append(KNeighborsClassifier())
            model_params.append({'n_neighbors':[n_neighbors], 'weights':[weights]})
            model_names.append('K Nearest Neighbour')
    
    fitted_models = []
    model_scores = []
    for i in range(len(models)):
        print('performing test on the model {} of {}: {}'.format((i+1), len(models), model_names[i]))
        model = models[i]
        fitted_cv, _, _ = trainModel(model=model, X_tr=X_train, y_tr=y_train, parameters=model_params[i])
        fitted_whole_set, _, score = evaluateModel(model=fitted_cv, X_tr=X_train, y_tr=y_train,
                                                    X_test=X_test, y_test=y_test)
        fitted_models.append(fitted_whole_set)
        model_scores.append(score)
        print(model_names[i], score)
        
    return fitted_models, model_names, model_scores

if __name__ == '__main__':
    
    WORK_DIRECTORY = os.getcwd()   
    DATASET_FOLDER = 'data'
    TRAIN_FOLDER = 'train'
    TEST_FOLDER = 'test'
    data_path = os.path.join(WORK_DIRECTORY, DATASET_FOLDER)
    imgSize = None
    n_splits = 3
    save_model = 'saved_model'
    save_path = os.path.join(WORK_DIRECTORY, save_model)

    # Extract features and labels for train set and test set
    train_path = os.path.join(data_path, TRAIN_FOLDER)
    test_path = os.path.join(data_path, TEST_FOLDER)
    X_train_features, y_train = dataPrepare(train_path, imgSize=imgSize)
    X_test_features, y_test = dataPrepare(test_path, imgSize=imgSize)

    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train_features)
    X_test = scaler.transform(X_test_features)

    base_models, base_model_names, base_model_scores = buildModels(X_train, y_train)
    M_ACCURACY = 'accuracy'
    M_F1_SCORE = 'f1-score'
    M_COHEN_KAPPA = 'Cohen kappa'
    M_CONFUSION_MATRIX = 'Confusion Matrix'
    
    # Summarize model performance
    model_df = pd.DataFrame({'model': base_model_names,
                            M_ACCURACY: [score[M_ACCURACY] for score in base_model_scores],
                            M_F1_SCORE: [score[M_F1_SCORE] for score in base_model_scores],
                            M_COHEN_KAPPA: [score[M_COHEN_KAPPA] for score in base_model_scores],
                            M_CONFUSION_MATRIX: [score[M_CONFUSION_MATRIX] for score in base_model_scores]})
    model_df = model_df[['model', M_ACCURACY, M_F1_SCORE, M_COHEN_KAPPA,
                         M_CONFUSION_MATRIX]]
    model_df.to_csv(os.path.join(WORK_DIRECTORY, 'summary_stacking_model.csv'), index=False)
    model_df.sort_values(by=[M_ACCURACY, M_F1_SCORE, M_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
