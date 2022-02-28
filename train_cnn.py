import os
import sys
sys.path.insert(0, os.getcwd()) # add current working directory to pythonpath
import numpy as np
import pandas as pd
import warnings
import argparse
import gc
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from tensorflow.keras import applications, callbacks, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.applications import Xception

def getPredictionScore(y_lab, y_pred):
    M_ACCURACY = 'accuracy'
    M_F1_SCOREA = 'f1-score'
    M_COHEN_KAPPA = 'Cohen kappa'
    M_CONFUSION_MATRIX = 'Confusion Matrix'
   
    scores = {}
    scores[M_ACCURACY] = accuracy_score(y_lab, y_pred)
    scores[M_F1_SCOREA] = f1_score(y_lab, y_pred, labels=None, average='macro', sample_weight=None)
    scores[M_COHEN_KAPPA] = cohen_kappa_score(y_lab, y_pred)
    scores[M_CONFUSION_MATRIX] = confusion_matrix(y_lab, y_pred)
    
    return scores

def modelCreate(model_name='efficientNet'): 
    CLASSES = ['benign', 'malignant']
    
    if model_name == 'efficientNet':
        imgSize = 224
        base_model = applications.EfficientNetB0(include_top=False, weights="imagenet",
                                        input_shape=(imgSize, imgSize, 3));
    elif model_name == 'mobileNet':
        imgSize = 224
        base_model = applications.MobileNet(include_top=False, weights='imagenet',
                                          input_shape=(imgSize, imgSize, 3),
                                            pooling=None)
        
    elif model_name == 'inceptionV3':
        imgSize = 299
        base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                              input_shape=(imgSize, imgSize, 3),
                                                pooling=None)
        
    elif model_name == 'resnet50':
        imgSize = 224
        base_model = applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                              input_shape=(imgSize, imgSize, 3),
                                                pooling=None)
        
    else:
        raise ValueError('Backbone can only be mobileNet, inceptionV3, or resnet50.')
        
    for layer in base_model.layers:
        layer.trainable = False 
    x = base_model.output  
    x = GlobalAveragePooling2D()(x)
    predict = Dense(len(CLASSES), activation='softmax')(x)
    network = Model(inputs=base_model.input, outputs=predict)

    return network, imgSize

def cnnModel(tr_path, test_path,
                     backbone='efficientNet', bs=32, epochs=100, lr=0.001,
                     save_path=None):
    model, imgSize = modelCreate(model_name=backbone)
    dataGenerator = ImageDataGenerator(
            rotation_range=90,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True)
    trainGenerator = dataGenerator.flow_from_directory(
            tr_path,
            target_size=(imgSize, imgSize),
            batch_size=bs,
            shuffle=True,
            class_mode='categorical')
    y_trn = trainGenerator.classes   
    validGenerator = dataGenerator.flow_from_directory(
            test_path,
            target_size=(imgSize, imgSize),
            batch_size=bs,
            shuffle=False,
            class_mode='categorical')
    
    callbackList = []
    stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, \
                patience=30, verbose=2, mode='auto', restore_best_weights=True)
    callbackList.append(stop)
    
    if save_path is not None:
        checkPoint = callbacks.ModelCheckpoint(os.path.join(save_path, backbone + '.h5'), monitor='val_acc', \
                    verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callbackList.append(checkPoint)
        
    print('training the ' + backbone + ' model\n' )
    
    # Train all layers
    for layer in model.layers:
        layer.trainable = True
    optimizer = optimizers.SGD(lr=lr/10.0, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit_generator(trainGenerator, validation_data=validGenerator, epochs=epochs, callbacks=callbackList, verbose=1)
    
    # Evaluate the model
    y_valid = validGenerator.classes
    y_valid_predict_prob = model.predict_generator(validGenerator)
    y_valid_predict = np.argmax(y_valid_predict_prob, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = getPredictionScore(y_valid, y_valid_predict)
        
    return model, history, scores
    
if __name__ == '__main__':
    WORK_DIRECTORY = os.getcwd()
    DATASET_FOLDER = 'data'
    TRAIN_FOLDER = 'train'
    TEST_FOLDER = 'test'
    data_path = os.path.join(WORK_DIRECTORY, DATASET_FOLDER)
    save_model = 'saved_model'
    save_path = os.path.join(WORK_DIRECTORY, save_model)
    
    cnnModel_names = ['efficientNet' ,'mobileNet', 'inceptionV3', 'resnet50']
    bs = 16
    epochs = 50
    lr = 0.001
    save_path = WORK_DIRECTORY
    
    tr_path = os.path.join(data_path, TRAIN_FOLDER)
    test_path = os.path.join(data_path, TEST_FOLDER)
      
    if save_path is not None:
        os.makedirs(os.path.join(save_path, 'cnnModels'), exist_ok=True)
        
    cnnModel_scores = []
    models =[]
    histories = []
    for backbone in cnnModel_names:
        model, history, scores = cnnModel(tr_path, test_path,
                     backbone=backbone, bs=bs, epochs=epochs, lr=lr,
                     save_path=os.path.join(save_path, 'cnnModels'))
        cnnModel_scores.append(scores)
        models.append(model)
        histories.append(history)
        model_name = backbone+'.h5'
        model.save(model_name)
        print(backbone, scores)
                
        K.clear_session()
        del model
        gc.collect()
            
    his_eff = histories[0]
    his_mob = histories[1]
    his_inc = histories[2]
    his_res = histories[3]
     
    import matplotlib.pyplot as plt
    
    # summarize history for loss
    plt.plot(his_eff.history['val_loss'])
    plt.plot(his_mob.history['val_loss'])
    plt.plot(his_inc.history['val_loss'])
    plt.plot(his_res.history['val_loss'])
    
    plt.title('cnn models loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['efficientnet', 'mobileNet','inceptionv3', 'resnet50' , ], loc='upper left')
    plt.show()


    plt.plot(his_eff.history['val_accuracy'])
    plt.plot(his_mob.history['val_accuracy'])
    plt.plot(his_inc.history['val_accuracy'])
    plt.plot(his_res.history['val_accuracy'])
    
    plt.title('cnn models accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['efficientnet', 'mobileNet','inceptionv3', 'resnet50' , ], loc='lower right')
    plt.show()

    M_ACCURACY = 'accuracy'
    M_F1_SCOREA = 'f1-score'
    M_COHEN_KAPPA = 'Cohen kappa'
    M_CONFUSION_MATRIX = 'Confusion Matrix'
 
 # Summarize model performance
    model_df = pd.DataFrame({'model': cnnModel_names,
                              M_ACCURACY: [score[M_ACCURACY] for score in cnnModel_scores],
                             M_F1_SCOREA: [score[M_F1_SCOREA] for score in cnnModel_scores],
                             M_COHEN_KAPPA: [score[M_COHEN_KAPPA] for score in\
                                                         cnnModel_scores],
                             M_CONFUSION_MATRIX: [score[M_CONFUSION_MATRIX] for score in\
                                                              cnnModel_scores]                            
                              })
    model_df = model_df[['model', M_ACCURACY, M_F1_SCOREA, M_COHEN_KAPPA,
                          M_CONFUSION_MATRIX]]
    model_df.to_csv(os.path.join(WORK_DIRECTORY, 'summary_cnnModel.csv'), index=False)
    model_df.sort_values(by=[M_ACCURACY, M_F1_SCOREA, M_COHEN_KAPPA],
                          ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))   
    
