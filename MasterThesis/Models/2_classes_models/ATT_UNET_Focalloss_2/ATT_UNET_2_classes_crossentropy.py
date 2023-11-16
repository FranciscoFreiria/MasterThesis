# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:26:58 2022

@author: franc
"""

import os
import tensorflow as tf
import functions_2_classes_file as f
import numpy as np
import json
#import random
import time
from tqdm import tqdm
#from keras.metrics import MeanIoU
#from tensorflow.keras import losses
#from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from skimage.io import imsave
#import matplotlib.pyplot as plt
from segmentation_models.losses import CategoricalFocalLoss
#from segmentation_models.metrics import IOUScore, FScore
#from sklearn.metrics import confusion_matrix 

import sys
sys.stderr = sys.stdout

tf.keras.backend.clear_session() 
tf.config.list_physical_devices('GPU')


#############################################GLOBAL_VARIABLES####################################################################################
 # Hyperparameters
#[HYPER] Mudar testar ou correr 
BATCH_SIZE = 4
EPOCHS = 5

#seed = 42
#np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

NUM_CLASSES = 1

#[HYPER] Mudar testar ou correr  
IMAGE_PATH = './Train8/images/'
MASK_PATH = './Train8/masks/'

#[HYPER] Mudar testar ou correr 
AUG_IMAGE_PATH = f.create_folder('Train8_aug/images/')
AUG_MASK_PATH = f.create_folder('Train8_aug/masks/')
folderJson_test = f.create_folder('Json/Test')
folderJson_train = f.create_folder('Json/Train')
folderSave_results = f.create_folder('Save/Results/')
folderSave_graphs = f.create_folder('Save/Graphs/')
folderSave_pred = f.create_folder('Save/Pred/')
folderSave_graphs_Acc = f.create_folder('Save/Graphs/Accuracy/')
folderSave_graphs_Dice = f.create_folder('Save/Graphs/Dice/')
folderSave_graphs_IOU = f.create_folder('Save/Graphs/IOU/')
folderSave_graphs_Precision = f.create_folder('Save/Graphs/Precision/')
folderSave_graphs_Recall = f.create_folder('Save/Graphs/Recall/')
folderSave_graphs_Loss = f.create_folder('Save/Graphs/Loss/')
    
image_ids = next(os.walk(IMAGE_PATH))[2]
mask_ids = next(os.walk(MASK_PATH))[2]

image_ids = sorted(image_ids)
mask_ids = sorted(mask_ids)

######################################MAIN######################################################################################################

def main():
    #[HYPER] Mudar o numero de interações onde é necessario data_aug
    if(c<1):
        #criar data aug
        print('##############################')
        print('Generate data augmentation')
        print('##############################')
        #[HYPER] Numero de imagens da dataaug
        num = 10
        f.Generate_aug(num, IMAGE_PATH, train_ids, MASK_PATH, train_mask_ids, AUG_IMAGE_PATH, AUG_MASK_PATH)
    else:
        print('##############################')
        print('No data augmentation generation needed')
        print('##############################')
        
    aug_image_ids = next(os.walk(AUG_IMAGE_PATH))[2]
    aug_mask_ids = next(os.walk(AUG_MASK_PATH))[2]  
    aug_image_ids = sorted(aug_image_ids)
    aug_mask_ids = sorted(aug_mask_ids)

    print('Load train images and masks')
    X_train = []
    Y_train = []
    for n, (i_id, m_id) in tqdm(enumerate(zip(train_ids + aug_image_ids, train_mask_ids + aug_mask_ids )), total=len(train_ids + aug_image_ids)):
        if (n < len(train_ids)):
            X_train.append(f.load_image(IMAGE_PATH, i_id))
            Y_train.append(f.load_mask(MASK_PATH, m_id))
        else:
            X_train.append(f.load_image(AUG_IMAGE_PATH, i_id))
            Y_train.append(f.load_mask(AUG_MASK_PATH, m_id))
    
    #[HYPER] Para astype(np.float32) nas focalloss           
    X_train = np.array(X_train)
    Y_train = np.array(Y_train).astype(np.float32)
            
    print('Load test image')
    X_test = []
    X_test = f.load_image(IMAGE_PATH, test_ids[0])
    X_test = np.array(X_test)
    X_test = np.expand_dims(X_test, axis=0)
    print('Load test mask')
    Y_test = []
    Y_test = f.load_mask(MASK_PATH, test_mask_ids[0])
    #[HYPER] Para astype(np.float32) nas focalloss    
    Y_test = np.array(Y_test).astype(np.float32)
    Y_test = np.expand_dims(Y_test, axis=0)
    print('Done!')
 
    X_train, Y_train = shuffle(X_train, Y_train)
    
    
    ####Sanity test####
    
    #image_x = random.randint(0, (len(Y_train[1::])-1))
    #f.plot_image(np.squeeze(X_train[image_x]))
    #f.plot_image(np.squeeze(Y_train[image_x]))
    
    ####Check test####
    '''
    for z in range(len(Y_train[1::])+1):
        print("#############Image################")
        f.plot_image(np.squeeze(X_train[z]))
        f.plot_image(np.squeeze(Y_train[z]))
        print("#############END################")
    '''
    ################################
    #Categorias
    
    #[HYPER] To categorical para 3 classes
    #Y_train_cat = to_categorical(Y_train, num_classes=NUM_CLASSES) 
    #Y_test_cat = to_categorical(Y_test, num_classes=NUM_CLASSES)
    ################################
    #Build Model 
    
    #[HYPER] Mudar Modelo
    unet_model = f.Attention_UNet(NUM_CLASSES)
    #unet_model = f.unet_model()
    
    #[HYPER] Mudar funcao de perda
    loss = [CategoricalFocalLoss()]
    #loss = [tf.keras.losses.BinaryCrossentropy()]
    #confusion_matrix = f.ConfusionMatrix(num_classes=2)
    #metric = [tf.keras.metrics.BinaryAccuracy(), FScore(name = 'dice_coef'), IOUScore(name = 'IOU_mean'), tf.keras.metrics.Precision()]
    #metrics=[confusion_matrix, confusion_matrix.accuracy,tf.keras.metrics.BinaryAccuracy()]
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(),
                       f.BinaryTruePositives(), f.BinaryTrueNegatives(),
                       f.BinaryFalsePositives(), f.BinaryFalseNegatives()])
    unet_model.summary()
    
    #Modelcheckpoint
    cb = [tf.keras.callbacks.LearningRateScheduler(f.step_decay)]
    ####################################
    #Results Train 
    start_train_time = time.time()
    
    results_train = unet_model.fit(X_train, Y_train, 
                    verbose=1,
                    batch_size = BATCH_SIZE,
                    shuffle=False,
                    callbacks=cb,
                    epochs=EPOCHS) 
    execution_train_time = (time.time() - start_train_time)
    
    test_id = test_ids[0].replace('.png', '')
    #[HYPER] Mudar nome do ficheiro de treino
    unet_model.save('ATT_UNet_2_classes_FL_' + test_id + '_' + str(c) + '_.hdf5')
    
    ####################################
    #Predict    
    start_test_time = time.time()
    
    Y_pred = unet_model.predict(X_test, verbose=1)
    evaluation_out = unet_model.evaluate(X_test, Y_test, verbose=1)

    execution_test_time = (time.time() - start_test_time)
    
    #[HYPER] Mudar para 3 classes
    Y_pred = np.squeeze(Y_pred, axis = 0)
    Y_pred_img = np.where(Y_pred >= 0.5, 1,0)
    Y_pred_img = Y_pred_img * 255    
    ####################################
    #Results Test
    
    imsave((folderSave_pred + test_id + '_' + str(c) + '.png'), Y_pred_img.astype(np.uint8), check_contrast = False)
    
    fig_pred_path = str((folderSave_results + test_id + '_fig_' + str(c) + '.png'))
    f.Plot_result(X_test, Y_test, np.squeeze(Y_pred_img), fig_pred_path)

    dice_array = []
    iou_array = []
    precision_array = []
    recall_array = []

    for l in range (len(results_train.history['binary_true_negatives'])):
        dice, iou, precision, recall =  f.metrics_class(results_train.history['binary_true_negatives'][l],
                                                                          results_train.history['binary_false_positives'][l],
                                                                          results_train.history['binary_false_negatives'][l],
                                                                          results_train.history['binary_true_positives'][l])        
        dice_array.append(round(float(dice),6))
        iou_array.append(round(float(iou),6))
        precision_array.append(round(float(precision),6))
        recall_array.append(round(float(recall),6))
   
   
    loss = results_train.history['loss']
    acc = results_train.history['binary_accuracy']
    loss_pred_path = os.path.join(folderSave_graphs_Loss, test_id + '_graph_' + str(c) + '.png')
    acc_pred_path = os.path.join(folderSave_graphs_Acc, test_id + '_graph_' + str(c) + '.png')
    dice_pred_path = os.path.join(folderSave_graphs_Dice, test_id + '_graph_' + str(c) + '.png')
    IoU_pred_path = os.path.join(folderSave_graphs_IOU, test_id + '_graph_' + str(c) + '.png')
    precision_pred_path = os.path.join(folderSave_graphs_Precision, test_id + '_graph_' + str(c) + '.png')
    recall_pred_path = os.path.join(folderSave_graphs_Recall, test_id + '_graph_' + str(c) + '.png')
    
    f.Plot_graph(acc, acc_pred_path, 'Training Accuracy' )
    f.Plot_graph(dice_array, dice_pred_path, 'Training Dice' )
    f.Plot_graph(iou_array, IoU_pred_path, 'Training IoU' )
    f.Plot_graph(precision_array, precision_pred_path, 'Training Precision' )
    f.Plot_graph(recall_array, recall_pred_path, 'Training Recall' )
    f.Plot_graph(loss, loss_pred_path, 'Training Loss' )       

    evaluation_train_it = []
    evaluation_train_it.append(results_train.history['loss'])
    evaluation_train_it.append(results_train.history['binary_true_negatives'])
    evaluation_train_it.append(results_train.history['binary_false_positives'])
    evaluation_train_it.append(results_train.history['binary_false_negatives'])
    evaluation_train_it.append(results_train.history['binary_true_positives'])
    evaluation_train_it.append(results_train.history['binary_accuracy'])
    evaluation_train_it.append(dice_array)
    evaluation_train_it.append(iou_array)
    evaluation_train_it.append(precision_array)
    evaluation_train_it.append(recall_array)
    evaluation_train_it.append(round(execution_train_time,8))

    dice_test, iou_test, precision_test, recall_test  = f.metrics_class(evaluation_out[3], evaluation_out[4], evaluation_out[5], evaluation_out[2])
    
    evaluation_test_it = []
    evaluation_test_it.append(evaluation_out[0])
    evaluation_test_it.append(evaluation_out[2])
    evaluation_test_it.append(evaluation_out[3])
    evaluation_test_it.append(evaluation_out[4])
    evaluation_test_it.append(evaluation_out[5])
    evaluation_test_it.append(evaluation_out[1])

    evaluation_test_it.append(round(float(dice_test),5))
    evaluation_test_it.append(round(float(iou_test),5))
    evaluation_test_it.append(round(float(precision_test),5))
    evaluation_test_it.append(round(float(recall_test),5))
    evaluation_test_it.append(round(execution_test_time,7))
    
    return evaluation_train_it, evaluation_test_it

######################################CICLO######################################################################################################

for n, (i_ids, m_ids) in enumerate(zip(image_ids, mask_ids)):
    
    test_ids = [image_ids.pop(0)]
    test_mask_ids = [mask_ids.pop(0)]
    train_ids = image_ids
    train_mask_ids = mask_ids
    global c
    
    print('##############################')
    print('Test Image:',test_ids)
    print('Test Mask:',test_mask_ids)
    print('##############################')
    print('##############################')
    print('Train Image:',train_ids)
    print('Train Mask:',train_mask_ids)
    print('##############################')

    if (n >=0):
        for c in range(1):    	
            if __name__ == "__main__":
                evaluation_train_it, evaluation_test_it  = main()
                train_id = {
                        'Test_img': {'name': test_ids[0],
                                     'Test_id' : c,
                                     'Train_time' : evaluation_train_it[-1],
                                     'epochs' : len(evaluation_train_it[0]),
                                     'Loss': evaluation_train_it[0],
                                     'cm' : {'tn':evaluation_train_it[1],
                                             'fp':evaluation_train_it[2],
                                             'fn':evaluation_train_it[3],
                                             'tp':evaluation_train_it[4]
                                             },
                                     'Accuracy': evaluation_train_it[5],
                                     'DiceCoef': evaluation_train_it[6],
                                     'IoU': evaluation_train_it[7],
                                     'Precision': evaluation_train_it[8],
                                     'Recall': evaluation_train_it[9],
                                         }
                                     }
                test_id = {
                        'Test_img': {'name' : test_ids[0],
                                    'Test_id' : c,
                                     'Test_time': evaluation_test_it[-1],
                                     'Loss'    : evaluation_test_it[0],
                                     'cm' : {'tn':evaluation_test_it[2],
                                             'fp':evaluation_test_it[3],
                                             'fn':evaluation_test_it[4],
                                             'tp':evaluation_test_it[1]
                                             },
                                     'Accuracy': evaluation_test_it[5],
                                     'DiceCoef': evaluation_test_it[6],
                                     'IoU'     : evaluation_test_it[7],
                                     'Precision' : evaluation_test_it[8],
                                     'Recall' : evaluation_test_it[9]
                                 }
                            }
            test_id_name = test_ids[0].replace('.png', '')

            train_filename=os.path.join(folderJson_train,'resultados_train_' + test_id_name + '_' + str(c) + '.json')
            test_filename=os.path.join(folderJson_test,'resultados_test_' + test_id_name + '_' + str(c) + '.json')


            with open(train_filename, 'w') as fout:
               json.dump(train_id,fout,indent=2)
            fout.close() 
            with open(test_filename, 'w') as fout:
                json.dump(test_id,fout,indent=2)
            fout.close()
            print(Stop)
            tf.keras.backend.clear_session()        

    else:
        print('##############################')
        print('Nao faz nada')
        print('##############################')
    image_ids.append(test_ids[0])
    mask_ids.append(test_mask_ids[0])

