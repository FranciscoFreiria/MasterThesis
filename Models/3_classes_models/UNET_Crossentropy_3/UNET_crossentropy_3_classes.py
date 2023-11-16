# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:26:58 2022

@author: franc
"""

import os
import tensorflow as tf
import functions_file as f
import numpy as np
import json
#import random
import time
from tqdm import tqdm
#from keras.metrics import MeanIoU
#from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from skimage.io import imsave
#import matplotlib.pyplot as plt
#from segmentation_models.losses import CategoricalFocalLoss
#from segmentation_models.metrics import IOUScore, FScore
#from sklearn.metrics import confusion_matrix 

tf.keras.backend.clear_session() 
tf.config.list_physical_devices('GPU')


#############################################GLOBAL_VARIABLES####################################################################################
 # Hyperparameters
#[HYPER] Mudar testar ou correr 
BATCH_SIZE = 8
EPOCHS = 75

#seed = 42
#np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

NUM_CLASSES = 3

#[HYPER] Mudar testar ou correr  
IMAGE_PATH = './Train7/images/'
MASK_PATH = './Train7/masks/'

#[HYPER] Mudar testar ou correr 
AUG_IMAGE_PATH = f.create_folder('Train7_aug/images/')
AUG_MASK_PATH = f.create_folder('Train7_aug/masks/')
folderJson_test = f.create_folder('Json/Test')
folderJson_train = f.create_folder('Json/Train')
folderCM_test = f.create_folder('CM/Test')
folderCM_train = f.create_folder('CM/Train')
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
    if(c<10):
        #criar data aug
        print('##############################')
        print('Generate data augmentation')
        print('##############################')
        #[HYPER] Numero de imagens da dataaug
        num = 1000
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
    Y_train_cat = to_categorical(Y_train, num_classes=NUM_CLASSES) 
    Y_test_cat = to_categorical(Y_test, num_classes=NUM_CLASSES)
    ################################
    #Build Model 
    
    #[HYPER] Mudar Modelo
    #att_unet_model = f.Attention_UNet(NUM_CLASSES)
    unet_model = f.unet_model()
    
    #[HYPER] Mudar funcao de perda
    #loss = [CategoricalFocalLoss()]
    loss = [tf.keras.losses.CategoricalCrossentropy()]
     
    #metric = [tf.keras.metrics.BinaryAccuracy(), FScore(name = 'dice_coef'), IOUScore(name = 'IOU_mean'), tf.keras.metrics.Precision()]
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=[f.ConfusionMatrix(num_classes = 3), tf.keras.metrics.CategoricalAccuracy()], 
    sample_weight_mode='temporal')
    unet_model.summary()
    
    #Modelcheckpoint
    cb = [tf.keras.callbacks.LearningRateScheduler(f.step_decay)]
    ####################################
    #Results Train 
    start_train_time = time.time()
    #[HYPER] Mudar Y_Train para 3 classes
    results_train = unet_model.fit(X_train, Y_train_cat, 
                    verbose=1,
                    batch_size = BATCH_SIZE,
                    shuffle=False,
                    callbacks=cb,
                    epochs=EPOCHS) 
    execution_train_time = (time.time() - start_train_time)
    
    test_id = test_ids[0].replace('.png', '')
    #[HYPER] Mudar nome do ficheiro de treino
    unet_model.save('UNet_3_classes_CE_' + test_id + '_' + str(c) + '_.hdf5')
    
    ####################################
    #Predict    
    start_test_time = time.time()
    
    Y_pred = unet_model.predict(X_test, verbose=1)
    #[HYPER] Mudar para 3 classes
    evaluation_out = unet_model.evaluate(X_test, Y_test_cat, verbose=1)

    execution_test_time = (time.time() - start_test_time)
    
    #[HYPER] Mudar para 3 classes
    Y_pred_img = np.argmax(Y_pred, axis=-1)[0,:,:]
    Y_pred_img = Y_pred_img * 127.5    
    ####################################
    #Results Test
    
    imsave((folderSave_pred + test_id + '_' + str(c) + '.png'), Y_pred_img.astype(np.uint8), check_contrast = False)
    
    fig_pred_path = str((folderSave_results + test_id + '_fig_' + str(c) + '.png'))
    f.Plot_result(X_test, Y_test, np.squeeze(Y_pred_img), fig_pred_path)

    #evaluation_train_it = results_train.history
    #evaluation_train_it = list(evaluation_train_it.values())
    
    dice_array = []
    iou_array = []
    precision_array = []
    recall_array = []
    for l in range (len(results_train.history['confusion_matrix'])):
        dice,iou, precision, recall =  f.metrics_class(results_train.history['confusion_matrix'][l])       
        dice_array.append(dice)
        iou_array.append(iou)
        precision_array.append(precision)
        recall_array.append(recall)


    loss = results_train.history['loss']
    acc = results_train.history['categorical_accuracy']
    loss_pred_path = os.path.join(folderSave_graphs_Loss, test_id + '_graph_' + str(c) + '.png')
    acc_pred_path = os.path.join(folderSave_graphs_Acc, test_id + '_graph_' + str(c) + '.png')
    dice_pred_path = os.path.join(folderSave_graphs_Dice, test_id + '_graph_' + str(c) + '.png')
    IoU_pred_path = os.path.join(folderSave_graphs_IOU, test_id + '_graph_' + str(c) + '.png')
    precision_pred_path = os.path.join(folderSave_graphs_Precision, test_id + '_graph_' + str(c) + '.png')
    recall_pred_path = os.path.join(folderSave_graphs_Recall, test_id + '_graph_' + str(c) + '.png')
    
    dice_array_mean = [item[0] for item in dice_array]
    dice_array_cl0 = [item[1] for item in dice_array]
    dice_array_cl1 = [item[2] for item in dice_array]
    dice_array_cl2 = [item[3] for item in dice_array]
    
    iou_array_mean = [item[0] for item in iou_array]
    iou_array_cl0 = [item[1] for item in iou_array]
    iou_array_cl1 = [item[2] for item in iou_array]
    iou_array_cl2 = [item[3] for item in iou_array]
    
    precision_array_mean = [item[0] for item in precision_array]
    precision_array_cl0 = [item[1] for item in precision_array]
    precision_array_cl1 = [item[2] for item in precision_array]
    precision_array_cl2 = [item[3] for item in precision_array]
    
    recall_array_mean = [item[0] for item in recall_array]
    recall_array_cl0 = [item[1] for item in recall_array]
    recall_array_cl1 = [item[2] for item in recall_array]
    recall_array_cl2 = [item[3] for item in recall_array]
    
    f.Plot_graph(acc, acc_pred_path, 'Training Accuracy' )
    f.Plot_multiple(dice_array_mean, dice_array_cl0, dice_array_cl1, dice_array_cl2, dice_pred_path, 'Training Dice' )
    f.Plot_multiple(iou_array_mean, iou_array_cl0, iou_array_cl1, iou_array_cl2, IoU_pred_path, 'Training IoU' )
    f.Plot_multiple(precision_array_mean, precision_array_cl0, precision_array_cl1, precision_array_cl2, precision_pred_path, 'Training Precision' )
    f.Plot_multiple(recall_array_mean, recall_array_cl0, recall_array_cl1, recall_array_cl2, recall_pred_path, 'Training Recall' )
    f.Plot_graph(loss, loss_pred_path, 'Training Loss' )       
    
    evaluation_train_it = []
    evaluation_train_it.append(results_train.history['loss'])
    evaluation_train_it.append(results_train.history['categorical_accuracy'])
    
    evaluation_train_it.append(dice_array_mean)
    evaluation_train_it.append(dice_array_cl0)
    evaluation_train_it.append(dice_array_cl1)
    evaluation_train_it.append(dice_array_cl2)
    
    evaluation_train_it.append(iou_array_mean)
    evaluation_train_it.append(iou_array_cl0)
    evaluation_train_it.append(iou_array_cl1)
    evaluation_train_it.append(iou_array_cl2)
    
    evaluation_train_it.append(precision_array_mean)
    evaluation_train_it.append(precision_array_cl0)
    evaluation_train_it.append(precision_array_cl1)
    evaluation_train_it.append(precision_array_cl2)
    
    evaluation_train_it.append(recall_array_mean)
    evaluation_train_it.append(recall_array_cl0)
    evaluation_train_it.append(recall_array_cl1)
    evaluation_train_it.append(recall_array_cl2)
    
    evaluation_train_it.append(round(execution_train_time,8))
    
    dice_test, iou_test, precision_test, recall_test  = f.metrics_class(evaluation_out[1])
   
    
    evaluation_test_it = []
    evaluation_test_it.append(evaluation_out[0])
    evaluation_test_it.append(evaluation_out[2])
    evaluation_test_it.append(dice_test)
    evaluation_test_it.append(iou_test)
    evaluation_test_it.append(precision_test)
    evaluation_test_it.append(recall_test)
    evaluation_test_it.append(round(execution_test_time,7))
    
    np.save(os.path.join(folderCM_train, test_id + '_cm_train_' + str(c) + '.npy'), results_train.history['confusion_matrix'])
    np.save(os.path.join(folderCM_test, test_id + '_cm_test_' + str(c) + '.npy'), evaluation_out[1])
    
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
        for c in range(10):    	
            if __name__ == "__main__":
                evaluation_train_it, evaluation_test_it = main()
                train_id = {
                        'Test_img': {'name': test_ids[0],
                                     'Test_id' : c,
                                     'Train_time' : evaluation_train_it[-1],
                                     'epochs' : len(evaluation_train_it[0]),
                                     'Loss': evaluation_train_it[0],
                                     'Accuracy': evaluation_train_it[1],
                                     'DiceCoef' : {'Mean Dice':evaluation_train_it[2],
                                             'Dice_class_0':evaluation_train_it[3],
                                             'Dice_class_1':evaluation_train_it[4],
                                             'Dice_class_2':evaluation_train_it[5],
                                             },
                                    'IoU' : {'Mean IoU':evaluation_train_it[6],
                                             'IoU_class_0':evaluation_train_it[7],
                                             'IoU_class_1':evaluation_train_it[8],
                                             'IoU_class_2':evaluation_train_it[9],
                                             },
                                    'Precision' : {'Mean Precision':evaluation_train_it[10],
                                             'Precision_class_0':evaluation_train_it[11],
                                             'Precision_class_1':evaluation_train_it[12],
                                             'Precision_class_2':evaluation_train_it[13],
                                             },
                                    'Recall' : {'Mean Recall':evaluation_train_it[14],
                                             'Recall_class_0':evaluation_train_it[15],
                                             'Recall_class_1':evaluation_train_it[16],
                                             'Recall_class_2':evaluation_train_it[17],
                                             }
                                         }
                                     }
                test_id = {
                        'Test_img': {'name': test_ids[0],
                                     'Test_id' : c,
                                     'Train_time' : evaluation_test_it[-1],
                                     'Loss': evaluation_test_it[0],
                                     'Accuracy': evaluation_test_it[1],
                                     'DiceCoef' : {'Mean Dice':evaluation_test_it[2][0],
                                             'Dice_class_0':evaluation_test_it[2][1],
                                             'Dice_class_1':evaluation_test_it[2][2],
                                             'Dice_class_2':evaluation_test_it[2][3],
                                             },
                                    'IoU' : {'Mean IoU':evaluation_test_it[3][0],
                                             'IoU_class_0':evaluation_test_it[3][1],
                                             'IoU_class_1':evaluation_test_it[3][2],
                                             'IoU_class_2':evaluation_test_it[3][3],
                                             },
                                    'Precision' : {'Mean Precision':evaluation_test_it[4][0],
                                             'Precision_class_0':evaluation_test_it[4][1],
                                             'Precision_class_1':evaluation_test_it[4][2],
                                             'Precision_class_2':evaluation_test_it[4][3],
                                             },
                                    'Recall' : {'Mean Recall':evaluation_test_it[5][0],
                                             'Recall_class_0':evaluation_test_it[5][1],
                                             'Recall_class_1':evaluation_test_it[5][2],
                                             'Recall_class_2':evaluation_test_it[5][3],
                                             }
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
            tf.keras.backend.clear_session()        

    else:
        print('##############################')
        print('Nao faz nada')
        print('##############################')
    image_ids.append(test_ids[0])
    mask_ids.append(test_mask_ids[0])

