# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:01:55 2022

@author: franc
"""

import cv2
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import glob

#Function to define border. 
#Just erode some pixels into objects and dilate tooutside the objects. 
#This region would be the border. Replace border pixel value to something other than 255. 
def exclude_border(image, border_size=5, n_erosions=1):
    
    only_border = np.where(image != 127, 0, image)
    
    erosion_kernel = np.ones((3,3), np.uint8)      ## Start by eroding edge pixels
    eroded_image = cv2.erode(only_border, erosion_kernel, iterations=n_erosions)
    
    fill_inside =  np.where(image==255, 255, eroded_image)
    
    dilation_kernel = np.ones((3, 3), np.uint8)   #Kernel to be used for dilation
    dilated  = cv2.dilate(fill_inside, dilation_kernel, iterations = 1)
    
    original_no_border = np.where(dilated == 127, 0, dilated)
    plt.imshow(original_no_border)
    
    
    #fill_inside_complete = np.where((fill_inside == 0) & (only_border == 127) & (image_no_border == 0), 255, fill_inside)
    #plt.imshow(fill_inside)
    #original_no_border = np.where(fill_inside==127, 0, fill_inside)
    #plt.imshow(original_no_border)
    #fill_inside_complete = np.logical_and(border_kidney == 255, fill_inside == 0)
    #fill_inside_complete = np.logical_and(image==127 , fill_inside == 0)
    #fill_inside_complete_2 = np.logical_and(image==127 , fill_inside == 0)
    #fill_inside_complete = np.where(image==127 and fill_inside == 0, 255, fill_inside)
    #plt.imshow(fill_inside)
    '''
    erosion_kernel = np.ones((3,3), np.uint8)      ## Start by eroding edge pixels
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)  
 
    ## Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated  = cv2.dilate(eroded_image, dilation_kernel, iterations = 1)
    #plt.imshow(dilated, cmap='gray')
    
    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated == 255, 127, dilated) 	
    
    #In the above dilated image, convert the eroded object parts to pixel value 255
    #What's remaining with a value of 127 would be the boundary pixels. 
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)
    
    #plt.imshow(original_with_border,cmap='gray')
    
    return original_with_border
    '''
    return original_no_border

#select the path
from pathlib import Path
path = 'C:/Users/franc/OneDrive/Área de Trabalho/TESE/ResultadosV2/Calculos_3_classes/Pred_Modelo4/*'
#path = 'C:/Users/franc/OneDrive/Área de Trabalho/TESE/ResultadosV2/Calculos_3_classes/Pred_Modelo1/*'
for file in glob.glob(path):
    name = Path(file).stem  #Extract name so processed images can be saved under the same name
    img = imread(file)  #now, we can read each file since we have the full path
    #img_2 = imread('C:/Users/franc/OneDrive/Área de Trabalho/TESE/ResultadosV2/Calculos_3_classes/Train7_2_classes/masks/000002_seg.png')
    #plt.imshow(img)
    #process each image
    processed_image = exclude_border(img, border_size=3, n_erosions=1)
    #Save images with same name as original image/mask
    imsave("C:/Users/franc/OneDrive/Área de Trabalho/TESE/ResultadosV2/Calculos_3_classes/NoBorder/Pred_Modelo4/"+ name + "_Pred_Modelo4_border.png", processed_image)
    print("Finished processing image ", name)    