#!/usr/bin/env python

import mxnet as mx
from mxnet import image
import numpy as np
import math
import os
import cv2
import natsort
import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import gluoncv
from gluoncv.utils.viz import get_color_pallete
from gluoncv.data.transforms.presets.segmentation import test_transform


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video',type=str, required=True,
	help='path to input video'),

parser.add_argument('-p', '--processor', default=True,
    help='compile model to gpu or cpu: type True for local gpu or False for cpu')

parser.add_argument('-o','--output',type=str,required=True,
                   help='video output name'),

parser.add_argument('-j','--json',type=str, default='True',
                   help='export to json True\False')
	
args = vars(parser.parse_args())

def detect_contours(mask):
    
    """
    takes masks --> return img with contours and contours corrdinates
        
    """
        
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    
    ret,thresh = cv2.threshold(mask,127,1,cv2.CALIB_CB_ADAPTIVE_THRESH)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(thresh, contours,-1, (255,0,0), 6)
    
    return img,contours
        

def dump_to_json(mask,contours,file_name,idx):
    
    """"
    takes mask, contours, output file name, and frame index -->
    returns json file saved to given path
    
    """
    
    import json
    
    with open('json_files/' + file_name,'w') as write_file:
        
             json.dump({
                "frame": idx ,
                "Label": 'Trees',
               "Annotations":[f"Object: {counter} Coordinates(X,Y): {contour.tolist()}" for counter,contour in enumerate(contours,1)],
                "height":mask.shape[0],
                "width":mask.shape[1]
            }, write_file, sort_keys=False, indent=1)    




def video_writer(image_folder,video_name,fps):
    
    video_name = video_name + '.avi'

    images = [img for img in os.listdir('Blended_Results')]
    images = natsort.natsorted(images,reverse=False)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name,fourcc , fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    
def blend(alpha,beta):

    os.mkdir('Blended_Results')

    for image in range(1,len(os.listdir('frames'))+1):

        frame='frames/frame%d.jpg' % image
        output='masks/mask_frame%d.png' % image

        img1= cv2.imread(frame)
        img2= cv2.imread(output)
        blended = cv2.addWeighted(src1=img1, alpha=alpha, src2=img2, beta=beta, gamma=5)

        cv2.imwrite('Blended_Results/Blended%d.jpg' % image, blended)
        
        
def display_video(video_name):
    import time

    cap = cv2.VideoCapture(video_name)

    while cap.isOpened():

        # Read the video file.
        ret, frame = cap.read()

        # If we got frames, show them.
        if ret == True:

             # Display the frame at same frame rate of recording
            time.sleep(1/75)
            cv2.imshow(video_name,frame)

            # Press Esc to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Or automatically break this whole loop if the video is over.
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    

def display(img,cmap=None):
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img)    
    


    
os.mkdir('json_files')
os.mkdir('frames')
os.mkdir('masks')

if args['processor']:

   ctx=mx.gpu(0)

else:
   
   ctx=mx.cpu(0)

model = gluoncv.model_zoo.get_model('fcn_resnet101_ade', pretrained=True,ctx=ctx)

cap = cv2.VideoCapture(args['video'])

frameRate = cap.get(5)

count=1;
while(cap.isOpened()):
    
    
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    
    if (ret != True):
        break
        
    if (frameId % math.floor(frameRate) == 0):
        filename ='frame%d.jpg' % count;
        cv2.imwrite('frames/' + filename, frame)

        img = image.imread('frames/' + filename)   

        img = test_transform(img, ctx)
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        mask = get_color_pallete(predict, 'ade20k')
        outp_filename ='mask_frame%d.png' % count;count+=1
        mask.save('masks/' + outp_filename)

        mmask = mpimg.imread('masks/' + outp_filename)
    
cap.release()


# Blending masks with original frames
# hyper-parameters indicates domination of each component
blend(0.5,0.5)

#visualize hyper-parameters affects for blending function
#img = cv2.imread('Blended_Results/Blended1.jpg')
#img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#display(img)

#composing new video from blended outputs
# 3rd component indicates number of frames per second
video_writer('Blended_Results',args['output'],6)
print('video saved to disk')

print('displaying video...')
display_video(args['output'] + '.avi')

if args['json']:
    #sorting image in folder by frame number 
    images = [img for img in os.listdir('masks')]
    images = natsort.natsorted(images,reverse=False)

    #contour detection and exporting annotations to JSON
    for idx,image in enumerate(images,1):

        img,contours = detect_contours(cv2.imread('masks/' + image))

        dump_to_json(img,contours,'frame%d.json' % idx,image)

print('json files saved to disk')
