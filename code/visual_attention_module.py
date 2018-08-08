# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:40:25 2018

@author: Nidhi
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


project_directory = os.path.abspath('..')
stshi_storage = os.path.join(project_directory,"data","va_output")

#set height and width of stshi templates
image_height = 200
image_width = 300


class Visual_Attention_Module():
    
    def process_scenes(video_src, duration):
        
        cap = cv2.VideoCapture(video_src) 
        ret, frame1 = cap.read()
        if ret == False:
            print("Could not read from " + str(video_src) + " !\n")

        
        frame_resized = cv2.resize(frame1, (image_width, image_height))
        prvs = cv2.cvtColor(frame_resized,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame_resized)
        hsv[...,1] = 255
        saliencyfilter = cv2.saliency.StaticSaliencyFineGrained_create();
        stshi_template = prvs.copy()

        '''
        Creating STSHI templates - 
        1. Decide on size of temporal window.
        2. Template = First frame's saliency
        Motion capture += Next frame motion saliency masked image - current frame motion saliency masked image
        '''
        
        frame_number = 0
        stshi_number = 0
        template_duration = duration
        
        while(cap.isOpened()):
    
            ret, frame = cap.read()
            if ret == False:        
                break
    
             
             
            #process every other frame.
            if frame_number%2==0:
                frame_number += 1;
                continue
            else:
                frame = cv2.resize(frame, (image_width, image_height))
                frame_number += 1;
           
            #dense optical flow
            next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            gray_motion = cv2.cvtColor(hsv,cv2.COLOR_BGR2GRAY)
            #rgb_motion = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
           
            #spatial saliency
            (success_filter, spatial_saliencyMap) = saliencyfilter.computeSaliency(frame);
    

            #threshmap of spatiotemporally salient locations
            spacetime_saliency_threshMap = cv2.threshold(spatial_saliencyMap+gray_motion, 80, 255,	cv2.THRESH_BINARY_INV)[1]#| cv2.THRESH_OTSU)[1];
            spacetime_saliencyMap = cv2.bitwise_and(frame, frame, mask = spacetime_saliency_threshMap)
    
            #sthi = old mhi blended with masked img
            stshi_template = cv2.addWeighted(spacetime_saliency_threshMap,0.6,stshi_template,0.4,0)
            
            if frame_number == template_duration:  
                frame_number = 0
                stshi_number += 1
                #save existing motion_history
                filename = os.path.join(stshi_storage,"stshi" + str(int(stshi_number)) + ".jpg")
                #DEBUG step:
                #print(filename,"is being saved")
                cv2.imwrite(filename, stshi_template)
                stshi_template = next.copy()   
            # Watch the visual attention module in action by using the commands below
            '''
            #The following commands work well with windows or Ubuntu/Debian systems with GTK+ 2.x 
            cv2.imshow('Raw Input',frame)
            cv2.imshow('Spatio-Temporal Saliency',spacetime_saliencyMap)
            cv2.imshow('STSHI',stshi_template)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break
                
            #after cap.release()
        cv2.destroyAllWindows()
            '''
            '''
            #If you want a simpler solution in Linux systems - use this instead:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(spacetime_saliencyMap, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(stshi_template, cv2.COLOR_BGR2RGB))
            plt.show()
            '''           
            
         
        cap.release()
    