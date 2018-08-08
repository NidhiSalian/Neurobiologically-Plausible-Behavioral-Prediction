# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:44:27 2018

@author: Nidhi
"""

import os
import cv2
import pandas as pd
import numpy as np

from visual_attention_module import Visual_Attention_Module
print("\n\nVisual Attention Module loaded.\n\n")

project_directory = os.path.abspath('..')
annotations_path = os.path.join(project_directory,"data","annotations")
video_labels_path_train = os.path.join(annotations_path, "Charades_v1_train.csv")
video_labels_path_test = os.path.join(annotations_path, "Charades_v1_test.csv")
activity_classes_path = os.path.join(annotations_path, "Charades_v1_classes.txt")
video_path = os.path.join(project_directory,"data","videos")
stshi_path =  os.path.join(project_directory,"data","va_output")


def empty_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            os.remove(os.path.join(root, file))

class Dataset(object):
    
    def hot_encode(self, label_type, all_labels, curr_labels):
        hot_vector = [0.0] * len(all_labels)
        indices_1 = []
        if(label_type == "activity"):
            #extract only the class categories - not the timestamps
            curr_labels = [words.split(None, 1)[0] for words in curr_labels.split(';')]
            for label in curr_labels:
                indices_1.append(all_labels.index(label))
        else:
            indices_1 = np.where(all_labels==curr_labels)[0]
        for index in indices_1:
            hot_vector[index] = 1.0
            
        #DEBUG step:
        #print(hot_vector)
        return hot_vector
            
    def __init__(self, phase, duration, maxlen):
        X_current_stshiseq = []
        X_all_stshiseq = []
        Y_all = []
        
        if phase == "training":
            video_labels_path = video_labels_path_train
        else:
            video_labels_path = video_labels_path_test
        
        df = pd.read_csv(video_labels_path, names = ["A", "B", "C", "D", "E", "F","G","H","I","J","K"], usecols = ["A","C","J"], skiprows = 1, nrows = 500)
        
        
        #DEBUG step:
        #print(df)
        
        #collect all activity classes and context classes
        with open(activity_classes_path, 'r') as f:
            activity_labels = [line.split(None, 1)[0] for line in f] #should be loaded from activity classes file
        context_labels = df.C.unique()
        
        total_rows = len(df.index)
        
        for i, row in df.iterrows(): 
            
            
            #send video to Visual Attention Module                
            video_src = os.path.join(video_path,str(row["A"]) + ".mp4")
            if os.path.isfile(video_src):
                Y_current_context_vector = self.hot_encode("context", context_labels, np.array(str(row["C"]))) #create HOT context vector for df["C"] 
                Y_current_activity_vector = self.hot_encode("activity", activity_labels,str( row["J"])) #create HOT activity vector fot df["J"]
                print("\nRow: {} of {}; Currently iterrated {}% of rows".format(i, total_rows, (i + 1)/total_rows * 100))
                print("Video : " + video_src + " is being processed.")
                #clear the va_output folder
                empty_folder(stshi_path)
                X_current_stshiseq = []
                Visual_Attention_Module.process_scenes(video_src, duration)
            else:
                print("Video : " + video_src + " could not be opened.")
                continue
            
            #open the stshi_temp file
        
            for x_file in os.listdir(stshi_path):
                stshi_img = np.asarray(cv2.imread(os.path.join(stshi_path,x_file), 0))
                #load image as a flattened matrix
                if stshi_img is not None:
                    X_current_stshiseq.append(np.ravel(stshi_img))
                    
            X_current_stshiseq = np.asarray(pad_vec_sequences(X_current_stshiseq, maxlen))
           
            X_all_stshiseq.append(X_current_stshiseq)
            
            Y_current = np.concatenate((Y_current_activity_vector, Y_current_context_vector))
            Y_all.append(Y_current)
            
        self.X_all_stshiseq= X_all_stshiseq
        self.activity_labels = activity_labels
        self.context_labels = context_labels
        self.Y_all = Y_all
        
        

#TODO - Pad each stshi sequence to equal length matrices - upto maxlen
def pad_vec_sequences(sequence,maxlen=36):
    new_sequence = []
    dimensions = np.shape(sequence)
    new_dimensions = list(dimensions)
    new_dimensions[0] = maxlen
    new_dimensions = tuple(new_dimensions)
    if dimensions[0]== maxlen:
        new_sequence = sequence
    elif dimensions[0] < maxlen:
        #pad sequence
        new_sequence = np.zeros(new_dimensions)
        new_sequence[:dimensions[0]] = sequence
    else:
        new_sequence = sequence[:maxlen]
    return new_sequence

    '''
	for sequence in sequences:
		
		orig_len, vec_len = np.shape(sequence)
		if orig_len < maxlen:
			new = np.zeros((maxlen,vec_len))
			new[maxlen-orig_len:,:] = sequence
		else:
			#print(sequence)
			new = sequence[orig_len-maxlen:,:]
		new_sequences.append(new)
	new_sequences = np.array(new_sequences)
	#print(new_sequences.shape)
    '''
	