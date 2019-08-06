'''
############THIS FILE IS TO COMPARE THE EFFICIENCIES OF TWO CENTER SELECTION METHODS
'''
import os
import pandas as pd
import numpy as np


# Load the data of breast cancer and do the data wrangling
os.chdir('/home/leo/Documents/Project_SelectCenters/DATA/BreastCancer')
data = pd.read_csv('breast-cancer-wisconsin.data')
data.dtypes
col_name = data.columns
data[data[col_name[2]] == '?']
missing = data[:][data[col_name[6]] == '?']
missing.shape
complete = data[:][data[col_name[6]] != '?']
complete[col_name[6]] = complete[col_name[6]].astype(np.int64)

attributes = col_name[1:10]
categories = col_name[-1]

df_att = complete[attributes]
nrow, ncol = df_att.shape
distance_matrix = np.zeros((nrow, nrow))
df_att.head()
complete.head()


distance_matrix = distance_matrix/ncol

'''
Preparation is for data wrangling and calculate the distance matrix according to the
selected distance
'''
class Distance_martix:
    def __init__(self, data_frame, attributes, categories):
        self.att = data_frame[attributes]
        self.cate = data_frame[categories]
    def Hamming_matrix(self):
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                distance_matrix[i,j] = np.sum(df_att.iloc[i] != df_att.iloc[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        self.hamming_matrix = distance_matrix/ncol
        
    def IOF_matrix():
        pass
    def OF_matrix():
        pass
    def Lin_matrix():
        pass
    def Burnaby_matrix():
        pass
    def Eskin_matrix():
        pass
    # Define the distances
BC = Distance_martix(complete, attributes, categories)
BC.Hamming_matrix()

