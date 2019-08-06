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
col = df_att[attributes[0]]



def Frequency_distinct(col):
    distinct_values = list(set(col))
    df_element = pd.DataFrame({'d_values':[], 'freq':[]})
    for i in distinct_values:
        df_element = df_element.append({'d_values':i, 'freq':sum(col == i)},\
                                        ignore_index = True)
    
    freq_col = col.apply(lambda x:np.int(df_element[x==df_element['d_values']]['freq']))
        
    return freq_col

freq_df = df_att.apply(Frequency_distinct, axis = 0)
np.log(freq_df)
df_att.iloc[1].equals(df_att.iloc[3])
df_att.iloc[1]
df_att.iloc[3]


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
        
    def IOF_matrix(self):
        freq_frame = self.att.apply(Frequency_distinct, axis = 0)
        freq_array_log = np.log(freq_frame).values
        self.freq_frame = freq_frame
        
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                if df_att.iloc[i].equals(df_att.iloc[j]):
                    distance_matrix[i,j] = 0
                else:
                    distance_matrix[i,j] = np.sum(freq_array_log[i]*freq_array_log[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        self.iof_matrix = distance_matrix/ncol

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

