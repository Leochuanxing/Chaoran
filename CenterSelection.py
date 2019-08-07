'''
############THIS FILE IS TO COMPARE THE EFFICIENCIES OF TWO CENTER SELECTION METHODS
'''
import os
import pandas as pd
import numpy as np
import math


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

# independent test
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
freq_array_log = np.log(freq_df)
freq_array_log.iloc[:5, :5]
head = freq_array_log.head()/ncol

np.sum(freq_array_log.iloc[1]*freq_array_log.iloc[2])/ncol

freq_df.head()
df_att.head()
np.sum(df_att['1.1'] == 8)
df_att.iloc[1].equals(df_att.iloc[3])



'''
Preparation is for data wrangling and calculate the distance matrix according to the
selected distance
'''
class Distance_martix:
    def __init__(self, data_frame, attributes, categories):
        self.att = data_frame[attributes]
        self.cate = data_frame[categories]
        nrow, ncol = self.att.shape
    def Hamming_matrix(self):
        
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                distance_matrix[i,j] = np.sum(df_att.iloc[i] != df_att.iloc[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        self.hamming_matrix = distance_matrix/ncol
        
    def IOF_matrix(self):
        freq_frame = self.att.apply(Frequency_distinct, axis = 0)
        freq_frame_log = np.log(freq_frame, dtype = 'f')
        self.freq_frame = freq_frame
        self.freq_frame_log = freq_frame_log
        
#        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                indicator =1- self.att.iloc[i].eq(self.att.iloc[j])
                product = np.float32(freq_frame_log.iloc[i]*freq_frame_log.iloc[j])
                distance_matrix[i,j] = np.sum(indicator * product)
                distance_matrix[j,i] = distance_matrix[i,j]
        self.iof_matrix = np.float32(distance_matrix/ncol)

    def OF_matrix(self):
        logk = np.log(nrow, dtype='f')
        distance_matrix = np.zeros((nrow, nrow))
        frame_log_f_div_k = self.freq_frame_log - logk
        self.frame_log_f_div_k  = frame_log_f_div_k 
        for i in range(nrow):
            for j in range(i, nrow):
                indicator =1- self.att.iloc[i].eq(self.att.iloc[j])
                product = frame_log_f_div_k.iloc[i]*frame_log_f_div_k.iloc[j]
                distance_matrix[i,j] = np.sum(indicator * product)
                distance_matrix[j,i] = distance_matrix[i,j]
                self.of_matrix = np.float32(distance_matrix/ncol)

    def Lin_matrix(self):
        distance_matrix = np.zeros((nrow, nrow))
        frame_log_f_div_k = self.frame_log_f_div_k 
        
        A_matrix = np.zeros(distance_matrix.shape)
        for i in range(nrow):
            for j in range(i, nrow):
                A_matrix[i,j] = np.sum(frame_log_f_div_k.iloc[i]+frame_log_f_div_k.iloc[j])
                A_matrix[j,i] = A_matrix[i,j]                

        for i in range(nrow):
            for j in range(i, nrow):
                equal_series = A_matrix[i,j]/(2*frame_log_f_div_k.iloc[i]) - 1
                unequal_series = A_matrix[i,j]/(2*np.log((self.freq_frame.iloc[i]+self.freq_frame.iloc[j])/nrow,  dtype='f')) - 1
                equal_indicator = self.att.iloc[i].eq(self.att.iloc[j])
                unequal_indicator = 1 - equal_indicator
                distance_matrix[i,j] = np.sum(equal_indicator*equal_series + unequal_indicator * unequal_series)
                distance_matrix[j,i] = distance_matrix[i,j]
        self.lin_matrix = np.float32(distance_matrix/ncol)

    def Burnaby_matrix():
        pass
    def Eskin_matrix():
        pass
    # Define the distances
BC = Distance_martix(complete, attributes, categories)
BC.Hamming_matrix()
BC.IOF_matrix()
BC.iof_matrix[:5, :5]
BC.iof_matrix.dtype
BC.OF_matrix()
BC.of_matrix[:5, :5]
BC.Lin_matrix()
BC.lin_matrix[:5, :5]
a =  np.array([[1,2,3], [4,5,6]])
np.log(a)
r = np.log(a, dtype='f')
type(r)
type(np.log(5, dtype='f'))
