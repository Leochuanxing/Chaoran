'''
############THIS FILE IS TO COMPARE THE EFFICIENCIES OF TWO CENTER SELECTION METHODS
'''
import os
import pandas as pd
import numpy as np

'''################################################################################'''

# Load the data of breast cancer and do the data wrangling
def Wrangling_BC():
    os.chdir('/home/leo/Documents/Project_SelectCenters/DATA/BreastCancer')
    data = pd.read_csv('breast-cancer-wisconsin.data')

    col_name = data.columns

    complete = data[:][data[col_name[6]] != '?']
    complete[col_name[6]] = complete[col_name[6]].astype(np.int64)
    
    attributes = col_name[1:10]
    categories = col_name[-1]
    
    df_att = complete[attributes]
    
    return df_att, attributes, categories

# independent test
#col = df_att[attributes[0]]
def Frequency_distinct(col):
    distinct_values = list(set(col))
    df_element = pd.DataFrame({'d_values':[], 'freq':[]})
    for i in distinct_values:
        df_element = df_element.append({'d_values':i, 'freq':sum(col == i)},\
                                        ignore_index = True)
    
    freq_col = col.apply(lambda x:np.int(df_element[x==df_element['d_values']]['freq']))
        
    return freq_col

#freq_df = df_att.apply(Frequency_distinct, axis = 0)
#freq_array_log = np.log(freq_df)
#freq_array_log.iloc[:5, :5]
#head = freq_array_log.head()/ncol
#
#np.sum(freq_array_log.iloc[1]*freq_array_log.iloc[2])/ncol
#
#freq_df.head()
#df_att.head()
#np.sum(df_att['1.1'] == 8)
#df_att.iloc[1].equals(df_att.iloc[3])
'''
Generate a small data set to test the following functions
'''
def Small_sample(size = 10):
    att1 = ['A', 'B']
    att2 = [1,2,3]
    att3 = ['D', 'E','F']
    label = ['y1', 'y2']
    # Randomly assign the values
    att1_v = np.random.choice(att1, size)
    att2_v = np.random.choice(att2, size)
    att3_v = np.random.choice(att3, size)
    label_v = np.random.choice(label, size)
    # Combine them into a data frame
    sample_data = pd.DataFrame({'att1':att1_v, 'att2':att2_v, 'att3': att3_v, 'label':label_v})
    return sample_data

small_sample = Small_sample(size=10)
small_sample

'''
#######################################################################################
Preparation is for data wrangling and calculate the distance matrix according to the
selected distance
*******************************************************************************
Inputs:
    data_frame: The training data frame
    attributes: a list gives the names of the attributes used as explatary variables, which are consistant
                with  the names of the columns.
    categories: a list contains the names of the columns of the labels.
Outputs:
    different types of distance matrices
'''
class Distance_martix:
    def __init__(self, data_frame, attributes, categories):
        self.att = data_frame[attributes]
        self.cate = data_frame[categories]
        nrow, ncol = self.att.shape
    def Hamming_matrix(self):
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                distance_matrix[i,j] = np.sum(self.att.iloc[i] != self.att.iloc[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        self.hamming_matrix = distance_matrix/ncol
        
    def IOF_matrix(self):
        nrow, ncol = self.att.shape
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
        nrow, ncol = self.att.shape
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
        nrow, ncol = self.att.shape
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

    def Burnaby_matrix(self):
        nrow, ncol = self.att.shape
        # Make sure there are at least two values for each attribute
        distance_matrix = np.zeros((nrow, nrow))
        nrow_minus_freq = nrow - self.freq_frame
        relative_freq = self.freq_frame / nrow
        for i in range(nrow):
            for j in range(i, nrow):
                equal_indicator = (self.att.iloc[i] == self.att.iloc[j])
                unequal_indicator =  1 - equal_indicator
                freq_product = self.freq_frame.iloc[i] * self.freq_frame.iloc[j]
                nrow_minus_freq_product = nrow_minus_freq.iloc[i] * nrow_minus_freq.iloc[j]
                numerator_series =np.log(freq_product / nrow_minus_freq_product, dtype = 'f')
                
                denorminator_series = 2 * np.sum(np.log(1-relative_freq, dtype = 'f'), axis = 0)
                
                unequal_series = numerator_series / denorminator_series
                unequal_series /= nrow
                
                distance_matrix[i,j] = np.sum(equal_indicator) + np.sum(unequal_indicator * unequal_series)
                distance_matrix[j,i] = distance_matrix[i,j]
                
        self.burnaby_matrix = distance_matrix

    def Eskin_matrix(self):
        nrow, ncol = self.att.shape
        ni_frame = self.att.apply(set, axis= 0).apply(lambda x:len(x))
        unequal_series = 2/(ni_frame.ni_frame)
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i,nrow):
                unequal_indicator = (self.att.iloc[i] != self.att.iloc[j])
                distance_matrix[i,j] = np.sum(unequal_indicator * unequal_series)
                distance_matrix[j,i] = distance_matrix[i,j]
        self.eskin_matrix = distance_matrix / ncol
    # Define the distances
attributes = ['att1', 'att2', 'att3']
categories = ['label']
SS = Distance_martix(small_sample, attributes, categories)
SS.Hamming_matrix()
SS.hamming_matrix

'''#################################################################################'''
'''
**************************************************************************************
CS_coverage_all: a function to find the oder of the centers to be eliminated and the corresponding 
                cutoff distances
Inputs:
    distance_matrix
    
Outputs:
    eliminated: a list, gives the indices, corresponding to the rows, to be removed, so that the leftovers
                can be used as centers
    radius: a list, gives the cutoff distances that corresponding to the elements in the eliminated.
    
For example:
    eliminated = [1,2,3], radius = [1, 1, 2]
    Means: to remove center corresponding to row 1, the cutoff distance need to be 1.
           to remove center corresponding to row 2, the cutoff distance need to be 1.
           to remove center corresponding to row 3, the cutoff distance need to be 2.
'''
def CS_coverage_all(distance_matrix):
    l = list(np.ndenumerate(distance_matrix))
    # get rid of the diagonal elements
    l_sub = [x for x in l if x[0][0] != x[0][1]]
    # arrange the distance in the increasing order
    l_sub.sort(key = lambda x:x[1])

    eliminated = []; radius = []
    while l_sub !=[]:
        eliminated.append(l_sub[0][0][1])
        radius.append(l_sub[0][1])
        # update l_sub
        l_sub = [x for x in l_sub if x[0][0] != l_sub[0][0][1] and x[0][1] != l_sub[0][0][1]]
        
    return eliminated, radius

'''
Design_matrix_coverage_by_nCenters: find centers with the size equal to nCenters

Inputs:
        distance_matrix: as above
        eliminated: as above
        nCenters: integer, the number of centers
Outputs:
    design_matrix: a submatrix of the distance matrix, with the rows corresponding to the
                    samples and the columns corresponding to the selected centers
'''
def Design_matrix_coverage_by_nCenters(distance_matrix, eliminated, nCenters):
    nrow, _ = distance_matrix.shape
    to_be_del = eliminated[:nrow - nCenters]
    design_matrix = np.delete(distance_matrix, to_be_del, axis=1)
    return design_matrix

'''
Design_matrix_coverage_by_radius: find centers with the size equal to nCenters

Inputs:
        distance_matrix: as above
        eliminated: as above
        radius: as above
        cutoff: float, gives the cutoff distance. For each center, any sample with 
            the distance from it no larger than cutoff will not be selected as a center.
Outputs:
    design_matrix: a submatrix of the distance matrix, with the rows corresponding to the
                    samples and the columns corresponding to the selected centers
'''

def Design_matrix_coverage_by_radius(distance_matrix, eliminated, radius, cutoff):
    nrow, _ = distance_matrix.shape
    to_be_del = []
    for ind, v in enumerate(radius):
        if v <= cutoff:
            to_be_del.append(eliminated[ind])
    design_matrix = np.delete(distance_matrix, to_be_del, axis=1)
    return design_matrix

eliminated, radius = CS_coverage_all(SS.hamming_matrix)
design_matrix = Design_matrix_coverage_by_nCenters(SS.hamming_matrix, eliminated, 3)
design_matrix_by_r = Design_matrix_coverage_by_radius(SS.hamming_matrix, eliminated, radius, cutoff = 0.7)
#design_matrix.shape
#design_matrix_by_r.shape
'''#################################################################################'''
'''
This function should return the loss and the gradient
'''
def Loss_Softmax(design_matrix, labels):
    pass
def Loss_SumSquares(design_matrix, observed):
    pass
'''#################################################################################'''
def One_step_train():
    pass
'''#################################################################################'''
def CS_Coeff():
    pass






