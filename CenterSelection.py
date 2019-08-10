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


'''
##########################################################################################3
****************************************************************************************8
         DEFINE THE RADIAL BASIS FUNCTIONS
         
RBF: the radial basis function
'''
def Thin_Plate_Spline(d):
    d += 0.001 # make sure the returned values is not overflown
    return d**2 * np.log(d)

def Gaussian(distance, radius):
    return np.exp(-distance**2/radius**2)
    
def Markov(distance, radius):
    return np.exp(-distance/radius)  

def Inverse_Multi_Quadric(distance,c, beta):
    return (distance**2 + c**2)**(-beta) 

design_mat = Gaussian(SS.hamming_matrix, 1)
design_mat[:5, :5]
SS.hamming_matrix[:5, :5]
np.exp(-0.3333333**2)
'''#################################################################################
            CALCULATE THE DESIGN MATRIX FOR CENTERS SELECTED BY COVERAGE METHOD

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
def Design_matrix_coverage_by_nCenters(distance_matrix, eliminated, nCenters, RBF = 'Gaussian'):
    nrow, _ = distance_matrix.shape
    to_be_del = eliminated[:nrow - nCenters]
    sample_center_dist_mat = np.delete(distance_matrix, to_be_del, axis=1)
    
    if RBF == 'Gaussian':
        design_matrix = Gaussian(sample_center_dist_mat, radius = 1)
    elif RBF == 'Markov':
        design_matrix = Markov(sample_center_dist_mat, radius = 1)
    elif RBF == 'Thin_Plate_Spline':
        design_matrix = Thin_Plate_Spline(sample_center_dist_mat)
    elif RBF == 'Inverse_Multi_Quadric':
        design_matrix = Inverse_Multi_Quadric(sample_center_dist_mat, c = 0.5, beta=1)
        
    return design_matrix, sample_center_dist_mat

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

def Design_matrix_coverage_by_radius(distance_matrix, eliminated, radius, cutoff, RBF = 'Gaussian'):
    nrow, _ = distance_matrix.shape
    to_be_del = []
    for ind, v in enumerate(radius):
        if v <= cutoff:
            to_be_del.append(eliminated[ind])
    sample_center_dist_mat = np.delete(distance_matrix, to_be_del, axis=1)
    
    if RBF == 'Gaussian':
        design_matrix = Gaussian(sample_center_dist_mat, radius = 1)
    elif RBF == 'Markov':
        design_matrix = Markov(sample_center_dist_mat, radius = 1)
    elif RBF == 'Thin_Plate_Spline':
        design_matrix = Thin_Plate_Spline(sample_center_dist_mat)
    elif RBF == 'Inverse_Multi_Quadric':
        design_matrix = Inverse_Multi_Quadric(sample_center_dist_mat, c = 0.5, beta=1)
    return design_matrix, sample_center_dist_mat

eliminated, radius = CS_coverage_all(SS.hamming_matrix)
design_matrix, _ = Design_matrix_coverage_by_nCenters(SS.hamming_matrix, eliminated, 3, RBF = 'Gaussian')
design_matrix_by_r,_ = Design_matrix_coverage_by_radius(SS.hamming_matrix, eliminated, radius, cutoff = 0.7, RBF = 'Gaussian')
n_row, n_col = design_matrix.shape
design_matrix_by_r.shape
design_matrix
design_matrix_by_r
SS.cate
SS.cate.loc[SS.cate.label == 'y1'] = 0
SS.cate.loc[SS.cate.label == 'y2'] = 1
coefficients = 0.1 * np.random.randn(n_col, 2)
coefficients
logit = design_matrix.dot(coefficients)
logit
prob = 1/(1+np.exp(-logit))
loss = np.average(- np.log(prob) * SS.cate - (1 - SS.cate) * np.log(1 - prob))
logit -= np.max(logit, axis = 1, keepdims = True) 
logit
'''#################################################################################
****************************************************************************************
                CALCULATE THE LOSS AND THE GRADIENTS

Suppose: design_matrix is of dimension (m,n)                

Loss_Sigmoid: Use the sigmoid as the loss function
Inputs:
    design_matrix: as above
    labels: an array corresponding to the rows of the design_matrix,
            with values of either 0 or 1. labels.shape = (m, 1)
    coefficients: an array contains the coeffients, coefficients.shape = (n, 1)
    reg: a float, the coefficient of regularization
Outputs:
    loss: float
    grad_coefficients: an array containing all the gradients corresopnding to the coefficients
'''
def Loss_Sigmoid(design_matrix, labels, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    
    logit = design_matrix.dot(coefficients)
    prob = 1/(1+np.exp(-logit))
    loss = np.average(- np.log(prob) * labels - (1 - labels) * np.log(1 - prob))
    # plus the regularization
    loss += reg * coefficients * coefficients
    
    # Calculate the gradient from the first part of loss
    grad_logit = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_logit)
    grad_coefficients /= nrow
    # Calculate the gradient from the regularizatio part
    grad_coefficients += 0.5 * reg * coefficients
    
    # return the above results
    return loss, grad_coefficients
'''
Loss_Softmax: this function applies to the case when the output classes are more than 2
Input:
    design_matrix: as above
    labels: a matrix of dimension (m, k), where k is the number of classes. The label of
            each sample is a vector of dimenion (1, k), and the values are either 0 or 1, with
            1 indicate the correct category.
    coefficients: a matrix of dimension (n,k)
    reg: as above
Output:
    similar as above
'''
def Loss_Softmax(design_matrix, labels, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    
    Wx = design_matrix.dot(coefficients)
    # Make sure the elements in Wx is not too big or too small
    Wx -= np.max(Wx, axis = 1, keepdims = True)
    # Calculate the probabilities
    exp = np.exp(Wx)
    prob = exp / np.sum(exp, axis = 1, keepdims = True)
    # Calculate  the loss
    loss = np.sum(prob * labels)/nrow
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradients
    grad_Wx = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_Wx)
    grad_coefficients /= nrow
    
    grad_coefficients += 2 * coefficients
    
    return loss, grad_coefficients
'''
Loss_SumSquares: the loss is measured by the sum of the sequares of the difference 
                between the predicted values and the observed values
Inputs:
    observed: an array of shape (m,1). with each element a float.
    design_matrix, coefficients and reg are the same as above
Outputs:
    the same as above
'''    
def Loss_SumSquares(design_matrix, observed, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    
    # Calculate the loss
    pred = design_matrix.dot(coefficients)
    loss = np.average((pred - observed) * (pred - observed))
    
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradient
    
    grad_coefficients = (design_matrix.T).dot(2 * (pred - observed))
    grad_coefficients /= nrow
    grad_coefficients += 2 * reg * coefficients
    
    return loss, grad_coefficients
'''
Integrate the above functions into one function for convenient usage.
Input:
    train_para: a dictionary, contains all the needed values to train a model.
'''
def Loss(train_para):
    design_matrix = train_para['design_matrix'] 
    observed = train_para['observed']
    reg = train_para['reg']
    coefficients = train_para['coefficients']
    loss_type = train_para['loss_type']
    if loss_type == 'Sigmoid':
        loss, grad_coefficients = Loss_Sigmoid(design_matrix, observed, coefficients, reg)
    elif loss_type == 'Softmax':
        loss, grad_coefficients = Loss_Softmax(design_matrix, observed, coefficients, reg)
    elif loss_type == 'SumSquares':
        loss, grad_coefficients = Loss_SumSquares(design_matrix, observed, coefficients, reg)
    return loss, grad_coefficients
'''#################################################################################'''
'''
Train_GD: train the model by using gradient descent
Inputs:
    gd_train_para: a dictionary, contains
        reg: coefficients of regularization
        setp_size: a float
        loss_type: string, gives types of different loss functions
        design_matrix:
        observed: observed values, the format of which decides the type of loss functions
Outputs:  
    coefficients: a matrix of shape (design_matrix.shape[0], observed.shape[1])
                    the values of the coefficients after n_iterations training      
'''
def Train_GD(train_para):
    # Take out the parameters
#    design_matrix = train_para['design_matrix'] 
#    observed = train_para['observed']
#    reg = train_para['reg']
    step_size = train_para['step_size']
#    loss_type = gd_train_para['loss_type']
    n_iterations = train_para['n_iterations']
    train_para['coefficients'] = coefficients
    
    
    for i in range(n_iterations):
        loss, grad_coefficients = Loss(train_para)
        # update the coefficients
        train_para['coefficients'] -= step_size * grad_coefficients
        '''Do we print the loss'''
        print(round(loss, 6))
            
    return train_para
 

def Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-2):
    
    nrow, ncol = np.shape(train_para['design_matrix'])
        
    '''Give the initial Hessian H. The dimension of H is 
    the same as the number of coefficients to be trained'''
    H = np.eye(ncol)

    # BFGS algorithm
    loss, grad_coeff = Loss(train_para)
    ternination_square = termination**2
    grad_square = ternination_square + 1
#    grad_square = (grad_coeff.T).dot(grad_coeff)
    while grad_square >= ternination_square:        
        p = - H.dot(grad_coeff)        
        # There should be both old and new coefficients in the train_para
        train_para['coefficients_old'] = train_para['coefficients']
        train_para['coefficients'] += p
        
        new_loss, new_grad_coeff = Loss(train_para)
        
        # Ramijo Back-tracking
        while new_loss > loss + c * (grad_coeff.T).dot(p):
            p *= rho
            train_para['coefficients'] = p + ['coefficients_old']            
            new_loss, new_grad_coeff = Loss(train_para)
        
        # update H
        s = p
        y = new_grad_coeff - grad_coeff
        r = (y.T).dot(s)
        I = np.eye(ncol)
#        I = np.eye(ncol+1)
        if r != 0:
            r = 1/r            
            H = (I - r*s.dot(y.T)).dot(H).dot(I - r*y.dot(s.T)) + r*s.dot(s.T)# Can be accelerate
        else:
            H = I
        # Update loss, grad_square and paramter
        loss = new_loss
        grad_coeff = new_grad_coeff
        grad_square = (new_grad_coeff.T).dot(new_grad_coeff)
        
        # print some values to monitor the training process        
        print('loss  ', loss, '    ','grad_square   ', grad_square)
        
    return grad_coeff, loss
'''#################################################################################'''
def CS_Coeff():
    pass





