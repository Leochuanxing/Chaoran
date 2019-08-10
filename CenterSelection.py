'''
############THIS FILE IS TO COMPARE THE EFFICIENCIES OF TWO CENTER SELECTION METHODS
'''
import os
import pandas as pd
import numpy as np
import copy

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
##########################################################################################3
****************************************************************************************8
         DEFINE THE RADIAL BASIS FUNCTIONS
         
RBF: the radial basis function
'''
def Thin_Plate_Spline(d):
    if d == 0:
        return 0
    else:
        return d**2 * np.log(d)

def Gaussian(distance, radius):
    return np.exp(-distance**2/radius**2)
    
def Markov(distance, radius):
    return np.exp(-distance/radius)  

def Inverse_Multi_Quadric(distance,c, beta):
    return (distance**2 + c**2)**(-beta) 

def Design_matrix(distance_matrix, RBF = 'Gaussian'):
    
    if RBF == 'Gaussian':
        design_matrix = Gaussian(distance_matrix, radius = 1)
    elif RBF == 'Markov':
        design_matrix = Markov(distance_matrix, radius = 1)
    elif RBF == 'Thin_Plate_Spline':
        design_matrix = Thin_Plate_Spline(distance_matrix)
    elif RBF == 'Inverse_Multi_Quadric':
        design_matrix = Inverse_Multi_Quadric(distance_matrix, c = 0.5, beta=1)
        
    return design_matrix

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
    '''There should be at least 3 values for each attribute in order to use Lin_matrix'''
    def Lin_matrix(self):
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        frame_log_f_div_k = self.frame_log_f_div_k 
        
        A_matrix = np.zeros(distance_matrix.shape)
        for i in range(nrow):
            for j in range(i, nrow):
                A_matrix[i,j] = np.sum(frame_log_f_div_k.iloc[i]+frame_log_f_div_k.iloc[j])
                A_matrix[j,i] = A_matrix[i,j]  
        self.A_matrix = A_matrix

        for i in [1]:
            for j in [1]:
                equal_series = A_matrix[i,j]/(2*frame_log_f_div_k.iloc[i]) - 1
                if self.att.iloc[i].equals(self.att.iloc[j]):
                    distance_matrix[i,j] = np.sum(equal_series)
                else:
                    unequal_series = A_matrix[i,j]/(2*np.log((self.freq_frame.iloc[i]+self.freq_frame.iloc[j])/nrow,  dtype='f'))- 1
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
        denorminator_series = pd.DataFrame()
        
        for col in relative_freq.columns:
            column = relative_freq[col]
            s = 0
            for val in set(self.att[col]):
                s += 2 * np.log(1-column[self.att[col] == val].iloc[0])
            denorminator_series[col] = [s]
                
        for i in range(nrow):
            for j in range(i, nrow):
                equal_indicator = (self.att.iloc[i] == self.att.iloc[j])
                unequal_indicator =  1 - equal_indicator
                freq_product = self.freq_frame.iloc[i] * self.freq_frame.iloc[j]
                nrow_minus_freq_product = nrow_minus_freq.iloc[i] * nrow_minus_freq.iloc[j]
                numerator_series =np.log(freq_product / nrow_minus_freq_product, dtype = 'f')
                
                unequal_series = numerator_series / denorminator_series
                unequal_series /= ncol
                
                distance_matrix[i,j] = np.sum(equal_indicator + unequal_indicator * unequal_series, axis = 1)
                distance_matrix[j,i] = distance_matrix[i,j]
                
        self.burnaby_matrix = distance_matrix

    def Eskin_matrix(self):
        nrow, ncol = self.att.shape
        ni_frame = self.att.apply(set, axis= 0).apply(lambda x:len(x))
        unequal_series = 2/(ni_frame*ni_frame)
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
SS.IOF_matrix()
SS.OF_matrix()
SS.Burnaby_matrix()
SS.burnaby_matrix
SS.Eskin_matrix()
SS.eskin_matrix
'''
###################################################################################
'''
def Split_dist_matrix(distance_matrix, design_matrix, train_ind, test_ind):
    
    train_distance_matrix = distance_matrix[train_ind, :][:, train_ind]
    test_distance_matrix = distance_matrix[test_ind, :][:, train_ind]
    train_design_matrix = design_matrix[train_ind, :][:, train_ind]
    test_design_matrix = design_matrix[test_ind, :][:, train_ind]
    
    return train_design_matrix, test_design_matrix, train_distance_matrix, test_distance_matrix
distance_matrix = SS.hamming_matrix
design_matrix = Design_matrix(distance_matrix, RBF = 'Gaussian')
train_design_matrix, test_design_matrix, train_distance_matrix, test_distance_matrix = \
                Split_dist_matrix(distance_matrix, design_matrix, list(range(8)), [8,9])
'''****************************************************************************************
********************************************************************************************
All the above functions have been verified by  22:00 Aug.10th'''
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
def Design_matrix_coverage_by_nCenters(train_design_matrix, test_design_matrix, eliminated, nCenters):
    
    nrow, _ = train_design_matrix.shape
    to_be_del = eliminated[:nrow - nCenters]
    sub_train_dm = np.delete(train_design_matrix, to_be_del, axis=1)
    sub_test_dm = np.delete(test_design_matrix, to_be_del, axis =1)
        
    return sub_train_dm, sub_test_dm

'''
Design_matrix_coverage_by_radius: find centers with the cut off distance cutoff

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

def Design_matrix_coverage_by_radius(train_design_matrix, test_design_matrix, eliminated, radius, cutoff):
    
    nrow, _ = train_design_matrix.shape
    to_be_del = []
    for ind, v in enumerate(radius):
        if v <= cutoff:
            to_be_del.append(eliminated[ind])
            
    sub_train_dm = np.delete(train_design_matrix, to_be_del, axis=1)
    sub_test_dm = np.delete(test_design_matrix, to_be_del, axis =1)
    
    return sub_train_dm, sub_test_dm
'''We should return the centers which will be used in testing'''

eliminated, radius = CS_coverage_all(train_distance_matrix)
train_center_dm, test_center_dm = Design_matrix_coverage_by_nCenters(train_design_matrix,test_design_matrix, eliminated, 3)
train_center_dm.shape
test_center_dm.shape
train_center_dm_by_r, test_center_dm_by_r = Design_matrix_coverage_by_radius(train_design_matrix,test_design_matrix, eliminated,radius, cutoff = 0.7)

SS.cate.loc[SS.cate.label == 'y1'] = 0
SS.cate.loc[SS.cate.label == 'y2'] = 1
observed = SS.cate
observed_train = observed.iloc[list(range(8))]
observed_test = observed.iloc[[8,9]]
observed_test
observed_train
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
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradient from the first part of loss
    grad_logit = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_logit)
    grad_coefficients /= nrow
    # Calculate the gradient from the regularizatio part
    grad_coefficients += 2 * reg * coefficients
    
    # return the above results
    return loss, grad_coefficients
'''
Loss_Softmax: this function applies to the case when the output classes are more than 2
Input:
    design_matrix: as above
    labels: a matrix of dimension (m, k), where k is the number of classes. The label of
            each sample is a vector of dimenion (1, k), and the values are either 0 or 1, with
            1 indicate the correct category.
    coefficients: a matrix of dimension (n*k, 1). For the convenience of latter usage,  we don't use the shape(n, k).
                    When (n,k) is reshaped into (n*k,1), we stack column by column.
    reg: as above
Output:
    similar as above
'''
def Loss_Softmax(design_matrix, labels, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    # Reshape the coefficients
    coefficients = coefficients.reshape((-1, ncol)).T
    
    Wx = design_matrix.dot(coefficients)
    # Make sure the elements in Wx is not too big or too small
    Wx -= np.max(Wx, axis = 1, keepdims = True)
    # Calculate the probabilities
    exp = np.exp(Wx)
    prob = exp / np.sum(exp, axis = 1, keepdims = True)
    
    log_prob = np.log(prob)
    # Calculate  the loss
    loss = np.sum(- log_prob * labels)/nrow
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradients
    grad_Wx = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_Wx)
    grad_coefficients /= nrow
    
    grad_coefficients += 2 * reg * coefficients
    
    grad_coefficients = grad_coefficients.T.reshape((-1, 1))
    
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
    
    for i in range(n_iterations):
        loss, grad_coefficients = Loss(train_para)
        # update the coefficients
        train_para['coefficients'] -= step_size * grad_coefficients
        '''Do we print the loss'''
        if i % 100 == 0:
            print(round(loss, 6))
            
    return train_para


def Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-2):   
    
    nrow, _ = np.shape(train_para['coefficients'])
        
    '''Give the initial Hessian H. The dimension of H is 
    the same as the number of coefficients to be trained'''
    H = np.eye(nrow)

    # BFGS algorithm
    loss, grad_coeff = Loss(train_para)
    ternination_square = termination**2
    grad_square = ternination_square + 1
#    grad_square = (grad_coeff.T).dot(grad_coeff)
    while grad_square >= ternination_square:        
        p = - H.dot(grad_coeff)        
        # There should be both old and new coefficients in the train_para
        train_para['coefficients_old'] = train_para['coefficients']
        train_para['coefficients'] = p + train_para['coefficients_old']
        
        new_loss, new_grad_coeff = Loss(train_para)
        
        # Ramijo Back-tracking
        while new_loss > loss + c * (grad_coeff.T).dot(p):
            p *= rho
            train_para['coefficients'] = p + train_para['coefficients_old']            
            new_loss, new_grad_coeff = Loss(train_para)
        
        # update H
        s = p
        y = new_grad_coeff - grad_coeff
        r = (y.T).dot(s)
        I = np.eye(nrow)
#        I = np.eye(ncol+1)
        if r != 0:
            r = 1/r            
            H = (I - r*s.dot(y.T)).dot(H).dot(I - r*y.dot(s.T)) + r*s.dot(s.T)# Can be accelerated
        else:
            H = I
        # Update loss, grad_square and paramter
        loss = new_loss
        grad_coeff = new_grad_coeff
        grad_square = new_grad_coeff.T.dot(new_grad_coeff)
        
        # print some values to monitor the training process        
        print('loss  ', loss, '    ','grad_square   ', grad_square)
        
    return train_para, loss


soft_observed =np.array( [[1, 0],
       [0, 1],
       [0, 1],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0]])
soft_observed.reshape((-1, 1))

train_para = {}
train_para['design_matrix'] = train_center_dm
train_para['observed'] = soft_observed
train_para['reg'] = 1
train_para['coefficients'] = np.random.randn(train_center_dm.shape[1]*train_para['observed'].shape[1], 1)*2
train_para['loss_type'] = 'Softmax'

initial = copy.deepcopy(train_para)

train_para['step_size'] = 1e-4
train_para['n_iterations'] = 10000
train_para = Train_GD(train_para)
train_para['coefficients']
initial['coefficients']

train_para = copy.deepcopy(initial)
train_para, loss = Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-2)
train_para['coefficients']
initial['coefficients']
'''#################################################################################'''
'''
**************THE FOLOWING BLOCK IS TO SELECT CENTER BY THE ABSOLUTE VALUES OF THE COEFFICIENTS
*************The general idea is to generate a list of centers and the corresponding trained 
************** coefficients, which can be applied to the testig set.

One_step_reduce_centers: This function is to reduce the centers on basis of the coefficients after 
                one round of training
Input:
    train_para: as above
    testing_design_matrix:
        a matrix with the rows the testing samples and the columns the centers. 
       the testing design matrix and the design matrix in the train_para should have the columuns corresponding 
       set of centers.
'''
def One_step_reduce_centers(train_para, testing_design_matrix):

    nrow, ncol = train_para['design_matrix'].shape
#    ratio = 3000/len(centers)
    if ncol > 1500 :
        termination = 1.5*ncol
    elif ncol<= 1500 and ncol >1000:
        termination =  ncol
    elif ncol<= 1000 and ncol>500: 
        termination = 10
    elif ncol <= 500:
        termination = ncol/1000
#    termination =10* len(centers)        
    train_para, loss = Train_RBFN_BFGS(train_para, rho=0.85, c = 1e-3, termination=termination)   
    
    m =1
    '''m is the number of centers to be removed at each round of training'''
    for i in range(m):    
        coeff = train_para['coefficients'] 
        # if it is the case of softmax, we have to reshape the coeff
        coeff = coeff.reshape((-1, ncol)).T
        sabs_coeff = np.sum(np.abs(coeff), axis = 1, keepdims = True)
        # find the index of the coefficients with the smallest absolute value
        ind_min = np.argmin(np.abs(sabs_coeff))
        # remove the smallest
        one_less_coeff = np.delete(coeff, ind_min, axis=0)
        train_para['coefficients'] = one_less_coeff.T.reshape((-1,1))
        testing_design_matrix = np.delete(testing_design_matrix, ind_min, axis = 1)
        train_para['design_matrix'] = np.delete(train_para['design_matrix'], ind_min, axis = 1)

    return testing_design_matrix
'''#######################################################################################'''
'''
*************Strategy:
                      1, Calculate a big distance matrix, with all the training and testing samples
******************** included. Then split this matrix into a square matrix with the rows and columns 
********************  the training samples, and a matrix with all the rows testing samples and all the columns
********************  the training samples.
                      2, Calculate the design matrices for the above two distance matrices
                      3, Select centers and sub select the corresponding designmatrics from step 2
'''



