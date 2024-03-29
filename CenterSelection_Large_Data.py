'''This file is to deal with large data set. It should be able to process small data set'''
import os
import pandas as pd
import numpy as np
import copy
import math
import time
'''################################################################################'''
'''
Generate a small data set to test the following functions
'''
def Small_sample(size = 10):
    att1 = ['A', 'B', 'C']
    att2 = [1,2,3, 4]
    att3 = ['D', 'E','F', 'G']
    label = ['y1', 'y2']
    # Randomly assign the values
    att1_v = np.random.choice(att1, size)
    att2_v = np.random.choice(att2, size)
    att3_v = np.random.choice(att3, size)
    label_v = np.random.choice(label, size)
    # Combine them into a data frame
    sample_data = pd.DataFrame({'att1':att1_v, 'att2':att2_v, 'att3': att3_v, 'label':label_v})
    return sample_data
#sample_data = Small_sample(size = 20)
#sample_data
'''
##########################################################################################3
****************************************************************************************8
         DEFINE THE RADIAL BASIS FUNCTIONS
         
RBF: the radial basis function
'''
def Thin_Plate_Spline(d):
    design_matrix = np.zeros(d.shape)
    nrow, ncol = d.shape
    for i in range(nrow):
        for j in range(ncol):
            dist = d[i,j]
            if dist != 0:
                design_matrix[i,j] = dist**2 * np.log(dist)
            else:
                design_matrix[i,j] = 0
            
    return design_matrix

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
'''
def Frequency_distinct(df_train):
    
    dict_freq_dict = {}; dict_log_freq_dict = {}
    for att in df_train.columns:
        col = df_train.loc[:, att]        
        distinct_values = list(set(col))
        freq_dict = {}; log_freq_dict = {}
        for v in distinct_values:
            freq_dict[v] = sum(col == v)
            log_freq_dict[v] = np.log(freq_dict[v], dtype = np.float32)
        dict_freq_dict[att] = copy.deepcopy(freq_dict)
        dict_log_freq_dict[att] = copy.deepcopy(log_freq_dict)
    return dict_freq_dict, dict_log_freq_dict
'''
*******************************************************************************
'''

'''
Inputs:
    data_frame: The training data frame
    attributes: a list gives the names of the attributes used as explatary variables, which are consistant
                with  the names of the columns.
    categories: a list contains the names of the columns of the labels.
Outputs:
    different types of distance matrices
'''



def Hamming_matrix(df_train, df_cc_right, equal):
    ncol, _ = df_cc_right.shape
    nrow, natt = df_train.shape
    distance_matrix = np.zeros((nrow, ncol))
    if equal:
        for i in range(nrow):
            for j in range(i, ncol):
                distance_matrix[i,j] = np.sum(df_train.iloc[i] != df_cc_right.iloc[j])
                distance_matrix[j,i] = distance_matrix[i,j]
    else:
        for i in range(nrow):
            for j in range(ncol):
                distance_matrix[i,j] = np.sum(df_train.iloc[i] != df_cc_right.iloc[j])
    distance_matrix /= natt       
    return distance_matrix



def IOF_matrix(df_train, df_cc_right, dict_element_dict, dict_log_freq_dict, equal):
    
    n_cc_right, _ = df_cc_right.shape
    nrow, natt = df_train.shape
    att_list = df_train.columns     
    
    df_freq_train_log = pd.DataFrame(0, index= list(range(nrow)), columns= df_train.columns)
    df_freq_cc_right_log = pd.DataFrame(0, index = list(range(n_cc_right)), columns=df_cc_right.columns)
    
    for att in att_list:
        log_freq_dict = dict_log_freq_dict[att]
        for i in range(nrow):
            df_freq_train_log.loc[i, att] = log_freq_dict[df_train[att][i]]
        for j in range(n_cc_right):
            df_freq_cc_right_log.loc[j, att] = log_freq_dict[df_cc_right[att][j]]
    # Calculate iof distance matrix
    iof_dist = np.zeros((nrow, n_cc_right))
#    of_dist = np.zeros_like(iof_dist)
#    logk = np.log(nrow, dtype=np.float32)
    if not equal:
        for i in range(nrow):
            for j in range(n_cc_right):
                indicator =1 - df_train.iloc[i].eq(df_cc_right.iloc[j])          
                product_iof = np.float32(df_freq_train_log.iloc[i]*df_freq_cc_right_log.iloc[j])
                iof_dist[i,j] = np.sum(indicator * product_iof)
    else:
        for i in range(nrow):
            for j in range(i, n_cc_right):
                indicator =1 - df_train.iloc[i].eq(df_cc_right.iloc[j])          
                product_iof = np.float32(df_freq_train_log.iloc[i]*df_freq_cc_right_log.iloc[j])
                iof_dist[i,j] = np.sum(indicator * product_iof)
                iof_dist[j,i] = iof_dist[i,j]
#            product_of = np.float32((df_freq_train_log.iloc[i]-logk)*(df_freq_cc_right_log.iloc[j]-logk))
#            of_dist[i,j] = np.sum(indicator * product_of)
    iof_dist /= natt
#    of_dist /= natt

    return iof_dist

def OF_matrix(df_train, df_cc_right, dict_element_dict, dict_log_freq_dict, equal):
    
    n_cc_right, _ = df_cc_right.shape
    nrow, natt = df_train.shape
    att_list = df_train.columns     
    
    df_freq_train_log = pd.DataFrame(0, index= list(range(nrow)), columns= df_train.columns)
    df_freq_cc_right_log = pd.DataFrame(0, index = list(range(n_cc_right)), columns=df_cc_right.columns)
    
    for att in att_list:
        log_freq_dict = dict_log_freq_dict[att]
        for i in range(nrow):
            df_freq_train_log.loc[i, att] = log_freq_dict[df_train[att][i]]
        for j in range(n_cc_right):
            df_freq_cc_right_log.loc[j, att] = log_freq_dict[df_cc_right[att][j]]
    # Calculate iof distance matrix
#    iof_dist = np.zeros((nrow, n_cc_right))
    of_dist = np.zeros((nrow, n_cc_right))
    logk = np.log(nrow, dtype=np.float32)
    if not equal:
        for i in range(nrow):
            for j in range(n_cc_right):
                indicator =1 - df_train.iloc[i].eq(df_cc_right.iloc[j])          
    #            product_iof = np.float32(df_freq_train_log.iloc[i]*df_freq_cc_right_log.iloc[j])
    #            iof_dist[i,j] = np.sum(indicator * product_iof)
                
                product_of = np.float32((df_freq_train_log.iloc[i]-logk)*(df_freq_cc_right_log.iloc[j]-logk))
                of_dist[i,j] = np.sum(indicator * product_of)
    else:
        for i in range(nrow):
            for j in range(i, n_cc_right):
                indicator =1 - df_train.iloc[i].eq(df_cc_right.iloc[j])          
    #            product_iof = np.float32(df_freq_train_log.iloc[i]*df_freq_cc_right_log.iloc[j])
    #            iof_dist[i,j] = np.sum(indicator * product_iof)
                
                product_of = np.float32((df_freq_train_log.iloc[i]-logk)*(df_freq_cc_right_log.iloc[j]-logk))
                of_dist[i,j] = np.sum(indicator * product_of)
                of_dist[j,i] = of_dist[i,j]
#    iof_dist /= natt
    of_dist /= natt

    return  of_dist

def Distance_matrix(row, column, distance_type, dict_element_dict, dict_log_freq_dict, equal = False):
    # Calculate the distance matrix
    if distance_type == 'Hamming':
        dist = Hamming_matrix(row, column, equal)
    elif distance_type == 'IOF':
        dist = IOF_matrix(row, column,\
                               dict_element_dict, dict_log_freq_dict, equal)
    elif distance_type == 'OF':
        dist = OF_matrix(row, column,\
                               dict_element_dict, dict_log_freq_dict, equal)      
    return dist
    

#    '''There should be at least 3 values for each attribute in order to use Lin_matrix'''
#    def Lin_matrix(self):
#        nrow, ncol = self.att.shape
#        distance_matrix = np.zeros((nrow, nrow))
#        frame_log_f_div_k = self.frame_log_f_div_k 
#        
#        A_matrix = np.zeros(distance_matrix.shape)
#        for i in range(nrow):
#            for j in range(i, nrow):
#                A_matrix[i,j] = np.sum(frame_log_f_div_k.iloc[i]+frame_log_f_div_k.iloc[j])
#                A_matrix[j,i] = A_matrix[i,j]  
#        self.A_matrix = A_matrix
#
#        for i in [1]:
#            for j in [1]:
#                equal_series = A_matrix[i,j]/(2*frame_log_f_div_k.iloc[i]) - 1
#                if self.att.iloc[i].equals(self.att.iloc[j]):
#                    distance_matrix[i,j] = np.sum(equal_series)
#                else:
#                    unequal_series = A_matrix[i,j]/(2*np.log((self.freq_frame.iloc[i]+self.freq_frame.iloc[j])/nrow,  dtype=np.float32))- 1
#                    equal_indicator = self.att.iloc[i].eq(self.att.iloc[j])
#                    unequal_indicator = 1 - equal_indicator   
#                    distance_matrix[i,j] = np.sum(equal_indicator*equal_series + unequal_indicator * unequal_series)
#
#                distance_matrix[j,i] = distance_matrix[i,j]
#                
#        self.lin_matrix = np.float32(distance_matrix/ncol)
#
#    def Burnaby_matrix(self):
#        nrow, ncol = self.att.shape
#        # Make sure there are at least two values for each attribute
#        distance_matrix = np.zeros((nrow, nrow))
#        nrow_minus_freq = nrow - self.freq_frame
#        relative_freq = self.freq_frame / nrow
#        denorminator_series = pd.DataFrame()
#        
#        for col in relative_freq.columns:
#            column = relative_freq[col]
#            s = 0
#            for val in set(self.att[col]):
#                s += 2 * np.log(1-column[self.att[col] == val].iloc[0])
#            denorminator_series[col] = [s]
#                
#        for i in range(nrow):
#            for j in range(i, nrow):
#                equal_indicator = (self.att.iloc[i] == self.att.iloc[j])
#                unequal_indicator =  1 - equal_indicator
#                freq_product = self.freq_frame.iloc[i] * self.freq_frame.iloc[j]
#                nrow_minus_freq_product = nrow_minus_freq.iloc[i] * nrow_minus_freq.iloc[j]
#                numerator_series =np.log(freq_product / nrow_minus_freq_product, dtype = np.float32)
#                
#                unequal_series = numerator_series / denorminator_series
#                unequal_series /= ncol
#                
#                distance_matrix[i,j] = np.sum(equal_indicator + unequal_indicator * unequal_series, axis = 1)
#                distance_matrix[j,i] = distance_matrix[i,j]
#                
#        self.burnaby_matrix = distance_matrix
#
#    def Eskin_matrix(self):
#        nrow, ncol = self.att.shape
#        ni_frame = self.att.apply(set, axis= 0).apply(lambda x:len(x))
#        unequal_series = 2/(ni_frame*ni_frame)
#        distance_matrix = np.zeros((nrow, nrow))
#        for i in range(nrow):
#            for j in range(i,nrow):
#                unequal_indicator = (self.att.iloc[i] != self.att.iloc[j])
#                distance_matrix[i,j] = np.sum(unequal_indicator * unequal_series)
#                distance_matrix[j,i] = distance_matrix[i,j]
#        self.eskin_matrix = distance_matrix / ncol

    # Define the distances

'''
###################################################################################
'''

'''****************************************************************************************
********************************************************************************************
All the above functions have been verified by  22:00 Aug.10th'''
'''#################################################################################
            CALCULATE THE DESIGN MATRIX FOR CENTERS SELECTED BY COVERAGE METHOD

**************************************************************************************

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
    
THE FLAW OF SOFTMAX: IF THE DIFFERENCES OF THE LOGIT VALUES ARE TWO BIG, THE LOSS FUNCTION MAY BE TOO BIG!
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
'''

def Loss_SVM(design_matrix, observed, coefficients, reg):
     
    nrow, ncol = design_matrix.shape
    # Reshape the coefficients
    coefficients = coefficients.reshape((-1, ncol)).T
    # Calculate the loss
    ii = np.zeros((observed.shape[1], observed.shape[1])) + 1
    Wx = design_matrix.dot(coefficients)
    s1 = Wx + 1
    obs = observed * Wx
    obsii = obs.dot(ii)
    ad = s1 - obsii
    d = ad * (1-observed)
    ind = (d>0)
    sd = d * ind
    loss = np.sum(sd)
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradients
    grad_d = ind
    grad_ad = grad_d * (1-observed)
    grad_s1 = grad_ad
    grad_obsii = - grad_ad
    grad_Wx = grad_s1
    grad_obs = grad_obsii.dot(ii)
    grad_Wx += observed * grad_obs
    grad_coeff = (design_matrix.T).dot(grad_Wx)
    
    grad_coeff += 2 * reg * coefficients
    # Reshape the gradient
    grad_coeff = grad_coeff.T.reshape((-1, 1))
    return loss, grad_coeff


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
    coefficients = train_para['coeff']
    loss_type = train_para['loss_type']
    if loss_type == 'Sigmoid':
        loss, grad_coefficients = Loss_Sigmoid(design_matrix, observed, coefficients, reg)
    elif loss_type == 'Softmax':
        loss, grad_coefficients = Loss_Softmax(design_matrix, observed, coefficients, reg)
    elif loss_type == 'SumSquares':
        loss, grad_coefficients = Loss_SumSquares(design_matrix, observed, coefficients, reg)
    elif loss_type == 'SVM':
        loss, grad_coefficients = Loss_SVM(design_matrix, observed, coefficients, reg)
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

    step_size = train_para['step_size']
#    loss_type = gd_train_para['loss_type']
    n_iterations = train_para['n_iterations']    
    
    for i in range(n_iterations):
        loss, grad_coefficients = Loss(train_para)
        # update the coefficients
        train_para['coeff'] -= step_size * grad_coefficients
        '''Do we print the loss'''
        if i % 100 == 0:
            print(round(loss, 6))
            
    return train_para, loss


def Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-3):   
    
    nrow, _ = np.shape(train_para['coeff']) 
    max_design = np.max(np.abs(train_para['design_matrix']))       
    # Create an iteration counter
    n_iteration = 0
    # BFGS algorithm
    loss, grad_coeff = Loss(train_para)
    # Initiate H. This H should not be large in case it may destroy the Loss function
    H = np.eye(nrow)
    H *= 1/(np.max(np.abs(grad_coeff)) * max_design)
    ternination_square = termination**2
    grad_square = ternination_square + 1
    while grad_square >= ternination_square:          
        # keep a record of this grad_square for monitoring the efficiency of this process
        n_iteration += 1
      
        p = - H.dot(grad_coeff)        
        # There should be both old and new coefficients in the train_para
        train_para['coeff_old'] = train_para['coeff']
        train_para['coeff'] = p + train_para['coeff_old']
        # Calculate the loss and gradient
        new_loss, new_grad_coeff = Loss(train_para)        
        # Ramijo Back-tracking
        while new_loss > loss + c * (grad_coeff.T).dot(p):
            p *= rho
            train_para['coeff'] = p + train_para['coeff_old']            
            new_loss, new_grad_coeff = Loss(train_para)        
        # update H
        s = p
        y = new_grad_coeff - grad_coeff
        r = (y.T).dot(s)
        I = np.eye(nrow)
        if r != 0:
            r = 1/r            
            H = (I - r*s.dot(y.T)).dot(H).dot(I - r*y.dot(s.T)) + r*s.dot(s.T)# Can be accelerated
        else:
            H = np.diag(np.random.uniform(0.5, 1, nrow))# try to eliminate the periodic dead loop
            H *= 1/(np.max(np.abs(new_grad_coeff))*max_design)# Make sure H is not too large
        # Update loss, grad_square and paramter
        loss = new_loss
        grad_coeff = new_grad_coeff
        grad_square = new_grad_coeff.T.dot(new_grad_coeff)            
        # print some values to monitor the training process 
        if n_iteration % 1 == 0:
            print('loss  ', loss, '    ','grad_square   ', grad_square)
            n_iteration = 0        
        
    return train_para, loss

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
def One_step_reduce_centers(train_para):
    
    method = train_para['train_method']
    nrow, ncol = train_para['design_matrix'].shape
    coeff = train_para['coeff']
    df_cc_right = train_para['df_cc_right']      

    termination = 1e-3
    '''Here, we want the coefficient is well trained at the number of centers we need'''
#    termination =10* len(centers)  
    if method == 'BFGS':      
        train_para, loss = Train_RBFN_BFGS(train_para, rho=0.85, c = 1e-3, termination=termination)
    elif method == 'GD':
        train_para, loss = Train_GD(train_para)     
    
    m =1
    '''m is the number of centers to be removed at each round of training'''
    for i in range(m):    
        # if it is the case of softmax, we have to reshape the coeff
        coeff = coeff.reshape((-1, ncol)).T
        sabs_coeff = np.sum(np.abs(coeff), axis = 1, keepdims = True)
        # find the index of the coefficients with the smallest absolute value
        ind_min = np.argmin(np.abs(sabs_coeff))
        # remove the smallest
        one_less_coeff = np.delete(coeff, ind_min, axis=0)
        train_para['coeff'] = one_less_coeff.T.reshape((-1,1))
        train_para['design_matrix'] = np.delete(train_para['design_matrix'], ind_min, axis = 1)
        train_para['df_cc_right'] = df_cc_right.drop(df_cc_right.index[ind_min])  

    return train_para, loss
'''##########################################################################'''
def Fix_index(df_train):
    nrow, _ = df_train.shape
    df_train.index = [str(i) for i in range(nrow)]
    return df_train

def Initial_cc(train_para):
    max_size = train_para['max_size']
    df_cc_left = train_para['df_cc_left']
    df_cc_right = train_para['df_cc_right']
    n_left = df_cc_left.shape[0]
    i = 0
    for i in range(n_left):
        if df_cc_right.shape[0]< max_size:
            h_array = Hamming_matrix(df_cc_right, df_cc_left.iloc[[i],:], equal=False)
            if (h_array != 0).all(axis = None):
                df_cc_right = df_cc_right.append(df_cc_left.iloc[[i],:], ignore_index=False)

        else:
            break    
    if i+1 < n_left:
        df_cc_left = df_cc_left.iloc[list(range(i, n_left)), :]
    else:
        df_cc_left = pd.DataFrame()
        
    return df_cc_right, df_cc_left
 
def Reduce_centers(train_para):
    RBF = train_para['RBF']
    reduce_step = train_para['reduce_step']
    df_train = train_para['df_train']
    df_cc_left = train_para['df_cc_left']
    nCenters_list = train_para['nCenters_list']
    nClass = train_para['nClass']
    distance_type = train_para['distance_type']
    dict_element_dict = train_para['dict_element_dict']
    dict_log_freq_dict = train_para['dict_log_freq_dict']
    attach = False
    while df_cc_left.shape[0] != 0:
        print('Left     ', df_cc_left.shape[0])# Print how many left
        for i in range(reduce_step):
            train_para, loss = One_step_reduce_centers(train_para)
            
        if not attach:
            attach_not_loss = loss
            df_cc_right_not = copy.deepcopy(train_para['df_cc_right'])
            design_matrix_not = copy.deepcopy(train_para['design_matrix'])
            coeff_not = copy.deepcopy(train_para['coeff'])
        else:
            if loss > attach_not_loss:# Not accept
                train_para['df_cc_right'] = copy.deepcopy(df_cc_right_not)
                train_para['design_matrix'] = copy.deepcopy(design_matrix_not)
                train_para['coeff'] = copy.deepcopy(coeff_not)
                print('not accept')
            else:# accept
                attach_not_loss = loss
                df_cc_right_not = copy.deepcopy(train_para['df_cc_right'])
                design_matrix_not = copy.deepcopy(train_para['design_matrix']) 
                coeff_not = copy.deepcopy(train_para['coeff'])
                print('accept')
        # Make up the lost cc
        df_cc_right, df_cc_left = Initial_cc(train_para)
        train_para['df_cc_right'] = df_cc_right
        train_para['df_cc_left'] = df_cc_left
        # make up the coefficients before the next round of reduction
        coeff = train_para['coeff'].reshape((nClass, -1))
        d = df_cc_right.shape[0] - coeff.shape[1] 
        # When attach we can make sure the loss is decreasing
        
        if d > 0:
            attach = True
            coeff = np.hstack((coeff, np.zeros((nClass, d))))
            train_para['coeff'] = copy.deepcopy(coeff.reshape((-1, 1)))
        # Make up the design matrix
            patch_dist = Distance_matrix(df_train, df_cc_right.iloc[-d:], distance_type,\
                                         dict_element_dict, dict_log_freq_dict, equal = False)
                
            patch_design = Design_matrix(patch_dist, RBF)
            train_para['design_matrix'] = np.hstack((train_para['design_matrix'], patch_design))
            
            # The termination here should be the same as in the one step reduce
#            train_para, _ = Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-4, termination = 1e-3)
    
    # Continue to reduce to the target number of centers
    test_para = {}
    test_para['df_Centers_list'] = []
    test_para['nCenters_list'] = nCenters_list
    test_para['coeff_list'] = []
    for nCenters in nCenters_list:
#        print(train_para['df_cc_right'].shape[0])
        while train_para['df_cc_right'].shape[0] > nCenters:
            train_para, _ = One_step_reduce_centers(train_para)
        # train again befor record the coefficients
        train_para, _ = Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-3)
        test_para['df_Centers_list'].append(copy.deepcopy(train_para['df_cc_right']))
        test_para['coeff_list'].append(copy.deepcopy(train_para['coeff']))
        
    return test_para
        

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        THE COVERAGE CENTER SELECTION
        
        
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
        
    eliminated.append(0)# The first sample will always be selected
        
    return eliminated, radius


def Estimate_radius(sub_df_train, distance_type, first_nCenters,\
                    dict_element_dict, dict_log_freq_dict):
    
#    sub_dict_element_dict, sub_dict_log_freq_dict = Frequency_distinct(sub_df_train)
    # Calculate the distance matrix
    print('Calculating the sub_dist matrix')
    sub_dist = Distance_matrix(sub_df_train, sub_df_train, distance_type,\
                               dict_element_dict, dict_log_freq_dict, equal = True)
    # Generate the required radius list
    eliminated, radius = CS_coverage_all(sub_dist)
    
    first_radius = radius[-first_nCenters]
    
    return first_radius
    
'''Select the centers according to the radius list returned in Estimate_radius'''    
def CS_coverage(df_train, first_radius, distance_type,\
                dict_element_dict, dict_log_freq_dict, nCenters_list):
    max_centers = df_train.iloc[[0], :]
    nrow, _ = df_train.shape
    for i in range(nrow):
        print('test the   ', i, '   for centers')
        dist = Distance_matrix(max_centers, df_train.iloc[[i], :], distance_type,\
                               dict_element_dict, dict_log_freq_dict, equal = False)

        if (dist > first_radius).all(axis = None):
           max_centers = max_centers.append(df_train.iloc[[i], :], ignore_index=False)
           
    # Begin with the max_centers
    print('max_centers    ', max_centers.shape[0])
    dist_mat = Distance_matrix(max_centers, max_centers, distance_type,\
                               dict_element_dict, dict_log_freq_dict, equal = True)
    eliminated, radius = CS_coverage_all(dist_mat)
    
    # Select according to radius
    centers_list = []; centers_ind_list = []
    for nCenters in nCenters_list:
        center_ind = [eliminated[n] for n in range(-nCenters, 0)]
        centers_ind_list.append(copy.deepcopy(center_ind))
        centers_list.append(max_centers.iloc[center_ind, :])
    
    return centers_list

def Coverage_train(train_para, first_radius, dict_element_dict, dict_log_freq_dict):
    df_train = train_para['df_train']
    distance_type = train_para['distance_type']
    nCenters_list = train_para['nCenters_list']
    RBF = train_para['RBF']
    nClass = train_para['nClass']
    df_Centers_list = CS_coverage(df_train, first_radius, distance_type,\
                dict_element_dict, dict_log_freq_dict, nCenters_list)
    # Calculate the design matrix
    dist = Distance_matrix(df_train, df_Centers_list[0], distance_type, dict_element_dict, dict_log_freq_dict, equal=False)
    design_dist = Design_matrix(dist, RBF)
    
    test_para = {}
    test_para['coeff_list'] = []
    for nCenters in nCenters_list:
        design_matrix = design_dist[:, list(range(-nCenters, 0))]
        train_para['design_matrix'] = design_matrix
        # Initiate the coefficients
        train_para['coeff'] = np.zeros((nCenters*nClass, 1))
        # Train
        train_para, _ = Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-3)
        # Load to test para
        test_para['coeff_list'].append(copy.deepcopy(train_para['coeff']))
    
    test_para['nCenters_list'] = train_para['nCenters_list']
    test_para['RBF'] = RBF
    test_para['distance_type'] = distance_type
    test_para['df_Centers_list'] = df_Centers_list
    
    return test_para


'''#####################################################################
               CALCULATE MCC
'''
'''
Denominator_factor: this is a small subfuntion of Test_MCC.
'''
def Denominator_factor(c_matrix):
    denominator_factor = 0
    nclass, _ = c_matrix.shape
    c_rowsum = np.sum(c_matrix, axis = 1, keepdims=True)
    for k in range(nclass):
        for kp in range(nclass):
            if kp != k:
                for lp in range(nclass):
                    denominator_factor += c_rowsum[k]*c_matrix[kp,lp]
    return denominator_factor

'''
Test_MCC:This function is to calculate the MCC. The binary classification should be considered as a 
        special case of the multiclass classification problem, with the number of class be 2.
'''
def Test_MCC(test_para):
    observed = test_para['observed']
    n_class = observed.shape[1]
    design_matrix_list = test_para['design_matrix_list']
    coefficients_list = test_para['coeff_list']
    nCenters_list = test_para['nCenters_list']
    
    test_para['mcc_list'] = [] 
    test_para['c_matrix_list'] = []

    for ind, nCenters in enumerate(nCenters_list):
        coefficients = coefficients_list[ind]
        design_matrix = design_matrix_list[ind]
        # Set the shape of coefficients
        coefficients = coefficients.reshape((-1, nCenters)).T
        pred_logit = design_matrix.dot(coefficients)
        max_logit = np.max(pred_logit, axis = 1, keepdims=True)
        pred_bool = pred_logit==max_logit
        # Calculate the C matrix
        c_matrix = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                pred_i = observed[pred_bool[:, i] == True, :]
                # pred_i is a subset of observed with the predicted class i
                observe_j = pred_i[pred_i[:, j] == 1, :]
                #observe_j is the set of samples with predicted class i and observed class j
                c_matrix[i,j] = observe_j.shape[0]
        test_para['c_matrix_list'].append(copy.deepcopy(c_matrix))        
        # Calculate the MCC
        numerator = [0]
        for ka in range(n_class):
            for la in range(n_class):
                for ma in range(n_class):
                    numerator += c_matrix[ka, ka]*c_matrix[la, ma] - c_matrix[ka, la]*c_matrix[ma, ka]
                    
                    
        df1 = Denominator_factor(c_matrix)
        df2 = Denominator_factor(c_matrix.T)
        denominator = df1**(0.5)*df2**(0.5)
        if denominator == 0:
            denominator = 1
        # Calculate the MCC
#        print(numerator)
        mcc = numerator/denominator
        test_para['mcc_list'].append(mcc[0])
        
    return test_para
'''#########################################################################################'''

def Main(df_train, df_test, test_observed, train_observed, nClass, reg_coverage, reg_coeff, nCenters_list,
         RBF, dist_type):
    # Load the basic train_para
    df_train = Fix_index(df_train)
    dict_element_dict, dict_log_freq_dict = Frequency_distinct(df_train)
    
    # Set the train para
    train_para = {}      
    
    '''Those parameters is common to both coverage method and coefficient method'''
    '''*************************************************************************'''    
    # Adjustable parameters for different RBF, distance_type loss_type, train_method and reg
    train_para['RBF'] = RBF
    train_para['distance_type'] = dist_type
    train_para['loss_type'] = 'Softmax'
    train_para['train_method'] = 'BFGS'
   
    
    # Parameters about selecting centers
    train_para['nCenters_list'] = nCenters_list
    
    # Load the training data    
    train_para['df_train'] = df_train
    train_para['observed'] = train_observed
    train_para['nClass'] = nClass
    train_para['dict_element_dict'] = dict_element_dict
    train_para['dict_log_freq_dict'] = dict_log_freq_dict
    '''*************************************************************************'''
    
    
    '''Those parameters is only for coverage method'''
    '''*************************************************************************''' 
    # Load the para for Coverage center selection
    first_nCenters = 64; nSub = 1500
    sub_df_train = df_train.iloc[list(range(nSub)), :]
    first_radius = Estimate_radius(sub_df_train, train_para['distance_type'], first_nCenters,\
                    dict_element_dict, dict_log_freq_dict)
    train_para['reg'] = reg_coverage
    '''*************************************************************************'''


    '''Test the results of coverage method'''
    '''*************************************************************************'''    
    # Train the Coverage centers
    test_para_coverage = Coverage_train(train_para, first_radius, dict_element_dict, dict_log_freq_dict)
    
    # Calculate the design matrix for the test
    df_Centers_list = test_para_coverage['df_Centers_list']
    distance_type = test_para_coverage['distance_type']
    RBF = test_para_coverage['RBF']
    nCenters_list = test_para_coverage['nCenters_list']
    
    L_dist = Distance_matrix(df_test, df_Centers_list[0], distance_type,\
                             dict_element_dict, dict_log_freq_dict, equal = False)
    L_design_matrix = Design_matrix(L_dist, RBF)
    
    # load the test_para_coverage with more values
    test_para_coverage['design_matrix_list'] = []
    for nCenters in nCenters_list:
        design_matrix = L_design_matrix[:, list(range(-nCenters, 0))]
        test_para_coverage['design_matrix_list'].append(copy.deepcopy(design_matrix))    
    test_para_coverage['observed'] = test_observed
    
    # Calculate mcc
    test_para_coverage = Test_MCC(test_para_coverage)
    '''*************************************************************************'''
    
    
    '''Load the train_para for coefficients method'''
    '''*************************************************************************'''  
    start_time = time.time()
    train_para['reg'] = reg_coeff
    train_para['max_size'] = train_para['nCenters_list'][0] + 1 # The maximum size of the selected centers, 
#    train_para['max_size'] = 100
    train_para['df_cc_left'] = df_train
    train_para['df_cc_right'] = pd.DataFrame()
    train_para['reduce_step'] = 1# The bigger the harder to train  
    
    # Calculate the initial design matrix
    df_cc_right, df_cc_left = Initial_cc(train_para) 
    train_para['df_cc_left'] = df_cc_left
    train_para['df_cc_right'] = df_cc_right 
    #df_cc_right.shape
    #df_cc_right
    #df_cc_left.shape
    #'''Load the design matrix according to different matrix type'''
    dist = Distance_matrix(df_train, df_cc_right,train_para['distance_type'],\
                                   dict_element_dict, dict_log_freq_dict, equal = False)
    #hamming_dist.shape
    ## Calculate design matrix
    design_matrix = Design_matrix(dist, RBF=train_para['RBF'])
    train_para['design_matrix'] = design_matrix
    # The initial coeff
    train_para['coeff'] = np.zeros((design_matrix.shape[1]*train_para['nClass'], 1))
    #Find the centers  
    test_para_coefficients = Reduce_centers(train_para)
    # Load the design matrix
    test_para_coefficients['design_matrix_list'] = []
    # Calculate the design matrix for the test_para_coeff
    for df_Centers in test_para_coefficients['df_Centers_list']:
        dist = Distance_matrix(df_test, df_Centers,train_para['distance_type'],\
                                       dict_element_dict, dict_log_freq_dict, equal = False)
        design_matrix = Design_matrix(dist, train_para['RBF'])
        test_para_coefficients['design_matrix_list'].append(copy.deepcopy(design_matrix))
    test_para_coefficients['observed'] = test_observed
    
    # Calculate the mcc
    test_para_coefficients = Test_MCC(test_para_coefficients)
    print('time   ', time.time() - start_time)
    return test_para_coverage, test_para_coefficients
'''#####################################################################################'''
'''TEST THE ABOVE ON BREAST CANCER DATA'''

def Wrangling_BC():
    os.chdir('/home/leo/Documents/Project_SelectCenters/DATA/BreastCancer')
    data = pd.read_csv('breast-cancer-wisconsin.data')

    col_name = data.columns

    complete = data[:][data[col_name[6]] != '?']
    complete[col_name[6]] = complete[col_name[6]].astype(np.int64)
    
    attributes = list(col_name[1:10])
    categories = [col_name[-1]]
    
    return complete, attributes, categories

#complete, attributes, categories = Wrangling_BC()
#complete.shape
#
#test = complete.iloc[list(range(140))]
#train = complete.iloc[list(range(140, 682))]
#
#df_test = test[attributes]
#
#df_train = train.loc[:, attributes]
#df_train = Fix_index(df_train)
#
#test_observed = test[categories]
#test_observed[test_observed['2.1'] == 2] = 0
#test_observed[test_observed['2.1'] == 4] = 1
#test_observed = np.float64(np.hstack((test_observed.values, 1 - test_observed.values)))
#
#train_observed = train[categories]
#train_observed[train_observed['2.1'] == 2] = 0
#train_observed[train_observed['2.1'] == 4] = 1
#train_observed = np.float64(np.hstack((train_observed.values, 1 - train_observed.values)))
#
##df_test.shape
##df_train.shape
##test_observed.shape
##train_observed.shape
#
#test_para_coverage, test_para_coefficients= Main(df_train, df_test, test_observed, train_observed)
#
#test_para_coefficients.keys()
#test_para_coefficients['df_Centers_list']
#test_para_coefficients['mcc_list']
#test_para_coefficients['c_matrix_list']
##test_para_coefficients_new['df_Centers_list']
##test_para_coefficients_new['mcc_list']
##test_para_coefficients_new['c_matrix_list']
##
#test_para_coverage['coeff_list']
#test_para_coverage['nCenters_list']
#test_para_coverage['RBF']
#test_para_coverage['distance_type']
#test_para_coverage['df_Centers_list']
#test_para_coverage['design_matrix_list'][0].shape
#test_para_coverage['mcc_list']
#test_para_coverage['c_matrix_list']
##############################################################################
'''TEST THE ABOVE ON BREAST CANCER DATA'''
def Wrangle_NUR():
    os.chdir('/home/leo/Documents/Project_SelectCenters/DATA/Nursery')
    data = pd.read_csv('nursery.data')
    # Change the column name
    data.columns = ['parents', 'has_nur', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
    att = ['parents', 'has_nur', 'form', 'children', 'housing', 'finance', 'social', 'health']
    
#    classes = ['not_recom', 'priority', 'spec_prior']
    classes = ['priority', 'spec_prior']
    nClass = len(classes) + 1
    # Shuffle the data
    data = data.sample(frac = 1)
    # Split into train and test
    train = data.iloc[list(range(10000))]
    test = data.iloc[list(range(10000, data.shape[0]))]
    
    df_train = train[att]
    df_test = test[att]
    
    # Generate the observed
    train_observed = np.zeros((df_train.shape[0], nClass))
    test_observed = np.zeros((df_test.shape[0], nClass))
    for ind, cla in enumerate(classes):
        train_observed[:, ind] = train['class'].eq(cla) * 1
        test_observed[:, ind] = test['class'].eq(cla) * 1
    train_observed[:, nClass-1] = (1- np.sum(train_observed, axis= 1, keepdims=True)).T
    test_observed[:, nClass-1] = (1 - np.sum(test_observed, axis = 1, keepdims=True)).T
    
    return df_train, df_test, train_observed, test_observed, nClass


def NUR_main():
    os.chdir('/home/leo/Documents/Project_SelectCenters/DATA/Nursery')
    data = pd.read_csv('nursery.data')
    # Change the column name
    data.columns = ['parents', 'has_nur', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
    att = ['parents', 'has_nur', 'form', 'children', 'housing', 'finance', 'social', 'health']
    
    #    classes = ['not_recom', 'priority', 'spec_prior']
    classes = ['priority', 'spec_prior']
    nClass = len(classes) + 1
    # Shuffle the data
    data = data.sample(frac = 1)
#    data = data.iloc[list(range(500))]# This is for small batch test
    # Split into train and test
    n_train = math.floor(data.shape[0]/4 * 3)
    train = data.iloc[list(range(n_train))]
    test = data.iloc[list(range(n_train, data.shape[0]))]
    
    df_train = train[att]
    df_test = test[att]
    
    # Generate the observed
    train_observed = np.zeros((df_train.shape[0], nClass))
    test_observed = np.zeros((df_test.shape[0], nClass))
    
    for inde, cla in enumerate(classes):
        train_observed[:, inde] = train['class'].eq(cla) * 1
        test_observed[:, inde] = test['class'].eq(cla) * 1
        
    train_observed[:, nClass-1] = (1- np.sum(train_observed, axis= 1, keepdims=True)).T
    test_observed[:, nClass-1] = (1 - np.sum(test_observed, axis = 1, keepdims=True)).T
    
    # Split the train into cross_train and cross_test
    n_cross_train = math.floor(train.shape[0]/3 * 2)
    cross_train = train.iloc[list(range(n_cross_train))]
    cross_test = train.iloc[list(range(n_cross_train, train.shape[0]))]
    
    df_cross_train = cross_train[att]
    df_cross_test = cross_test[att]
    
    cross_train_observed = train_observed[list(range(n_cross_train)),:]
    cross_test_observed = train_observed[list(range(n_cross_train, train.shape[0])), :]
    
    
    
    # Do the cross validation
    nCenters_list=[64,32,8, 4, 2]
    reg_list = [0.01, 0.05, 0.2] 
    RBF_list = ['Gaussian', 'Markov', 'Inverse_Multi_Quadric', 'Thin_Plate_Spline']
    dist_type_list = ['Hamming', 'IOF', 'OF']
    
    for RBF in RBF_list:
        for dist_type in dist_type_list:
            cross_mcc_coverage = np.zeros((len(reg_list), len(nCenters_list)))
            cross_mcc_coefficients = np.zeros((len(reg_list), len(nCenters_list)))
            for ind, reg in enumerate(reg_list):
                test_para_coverage, test_para_coefficients=\
                 Main(df_cross_train, df_cross_test, cross_test_observed, cross_train_observed, nClass, reg,reg, nCenters_list,\
                      RBF, dist_type)
                 
                cross_mcc_coefficients[ind, :]=np.array(test_para_coefficients['mcc_list']).reshape((1, -1))
                cross_mcc_coverage[ind, :]=np.array(test_para_coverage['mcc_list']).reshape((1, -1)) 
            
            # Do the test with the best parameter
            mcc_list_coverage = []; mcc_list_coeff = []
            nCenter_reg_list_coverage = []; nCenter_reg_list_coeff = []
            results = {}# Load the results to this dictionary
            for index, nCenters in enumerate(nCenters_list):
                reg_coeff =reg_list[np.argmax(cross_mcc_coefficients[:, index])]
                reg_coverage = reg_list[np.argmax(cross_mcc_coverage[:, index])]
                
                nCenter_reg_list_coeff.append([nCenters, reg_coeff])
                nCenter_reg_list_coverage.append([nCenters, reg_coverage])
                
                test_para_coverage, test_para_coefficients=\
                Main(df_train, df_test, test_observed, train_observed, nClass, reg_coverage, reg_coeff, [nCenters], 
                     RBF, dist_type)
                
                mcc_list_coverage.append(copy.deepcopy(test_para_coverage['mcc_list'][0]))
                mcc_list_coeff.append(copy.deepcopy(test_para_coefficients['mcc_list'][0]))        
                
                # Load the results 
                results['mcc_list_coverage'] = mcc_list_coverage
                results['mcc_list_coeff'] = mcc_list_coeff
                results['nCenter_reg_list_coverage'] = nCenter_reg_list_coverage
                results['nCenter_reg_list_coeff'] = nCenter_reg_list_coeff
                
                # Save the results for each calculation in case it may collapse somewhere
                os.chdir('/home/leo/Documents/Project_SelectCenters/Code/Results/NUR')
                RES = pd.DataFrame(results)
                RES.to_pickle(RBF+'_'+dist_type+'_'+'NUR_results.pkl')
    
    return

if __name__ == '__main__':   
    NUR_main()

#r = pd.read_pickle('NUR_results.pkl') 
#r1 = pd.read_pickle('NUR_results1.pkl') 
#r.columns
#r['mcc_list_coverage']
#r['mcc_list_coeff']
#r['nCenter_reg_list_coverage']
#r['nCenter_reg_list_coeff']
#
#r1['mcc_list_coverage']
#r1['mcc_list_coeff']
#r1['nCenter_reg_list_coverage']
#r1['nCenter_reg_list_coeff']


