'''This file is to deal with large data set. It should be able to process small data set'''
import os
import pandas as pd
import numpy as np
import copy
import math
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



def Hamming_matrix(df_train, df_cc_right):
    ncol, _ = df_cc_right.shape
    nrow, natt = df_train.shape
    distance_matrix = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            distance_matrix[i,j] = np.sum(df_train.iloc[i] != df_cc_right.iloc[j])
    distance_matrix /= natt       
    return distance_matrix



def IOF_OF_matrix(df_train, df_cc_right, dict_element_dict, dict_log_freq_dict):
    
    n_cc_right, _ = df_cc_right.shape
    nrow, natt = df_train.shape
    att_list = df_train.columns     
    
    df_freq_train_log = pd.DataFrame(0, index= list(range(nrow)), columns= df_train.columns)
    df_freq_cc_right_log = pd.DataFrame(0, index = list(range(n_cc_right)), columns=df_cc_right.columns)
    
    for att in att_list:
        log_freq_dict = dict_log_freq_dict[att]
        for i in range(nrow):
            df_freq_train_log.loc[i, att] = log_freq_dict[df_train.loc[i,att]]
        for j in range(n_cc_right):
            df_freq_cc_right_log.loc[j, att] = log_freq_dict[df_cc_right.loc[j,att]]
    # Calculate iof distance matrix
    iof_dist = np.zeros((nrow, n_cc_right))
    of_dist = np.zeros_like(iof_dist)
    logk = np.log(nrow, dtype=np.float32)
    for i in range(nrow):
        for j in range(n_cc_right):
            indicator =1 - df_train.iloc[i].eq(df_cc_right.iloc[j])          
            product_iof = np.float32(df_freq_train_log.iloc[i]*df_freq_cc_right_log.iloc[j])
            iof_dist[i,j] = np.sum(indicator * product_iof)
            
            product_of = np.float32((df_freq_train_log.iloc[i]-logk)*(df_freq_cc_right_log.iloc[j]-logk))
            of_dist[i,j] = np.sum(indicator * product_of)
    iof_dist /= natt
    of_dist /= natt

    return iof_dist, of_dist



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


def Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-4, termination = 1e-2):   
    
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

    termination = 1e-3
    '''Here, we want the coefficient is well trained at the number of centers we need'''
#    termination =10* len(centers)  
    if method == 'BFGS':      
        train_para, loss = Train_RBFN_BFGS(train_para, rho=0.85, c = 1e-3, termination=termination)
    elif method == 'GD':
        train_para, loss = Train_GD(train_para)           

    return train_para
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
            h_array = Hamming_matrix(df_cc_right, df_cc_left.iloc[[i],:])
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
    df_cc_left = train_para['df_cc_left']
    nCenters_list = train_para['nCenters_list']
    nClass = train_para['nClass']
    distance_type = train_para['distance_type']
    dict_element_dict = train_para['dict_element_dict']
    dict_log_freq_dict = train_para['dict_log_freq_dict']
    
    while df_cc_left.shape[0] != 0:
        print('Left     ', df_cc_left.shape[0])# Print how many left
        for i in range(reduce_step):
            train_para = One_step_reduce_centers(train_para)
        # Make up the lost cc
        df_cc_right, df_cc_left = Initial_cc(train_para)
        train_para['df_cc_right'] = df_cc_right
        train_para['df_cc_left'] = df_cc_left
        # make up the coefficients before the next round of reduction
        coeff = train_para['coeff'].reshape((nClass, -1))
        d = df_cc_right.shape[0] - coeff.shape[1] 
        if d > 0:
            coeff = np.hstack((coeff, np.zeros((nClass, d))))
            train_para['coeff'] = copy.deepcopy(coeff.reshape((-1, 1)))
        # Make up the design matrix
            if distance_type == 'Hamming':
                patch_dist = Hamming_matrix(df_train, df_cc_right.iloc[-d:])
            elif distance_type == 'IOF':
                patch_dist, _ = IOF_OF_matrix(df_train, df_cc_right,\
                                              dict_element_dict, dict_log_freq_dict)
            elif distance_type == 'OF':
                _, patch_dist = IOF_OF_matrix(df_train, df_cc_right,\
                                              dict_element_dict, dict_log_freq_dict)
                
            patch_design = Design_matrix(patch_dist, RBF)
            train_para['design_matrix'] = np.hstack((train_para['design_matrix'], patch_design))
    
    # Continue to reduce to the target number of centers
    test_para = {}
    test_para['df_Centers_list'] = []
    test_para['nCenters_list'] = nCenters_list
    test_para['coeff_list'] = []
    for nCenters in nCenters_list:
        while train_para['df_cc_right'].shape[0] > nCenters:
            train_para = One_step_reduce_centers(train_para)
        test_para['df_Centers_list'].append(copy.deepcopy(train_para['df_cc_right']))
        test_para['coeff_list'].append(copy.deepcopy(train_para['coeff']))
        
    return test_para
        
# Load the train_para
    
sample_data = Small_sample(size = 500)
df_train = sample_data.loc[:,['att1', 'att2', 'att3']]
df_train = Fix_index(df_train)

# First batch of train_para
train_para = {}
train_para['df_cc'] = df_train
train_para['max_size'] = 20
train_para['df_cc_left'] = df_train
train_para['df_cc_right'] = pd.DataFrame()
train_para['reduce_step'] = 1# The bigger the harder to train
train_para['RBF'] = 'Gaussian'

df_cc_right, df_cc_left = Initial_cc(train_para) 
train_para['df_cc_left'] = df_cc_left
train_para['df_cc_right'] = df_cc_right 
df_cc_right.shape
df_cc_right
df_cc_left.shape
hamming_dist = Hamming_matrix(df_train, df_cc_right)
hamming_dist.shape
# Calculate design matrix
design_matrix = Design_matrix(hamming_dist, RBF=train_para['RBF'])
train_para['design_matrix'] = design_matrix
design_matrix.shape

dict_element_dict, dict_log_freq_dict = Frequency_distinct(df_train)
#iof_dist, of_dist = IOF_OF_matrix(df_train, df_cc_right, dict_element_dict, dict_log_freq_dict)

train_para['nCenters_list']= [5, 3]
train_para['nClass'] = 2
train_para['distance_type'] = 'Hamming'
train_para['dict_element_dict'] = dict_element_dict
train_para['dict_log_freq_dict'] = dict_log_freq_dict
train_para['train_method'] = 'BFGS'
train_para['coeff'] = np.zeros((design_matrix.shape[1]*train_para['nClass'], 1))
train_para['reg'] = 0.1
train_para['loss_type'] = 'Softmax'
# Calculate the observed values
sample_data['label']
observed = sample_data[['label']]
observed[observed['label'] == 'y1'] = 0
observed[observed['label'] == 'y2'] = 1
observed_total = np.float64(np.hstack((observed.values, 1 - observed.values)))
observed_total
observed
train_para['observed'] = observed_total
train_para['coeff'].shape
train_para['design_matrix'].shape

test_para = Reduce_centers(train_para)

#test_para['df_Centers_list'][1].shape
#test_para['coeff_list'][1].shape
#test_para['df_Centers_list'][1]
#test_para['coeff_list'][1]
'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
#RBF = train_para['RBF']
#design_matrix = train_para['design_matrix']
#reduce_step = train_para['reduce_step']
#df_cc_left = train_para['df_cc_left']
#nCenters_list = train_para['nCenters_list']
#nClass = train_para['nClass']
#distance_type = train_para['distance_type']
#dict_element_dict = train_para['dict_element_dict']
#dict_log_freq_dict = train_para['dict_log_freq_dict']
##
#    
#while df_cc_left.shape[0] != 0:
#    for i in range(reduce_step):
#        train_para = One_step_reduce_centers(train_para)
#        # Make up the lost cc
#        df_cc_right, df_cc_left = Initial_cc(train_para)
#        train_para['df_cc_right'] = df_cc_right
#        train_para['df_cc_left'] = df_cc_left
#        # make up the coefficients before the next round of reduction
#        coeff = train_para['coeff'].reshape((nClass, -1))
#        d = df_cc_right.shape[0] - coeff.shape[1] 
#        if d > 0:
#            coeff = np.hstack((coeff, np.zeros((nClass, d))))
#            train_para['coeff'] = copy.deepcopy(coeff.reshape((-1, 1)))
#        # Make up the design matrix
#            if distance_type == 'Hamming':
#                patch_dist = Hamming_matrix(df_train, df_cc_right.iloc[-d:])
#            elif distance_type == 'IOF':
#                patch_dist, _ = IOF_OF_matrix(df_train, df_cc_right,\
#                                              dict_element_dict, dict_log_freq_dict)
#            elif distance_type == 'OF':
#                _, patch_dist = IOF_OF_matrix(df_train, df_cc_right,\
#                                              dict_element_dict, dict_log_freq_dict)
#                
#            patch_design = Design_matrix(patch_dist, RBF)
#            train_para['design_matrix'] = np.hstack((train_para['design_matrix'], patch_design))




















