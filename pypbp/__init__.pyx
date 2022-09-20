# Importanting the packages
from .pypbp import *
import numpy as np
import emd
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, canberra
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from scipy import interpolate
from scipy.integrate import trapz
from scipy.optimize import dual_annealing
import pandas as pd
import matplotlib.pyplot as plt
import os
# Cython functions
# Decomposition function
cdef list time_series
def get_decomposition(time_series):
            
    time_series_array = np.array(time_series)
    IMFs = emd.sift.sift(np.log(time_series_array+1))    

    return(np.exp(np.sum(IMFs[:, 0:5], axis=1)-1))

# Alert Zone procedure
cdef int t, ti 
cdef event_vector, classes
def obtain_patterns(list observation_set, 
                    double event,
                    int m, 
                    decompose, 
                    decomposition_function):
    '''Alert Zone Procedure implementation'''
    event_vector = []
    classes = []
    if decompose:
        decomposed_observation_set = decomposition_function(observation_set)
        
    for t in range(len(observation_set)):        
        if t > (m - 1):            
            if observation_set[t] >= event:                
                classes.append(1)
                for ti in range(m):   

                    if decompose:
                        event_vector.append(decomposed_observation_set[t - (m - ti)])
                    else:
                        event_vector.append(observation_set[t - (m - ti)])                    
            else:
                classes.append(0)
                for ti in range(m):                    
                    
                    if decompose:
                        event_vector.append(decomposed_observation_set[t - (m - ti)])
                    else:
                        event_vector.append(observation_set[t - (m - ti)])                    
                
    patterns = np.array(event_vector)
    patterns = patterns.reshape(int(len(patterns)/m), m)                
    return(patterns, np.array(classes))

############################# Clustering procedure #############################
cdef int i, j
cdef double pearson_value, association_value
cdef list temporary_clusters, temporary_rows, pc_rows, pc
class Pattern_Matrix:
    '''This function obtrain a matrix with the patterns clustered according to the proposed association'''

    def obtain_pc_with_correlation(self, P, float d_cluster):

        temporary_clusters = []
        temporary_rows = []
        pc_rows = []
        pc = []

        for i in range(int(len(P))):

            if len(P) > 0:

                for j in range(int(len(P))):

                    pearson_value = pearsonr(P[0], P[j])[0]


                    if pearson_value >= d_cluster:
                        temporary_clusters.append(list(P[j]))
                        temporary_rows.append(j)

                P = np.delete(P, [temporary_rows], 0)

                pc_rows.append(len(temporary_rows))
                pc.append(np.array(temporary_clusters))

                temporary_rows = []
                temporary_clusters = []
            else: 
                break
        
        self.pc_rows = pc_rows
        self.pc = pc
        
        return(self.pc_rows, self.pc)
    
    def obtain_pc_with_distance(self, P, float d_cluster):

        temporary_clusters = []
        temporary_rows = []
        pc_rows = []
        pc = []

        for i in range(int(len(P))):

            if len(P) > 0:

                for j in range(int(len(P))):

                    association_value = 1/(canberra(P[0], P[j])+1)                    

                    if association_value >= d_cluster: # Corrigir aqui! distance_value
                        temporary_clusters.append(list(P[j]))
                        temporary_rows.append(j)

                P = np.delete(P, [temporary_rows], 0)

                pc_rows.append(len(temporary_rows))
                pc.append(np.array(temporary_clusters))

                temporary_rows = []
                temporary_clusters = []
            else: 
                break
        
        self.pc_rows = pc_rows
        self.pc = pc
        
        return(self.pc_rows, self.pc)

############################## Prediction based on d_base ############################################
cdef list p_means_list, correlations, associations, prediction
cdef double[:, :] p_means

def get_d_pred(float d_base, float alpha, int l):    
    
    return(d_base + (1 - d_base) * (l)**(-alpha))

class Prediction():
    
    def obtain_p_means_with_correlation(self, double[:] field_sample, tuple patterns):
        
        p_means_list = []
    
        for i in range(len(patterns[0])):
            
            p_means_list.append(list(np.mean(patterns[1][i], axis = 0)))
    
        
        p_means = np.array(p_means_list)
        correlations = []
    
        for j in range(len(p_means)):
            correlations.append(pearsonr(p_means[j], field_sample)[0])
            
        self.p_means = p_means
        self.correlations = np.array(correlations)
        self.patterns_row_numbers = patterns[0]
        
        return(self.p_means, self.correlations, self.patterns_row_numbers)
    
    def obtain_p_means_with_distance(self, double[:] field_sample, tuple patterns):
        
        p_means_list = []            
        
        for i in range(len(patterns[0])):
            
            p_means_list.append(list(np.mean(patterns[1][i], axis = 0)))
                
        
        # Correlação:
        p_means = np.array(p_means_list)            
        associations = []
            
    
        for j in range(len(p_means)):
            associations.append(1/(canberra(p_means[j], field_sample) + 1)) 
                    
            
        self.p_means = p_means        
        self.associations = np.array(associations) 
        self.patterns_row_numbers = patterns[0]
        
        return(self.p_means, self.associations, self.patterns_row_numbers)
    
    def predict_with_correlation(self, float d_base, float alpha):
        
        prediction = []        
        
        for i in range(len(self.sub_patterns)):
            
            if self.correlations[i] >= get_d_pred(d_base=d_base, l = self.patterns_row_numbers[i], alpha = alpha):
                
                prediction.append(1)
            
            else:
                
                prediction.append(0)
                
            if np.any(prediction) == 1:
            
                result = 1
            else:
                result = 0
                
        return(result)
    
    def predict_with_distance(self, float d_base, float alpha):
        
        prediction = []
            
        
        for i in range(len(self.associations)):
            
            if self.associations[i] >= get_d_pred(d_base=d_base, l = self.patterns_row_numbers[i], alpha = alpha):
                
                prediction.append(1)
            
            else:
                
                prediction.append(0)
                
            if np.any(prediction) == 1:
            
                result = 1
            else:
                result = 0
                
        return(result)
############################## Roc curve calculous ############################################
cdef tuple clustered_patterns
cdef float d_cluster, d_base, alpha
cdef list predictions
def pbp_prediction(patterns_array, clustered_patterns, d_base, alpha, outbreak_p_means, outbreak_prediction):
                   '''This function obtain a list of predictions from the PBP method'''
                   predictions = []
                   
                   for i in range(len(patterns_array)):
                       my_mean_applied = outbreak_p_means(field_sample = patterns_array[i,:],  patterns = clustered_patterns)
                       predictions.append(outbreak_prediction(d_base = d_base, alpha = alpha))
                    
                   return(predictions)
cdef float fpr, tpr
def check_zero_division(cm):
    '''This function computes the true and false positive rates from a confusion matrix cm'''
    if any(np.sum(cm, axis = 1)==0):            
            tpr = cm[:,1][1]/(np.sum(cm, axis = 1)[1]+1e-16)
            fpr = cm[:,1][0]/(np.sum(cm, axis = 1)[0]+1e-16)    
    else:           
        rates = cm[:,1]/np.sum(cm, axis = 1)    
        fpr = rates[0]    
        tpr = rates[1]
    if np.isnan(tpr):        
        tpr=0
        
    return(tpr, fpr)

def check_and_compute_rates(predictions, classes, cm):                                    
        
    if (sum(predictions) == 0 and sum(classes) == 0):        
        fpr = 0
        tpr = 0
    elif (sum(predictions) == len(predictions) and sum(classes) == len(classes)):        
        fpr = 0
        tpr = 1
    else:            
        tpr, fpr = check_zero_division(cm)
        
    return(tpr, fpr)

def get_rates(predictions, classes):
        
        
    cm = confusion_matrix(y_true = classes, y_pred = predictions)
    tpr, fpr = check_and_compute_rates(predictions, classes, cm)
        
    return(tpr, fpr)

def get_rates_by_cross_validation(patterns, 
                                  classes, 
                                  d_cluster, 
                                  d_base, 
                                  alpha):
    
    tpr = []
    fpr = []
        
        
    for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state = 12).split(patterns):
        
        x_train, x_test = patterns[train_index], patterns[test_index]
        y_train, y_test = classes[train_index], classes[test_index]
        
        pattern_matrix = Pattern_Matrix()
        clustered_patterns = pattern_matrix.obtain_pc_with_distance(x_train[np.where(y_train==1), :][0], 
                                                                    d_cluster)
        prediction = Prediction()
        pbp_predictions = pbp_prediction(patterns_array=x_test, 
                                         clustered_patterns = clustered_patterns, 
                                         d_base = d_base, alpha = alpha, 
                   outbreak_p_means = prediction.obtain_p_means_with_distance,
                   outbreak_prediction = prediction.predict_with_distance)
        rates = get_rates(predictions = pbp_predictions, classes = y_test)
        tpr.append(rates[0])
        fpr.append(rates[1])
    
    return(np.mean(tpr[tpr!=np.nan]), np.mean(fpr[fpr!=np.nan]))
cdef int m
cdef float xstar
def get_auroc(v, 
              time_series, 
              xstar, 
              decompose,
              decomposition_function, 
              verbose):
    
    tpr = []
    fpr = []
    m, d_cluster,  alpha = v
    patterns = Pattern_Matrix()
    predicting_outbreak = Prediction()        
    variables, classes = obtain_patterns(observation_set = time_series, 
                                         event = xstar, 
                                         m = int(np.round(m)),
                                         decompose = decompose, 
                                         decomposition_function = decomposition_function)    
    for d_base in np.arange(0, 1.1, 0.1):

            rates = get_rates_by_cross_validation(patterns = variables, 
                                   classes = classes, 
                                   d_cluster = d_cluster, 
                                   d_base = d_base, alpha = 1)

            tpr.append(rates[0])
            fpr.append(rates[1])
    interpolate_function = interpolate.interp1d([1] +fpr + [0], [1]+tpr+ [0])
    auroc = - trapz(x = np.arange(0, 1.1, 0.1), y = interpolate_function(np.arange(0, 1.1, 0.1)))
    if verbose: 
        print('Obtain Area of Roc curve of %s'%(-auroc))
    
    return(auroc)

def dual_anneling_optim(time_series, 
                        xstar, 
                        decompose, 
                        decomposition_function,
                        maxfun,
                        verbose, 
                        bounds = [[1, 7], [0.1, 0.5], [0.5, 1.5]]):
    
    
    
    dual_anneling_result = dual_annealing(get_auroc, 
                                          bounds, 
                                          maxfun=maxfun, 
                                          args = (time_series, 
                                                  xstar, decompose, 
                                                  decomposition_function, 
                                                  verbose), 
                                          seed = 12) 
    
    m, d_cluster, alpha = dual_anneling_result.x
    auroc = dual_anneling_result.fun
    result = {'m': int(np.round(m)), 
              'd_cluster': d_cluster, 
              'alpha': alpha, 
              'auroc': -auroc}
    
    return(result)

def get_fitted_parameters(time_series, 
                          xstar, 
                          decompose, 
                          maxfun,
                          verbose, 
                          decomposition_function):    
    
    tpr = []
    fpr = []
    
    
    parameters = dual_anneling_optim(time_series = time_series, 
                                                     xstar = xstar, 
                                                     maxfun = maxfun,
                                                     verbose = verbose, 
                                                     decompose = decompose, 
                                                     decomposition_function = decomposition_function)
    m = parameters['m']
    d_cluster = parameters['d_cluster']
    alpha = parameters['alpha']
    auroc = parameters['auroc']

    variables, classes = obtain_patterns(observation_set = time_series,
                                     event = xstar, m = m, 
                                     decompose = decompose, 
                                     decomposition_function = decomposition_function)            
    
    for d_base in np.arange(0, 1.1, 0.1):
        
        rates = get_rates_by_cross_validation(patterns = variables, 
                                   classes = classes, 
                                   d_cluster = d_cluster, 
                                   d_base = d_base, alpha = 1)

        tpr.append(rates[0])
        fpr.append(rates[1])
                    
    data = pd.DataFrame({'tpr': tpr,
              'fpr': fpr,
              'd_base':np.arange(0, 1.1, 0.1)})        
    parameters = {'m': int(np.round(m)), 
              'd_cluster': d_cluster, 
              'alpha': alpha, 
              'auroc': -auroc}
    
    return(parameters, data)

def get_d_base(data, rate_type, rate_threshold):
        
    if rate_type == 'fpr':        
        while (sum(data['fpr'] > rate_threshold) == len(data['fpr'])):            
            rate_threshold = rate_threshold + 0.01
            
        tpr_given_fpr_min = data['tpr'][data['fpr'] <= rate_threshold]
        d_base_options = data[['d_base', 'fpr']].loc[tpr_given_fpr_min.index[tpr_given_fpr_min == max(tpr_given_fpr_min)]]
        d_base = d_base_options['d_base'].loc[d_base_options.index[d_base_options['fpr'] == min(d_base_options['fpr'])][0]]

    else:
        while (sum(data['tpr'] < rate_threshold) == len(data['tpr'])):            
            rate_threshold = rate_threshold - 0.01

        fpr_given_tpr_max = data['fpr'][data['tpr'] >= rate_threshold]
        d_base_options = data[['d_base', 'tpr']].loc[fpr_given_tpr_max.index[fpr_given_tpr_max == min(fpr_given_tpr_max)]]
        d_base = d_base_options['d_base'].loc[d_base_options.index[d_base_options['tpr'] == max(d_base_options['tpr'])][0]]                                         
    result = {'d_base': d_base, 
              'rate_threshold': rate_threshold}
    return(result)

cdef float acc, f1, precision, recall
def get_method_statiscs(predictions, classes):                        
    
    acc = accuracy_score(y_true = classes,
                         y_pred = predictions)
    cm = confusion_matrix(y_true = classes,
                          y_pred = predictions)

    f1 = f1_score(y_true = classes, 
                  y_pred = predictions, 
                  zero_division = 0)
    precision = precision_score(y_true = classes, 
                                y_pred = predictions, 
                                zero_division = 0)
    recall = recall_score(y_true = classes, 
                          y_pred = predictions,
                          zero_division = 0)
    tpr, fpr = check_and_compute_rates(predictions, 
                                       classes, cm)  
    result = {'accuracy': acc, 'precision': precision, 
              'recall': recall, 'tpr': tpr, 'fpr': fpr, 
              'f1': f1}  
    
    return(result)

def pbp_fit(time_series, xstar, verbose = False, maxfun = 10, decopose = False, train_percentage = 0.7):    
    
    time_series_length = int(len(time_series))
    observation_number = int(time_series_length * train_percentage)
    
    if decopose:
        parameters, data = get_fitted_parameters(time_series = time_series[0:observation_number], 
                                     xstar = xstar, 
                                     decompose = True, 
                                     verbose = verbose,
                                     maxfun = maxfun,
                                     decomposition_function = get_decomposition)
        train_variables, train_classes = obtain_patterns(observation_set = time_series[0:observation_number],
                                     event = xstar, m = parameters['m'], 
                                     decompose = True, 
                                     decomposition_function = get_decomposition)   
    
        test_variables, test_classes = obtain_patterns(observation_set = time_series[(observation_number+1):time_series_length],
                                        event = xstar, m = parameters['m'], 
                                        decompose = True, 
                                        decomposition_function = get_decomposition) 
        
    else:
        parameters, data = get_fitted_parameters(time_series = time_series[0:observation_number], 
                                     xstar = xstar, 
                                     decompose = False, 
                                     verbose = verbose,
                                     maxfun = maxfun,
                                     decomposition_function = get_decomposition)
    
        train_variables, train_classes = obtain_patterns(observation_set = time_series[0:observation_number],
                                        event = xstar, m = parameters['m'], 
                                        decompose = False, 
                                        decomposition_function = get_decomposition)   
        
        test_variables, test_classes = obtain_patterns(observation_set = time_series[(observation_number+1):time_series_length],
                                        event = xstar, m = parameters['m'], 
                                        decompose = False, 
                                        decomposition_function = get_decomposition)  
    
    
    
    result = []
    for rate_threshold, rate_type in zip([0.9, 0.8, 0.2, 0.1], ['tpr', 'tpr', 'fpr' ,'fpr']):
        
        d_base_info = get_d_base(data = data, 
                            rate_type = rate_type, 
                            rate_threshold = rate_threshold)
        d_base = d_base_info['d_base']
        
    
        pattern_matrix = Pattern_Matrix()
        clustered_patterns = pattern_matrix.obtain_pc_with_distance(train_variables[np.where(train_classes==1), :][0], 
                                                                    parameters['d_cluster'])
        
        prediction = Prediction()        
        estimations = pbp_prediction(patterns_array = test_variables, 
                                     clustered_patterns = clustered_patterns,
                                     d_base = d_base, alpha = parameters['alpha'], 
                                     outbreak_p_means = prediction.obtain_p_means_with_distance, 
                                     outbreak_prediction = prediction.predict_with_distance)
        
        metrics = get_method_statiscs(predictions = estimations, 
                                      classes = test_classes)
        acc = metrics['accuracy']
        f1 = metrics['f1']
        precision = metrics['precision']
        recall = metrics['recall']
        tpr = metrics['tpr']
        fpr = metrics['fpr']
                                
        result.append([acc, f1, precision, recall, tpr, fpr, 
                       parameters['m'], parameters['d_cluster'], 
                       parameters['alpha'], parameters['auroc'], 
                       data])

    # Preparing summary dataset
    data_base= pd.DataFrame({'Model': [np.NaN], 
                  'Decomposition': [np.NaN],                  
                  'Criteria': [np.NaN],
                  'Threshold': [np.NaN],
                  'Acc': [np.NaN],
                  'FI-Score': [np.NaN],
                  'Precision': [np.NaN],
                  'Recall': [np.NaN],
                  'Tpr': [np.NaN],
                  'Fpr': [np.NaN],
                  'm': [np.NaN],
                  'd_cluster': [np.NaN],                 
                  'Alpha': [np.NaN],
                  'ROC': [np.NaN],
                  'd_base': [np.NaN],
                  'updated_threshold': [np.NaN],
                  'xstar': [np.NaN], 
                  'split': [np.NaN]})

    for d_base_number, rate_threshold, rate_type in zip([0, 1, 2, 3], [0.9, 0.8, 0.2, 0.1], ['tpr', 'tpr', 'fpr' ,'fpr']):
            
            d_base_threshold_info = get_d_base(result[d_base_number][10], 
                       rate_type = rate_type, 
                       rate_threshold = rate_threshold) 
            
            data1 = pd.DataFrame({'Model': ['Real'], 
                      'Decomposition': ['No'],                  
                      'Criteria': [rate_type],
                      'Threshold': [rate_threshold],
                      'Acc': [result[d_base_number][0]],
                      'FI-Score': [result[d_base_number][1]],
                      'Precision': [result[d_base_number][2]],
                      'Recall': [result[d_base_number][3]],
                      'Tpr': [result[d_base_number][4]],
                      'Fpr': [result[d_base_number][5]],
                      'm': [result[d_base_number][6]],                  
                      'd_cluster': [result[d_base_number][7]],           
                      'Alpha': [result[d_base_number][8]],                  
                      'ROC': [abs(result[d_base_number][9])],
                      'd_base': [d_base_threshold_info['d_base']],
                      'updated_threshold': [d_base_threshold_info['rate_threshold']],          
                      'xstar': [xstar], 
                      'split': [train_percentage]})
            data_base = pd.concat([data_base, data1], axis=0)
    parameters['d_base'] = d_base_threshold_info['d_base']    
    data_base = data_base.dropna()
    return(data_base, xstar, clustered_patterns, parameters) 

def get_ci(outbreak_patterns):
    '''This function obtain the confidence interval based on percentile of the patterns'''            
    
    outbreak_patterns25 = np.quantile(outbreak_patterns, 0.25, axis = 0)
    outbreak_patterns50 = np.mean(outbreak_patterns, axis = 0)
    outbreak_patterns95 = np.quantile(outbreak_patterns, 0.75, axis = 0)
    
    outbreak_patterns_ci = np.transpose(np.stack((outbreak_patterns25,
                                                         outbreak_patterns50, 
                                                         outbreak_patterns95)))
    
    return(outbreak_patterns_ci)
    
def pbp_plot(time_series, clustered_patterns, parameters, 
             decompose = False, xnew = False):
    '''This functions plots the patterns obtained by the pbp method'''  
    # Plotting 
    dimention = clustered_patterns[1][0].shape[1]

    if dimention ==1:
        print('nao quero') 
        fig, ax = plt.subplots(figsize = (15, 9))
        ax.hist(clustered_patterns[1][0], alpha=0.3, 
                label = 'Outbreak pattern', color = '#F8766D')
        if xnew == False:
            print('')        
        else:
            ax.axvline(x=xnew, color='#000000', label='Current pattern', linewidth = 2)            
        ax.legend(prop={'size':15})
    else:  
        fig, ax = plt.subplots(figsize = (15, 9))
        xaxix = np.arange(1, dimention + 1, 1)
        reverse_xaxis = abs(np.sort(-np.arange(1, dimention+1, 1))) 
        n_cluster = len(clustered_patterns[1])    
        iteractions = 1
        for cluster in clustered_patterns[1]:
            outbreak_patterns_ci = get_ci(cluster)        
            if iteractions == n_cluster:
                ax.plot(xaxix, outbreak_patterns_ci[:,1], '#F8766D',
                label = 'Outbreak pattern', linewidth = 2)
                ax.fill_between(xaxix, outbreak_patterns_ci[:,0],
                                outbreak_patterns_ci[:,2], alpha=0.3, 
                                color = '#F8766D')
                ax.xaxis.set_major_locator(plt.MaxNLocator(dimention))
                ax.set_xlabel('Generations before an outbreak', size = 18)
                ax.set_ylabel('Number of individuals', size = 18)
                plt.xticks(reverse_xaxis, xaxix)
            else:                        
                ax.plot(xaxix, outbreak_patterns_ci[:,1], '#F8766D', linewidth = 2)
                ax.fill_between(xaxix, outbreak_patterns_ci[:,0],
                                outbreak_patterns_ci[:,2], alpha=0.3, 
                                color = '#F8766D')  
            iteractions += 1

        if xnew == False:
            print('')        
        else:
            ax.plot(xaxix, xnew, '#000000', label = 'Current pattern', linewidth = 2)    
        ax.legend(prop={'size':15})

    
        return fig