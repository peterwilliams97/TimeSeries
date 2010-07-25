"""

Main processing pipeline for TimeSeries project

Created on 25/07/2010

@author: peter
"""

from __future__ import division
import  copy as CP, numpy as NP, scipy as SP, pylab as PL, ols, time, optparse, os, csv, run_weka
from pylab import *

def getDaysOfWeekToKeep(vector, threshold):
    """ Given a vector of daily values, determines which days are outliers
        based on threshold * median
        Returns days in week to keep """
    if False:
        average_for_day = []  
        for day in range(7):
            day_vector = [vector[i] for i in range(day, (len(vector)//7)*7, 7)]
            average_for_day.append(getMean(day_vector))
    else:
        average_for_day = [getMean([vector[i] for i in range(day, (len(vector)//7)*7, 7)]) for day in range(7)]   
    median_day = sorted(average_for_day)[3]
    return [day for day in range(7) if average_for_day[day] >= median_day *threshold]

def getTrend(vector):
    """ Find the trend in a time-series vector
        Assume all elements are equal time apart """
    t = NP.arange(vector.shape[0])
    mymodel = ols.ols(vector,t,'y',['t'])
    return mymodel.b  # return coefficients

def analyzeTimeSeries(filename, max_lag, fraction_training):
    """ Main function. 
        Analyze time series in 'filename' (assumed to be a CSV for now)
        Create model with up to mag_lag lags
        Use the first fraction_training of data for training and the 
        remainder for testing
    """
    
    """ Assume input file is a CSV with a header row """
    time_series_data, header = csv.readCsvFloat2(filename, True)
    
    """ Assume a weekly pattern """
    number_training = (int(float(len(time_series_data))*fraction_training)//7)*7
    
    print 'number_training', number_training, 'fraction_training', fraction_training,'len(time_series_data)',len(time_series_data)
    assert(number_training > max_lag)
    
    time_series = NP.transpose(NP.array(time_series_data))
    print 'time_series.shape', time_series.shape
    
    training_time_series = NP.transpose(NP.array(time_series_data[:number_training]))
    print 'training_time_series.shape', training_time_series.shape
    
    num_series = training_time_series.shape[0]
    days_to_keep = [getDaysOfWeekToKeep(training_time_series[i,:]) for i in range(num_series)]
    filtered_time_series = NP.vstack([filterDaysOfWeek(training_time_series[i,:], days_to_keep[i]) for i in range(num_series)])
    print 'filtered_time_series.shape', filtered_time_series.shape
    
    trends = [getTrend(training_time_series[i,:]) for i in range(num_series)]
    timeSeriesToMatrixCsv(regression_matrix_csv, training_time_series, max_lag)
    run_weka.runMLPTrain(regression_matrix_csv, results_filename, model_filename, True)
 

def showArray(name, a):
    """ Display a numpy array """
    print name, ', shape =', a.shape
    print a
    print '--------------------'
    
if __name__ == '__main__':
    vector = NP.array([1.0, 2.5, 2.8, 4.1, 5.0])
    trend = getTrend(vector)
    print trend
    for i in range(vector.shape[0]):
        v_pred = trend[0] + i * trend[1]
        print i, vector[i], v_pred, v_pred - vector[i]
    
    t = NP.arange(vector.shape[0])
    predicted = NP.array([trend[0] + i * trend[1] for i in range(vector.shape[0])])
    s = NP.transpose(NP.vstack([vector, predicted]))
    
    showArray('t', t) 
    showArray('s', s)    
    # the main axes is subplot(111) by default
    plot(t, s)
    axis([amin(t), amax(t), 0, amax(s) ])
    xlabel('time (days)')
    ylabel('downloads')
    title('Dowloads over time')
    show()
    
