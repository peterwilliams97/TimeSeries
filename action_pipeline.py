"""

Main processing pipeline for TimeSeries project

Created on 25/07/2010

@author: peter
"""

from __future__ import division
import  copy, random, numpy as NP, scipy as SP, pylab as PL, ols, time, optparse, os, csv, run_weka

def showArray(name, a):
    """ Display a numpy array """
    print name, ', shape =', a.shape
    print a
    print '--------------------'
    
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

def getDaysOfWeekMask(days_to_keep, length):
    """ Make a mask based on days of the week """
    return [i % 7 in days_to_keep for i in range(length)]

def applyMask1D(vector, mask):
    """ Apply a mask to a 1d numpy array """
    assert(vector.shape[0] == len(mask))
    number_visible = sum([1 if x else 0 for x in mask])
    #showArray('applyMask1D', vector)
    y = NP.zeros(number_visible)
    count = 0
    for i in range(len(mask)):
        if mask[i]:
            y[count] = vector[i]
            #print i, count, y[count]
            count = count + 1
    return y

def getTrend(t, y):
    """ Find the trend in a time-series y
        With times vector t """
   # t = NP.arange(vector.shape[0])
    assert(t.shape[0] == y.shape[0])
    mymodel = ols.ols(y,t,'y',['t'])
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
 
def test1():
    vector_full = NP.array([1.0, 2.5, 2.8, 4.1, 5.1, 5.9, 6.9, 8.1])
    vector = vector_full[:-2]
    t =  NP.arange(vector.shape[0])
    showArray('t', t) 
    showArray('vector', vector)    
    
    mask = [True] * vector.shape[0]
    mask[2] = False
    print 'mask', len(mask), mask
   
    masked_vector = applyMask1D(vector, mask)
    masked_t = applyMask1D(vector, mask)
    trend = getTrend(t, vector)
    print trend
    for i in range(masked_t.shape[0]):
        v_pred = trend[0] + masked_t[i] * trend[1]
        print i, masked_vector[i], v_pred, v_pred - masked_vector[i]
    
    predicted = NP.array([trend[0] + i * trend[1] for i in range(masked_vector.shape[0])])
    corrected = NP.array([masked_vector[i] - predicted[i] for i in range(masked_vector.shape[0])])
    masked_s = NP.transpose(NP.vstack([masked_vector, predicted, corrected]))
    
    showArray('masked_t', masked_t) 
    showArray('masked_s', masked_s)    
    # the main axes is subplot(111) by default
    PL.plot(masked_t, masked_s)
    s_range = PL.amax(masked_s) - PL.amin(masked_s)
    axis([PL.amin(masked_t), PL.amax(masked_t), PL.amin(masked_s) - s_range*0.1, PL.amax(masked_s) + s_range*0.1 ])
    xlabel('time (days)')
    ylabel('downloads')
    title('Dowloads over time')
    show()
    
def test2():
    number_samples = 300
    days_to_keep = [2,3,4,5,6]
    vector_full = NP.array([2.0 + i * 10.0/number_samples  + random.uniform(-.5, .5) for i in range(number_samples)])
    mask_full = getDaysOfWeekMask(days_to_keep, vector_full.shape[0])

    vector = vector_full[:int(vector_full.shape[0]*0.8)]
    t = NP.arange(vector.shape[0])
    showArray('t', t) 
    showArray('vector', vector)    
    
    mask = getDaysOfWeekMask(days_to_keep, vector.shape[0])
    print 'mask', len(mask), mask
   
    masked_t = applyMask1D(t, mask)
    masked_vector = applyMask1D(vector, mask)
    showArray('masked_t', masked_t) 
    showArray('masked_vector', masked_vector) 
    
    trend = getTrend(t, vector)
    print trend
    for i in range(masked_t.shape[0]):
        v_pred = trend[0] + masked_t[i] * trend[1]
        print masked_t[i], masked_vector[i], v_pred, v_pred - masked_vector[i]
    
    predicted = NP.array([trend[0] + masked_t[i] * trend[1] for i in range(masked_vector.shape[0])])
    corrected = NP.array([masked_vector[i] - predicted[i] for i in range(masked_vector.shape[0])])
    masked_s = NP.transpose(NP.vstack([masked_vector, predicted, corrected]))
    
    showArray('masked_t', masked_t) 
    showArray('masked_s', masked_s)    
    # the main axes is subplot(111) by default
    PL.plot(masked_t, masked_s)
    s_range = PL.amax(masked_s) - PL.amin(masked_s)
    PL.axis([PL.amin(masked_t), PL.amax(masked_t), PL.amin(masked_s) - s_range*0.1, PL.amax(masked_s) + s_range*0.1 ])
    PL.xlabel('time (days)')
    PL.ylabel('downloads')
    PL.title('Dowloads over time')
    PL.show()    
        
if __name__ == '__main__':
    test2()