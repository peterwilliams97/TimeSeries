"""

Main processing pipeline for TimeSeries project

Created on 25/07/2010

@author: peter
"""

from __future__ import division
import  copy, random, time, math, optparse, os, re, numpy as NP, scipy as SP, pylab as PL, ols, statistics, csv, run_weka

def logit(x):
    # Protect against overfow in exp(-x)
    if x < -100.0:
        return 0.0
    return 1.0/(1.0+math.exp(-x))

def showArray(name, a):
    """ Display a numpy array """
    print name, ', shape =', a.shape
    print a
    print '--------------------'
    
def getDaysOfWeekToKeep(vector, threshold=0.2):
    """ Given a vector of daily values, determines which days are outliers
        based on threshold * median
        Returns days in week to keep """
    if False:
        average_for_day = []  
        for day in range(7):
            day_vector = [vector[i] for i in range(day, (len(vector)//7)*7, 7)]
            average_for_day.append(getMean(day_vector))
    else:
        average_for_day = [statistics.getMean([vector[i] for i in range(day, (len(vector)//7)*7, 7)]) for day in range(7)]   
    median_day = sorted(average_for_day)[3]
    return [day for day in range(7) if average_for_day[day] >= median_day*threshold]

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

def getTrend(t, y, mask = None):
    """ Find the trend in a time-series y
        With times vector t """
    if mask:
        t = applyMask1D(t, mask)
        y = applyMask1D(y, mask)
    assert(t.shape[0] == y.shape[0])
    mymodel = ols.ols(y,t,'y',['t'])
    return mymodel.b  # return coefficients

def removeTrend1D(trend, t, y, mask):
    """ Remove trend from numpy arrays y vs t with specified mask.
        Returns de-trended data
        NOTE: This must be applied to all data, not just the unmasked part """
    assert(t.shape[0] == y.shape[0])  
    yret = NP.array([y[i] - (trend[0] + t[i] * trend[1]) for i in range(t.shape[0])])
    print "removeTrend1D()", len(mask), t.shape, y.shape, yret.shape
    return y # !@#$ currently de-activated
    return yret

def addTrend1D(trend, t, y, mask):
    """ Add trend from numpy arrays v vs t with specified mask.
        Returns de-trended data """
    assert(t.shape[0] == y.shape[0])  
    return y # !@#$ currently de-activated
    return NP.array([y[i] + (trend[0] + t[i] * trend[1]) for i in range(t.shape[0])])

def timeSeriesToMatrixArray(time_series, masks, max_lag):
    """ Generate Weka format csv file for two time series.
        x_series and y_series which is believed to depend on x_series
        max_lag is number of lags in dependence
        
        !@#$ This is hard with numpy arrays. Use python lists
    """
    print 'timeSeriesToMatrixArray time_series.shape', time_series.shape, 'max_lag', max_lag
    num_rows = time_series.shape[1] - max_lag 
    assert(num_rows >= 1)
    for i in range(2):
        describeNPVector('time_series[%0d]'%i, time_series[i])
    regression_matrix = NP.zeros((num_rows, 2*max_lag + 1), dtype=float)
    regression_mask = NP.zeros((num_rows, 2*max_lag + 1))
    for i in range(max_lag):
        #print 'regression_mask[i,0:2*max_lag+1]', regression_mask[i,0:2*max_lag+1]
        regression_matrix[i,0:max_lag] = time_series[0,i:i+max_lag] 
        regression_matrix[i,max_lag:2*max_lag+1] = time_series[1,i:i+max_lag+1]
        regression_mask[i,0:max_lag] = masks[0][i:i+max_lag] 
        regression_mask[i,max_lag:2*max_lag+1] = masks[1][i:i+max_lag+1] 
        #print 'time_series[0,i:i+max_lag]', time_series[0,i:i+max_lag]
        #print 'time_series[1,i:i+max_lag+1]', time_series[1,i:i+max_lag+1]
        #print 'regression_matrix[i,0:2*max_lag+1]', regression_matrix[i,0:2*max_lag+1]
        #print 'regression_matrix[i]', regression_matrix[i]
        #exit()
        # print regression_matrix[i,:]
    # Normalize regression_matrix to mean 0 and stddev 1
    means   = {'x': NP.zeros(max_lag), 'y': NP.zeros(max_lag), 'z': 0.0}  
    stddevs = {'x': NP.zeros(max_lag), 'y': NP.zeros(max_lag), 'z': 0.0}  
    all_means = NP.mean(regression_matrix, axis=0)  
    all_stddevs = NP.std(regression_matrix, axis=0)    
    means['x'] = all_means[0:max_lag]
    stddevs['x'] = all_stddevs[0:max_lag]
    means['y'] = all_means[max_lag:2*max_lag]
    stddevs['y'] = all_stddevs[max_lag:2*max_lag]
    means['z'] = all_means[2*max_lag]
    stddevs['z'] = all_stddevs[2*max_lag]
   
    #print 'all_means', all_means
    #print 'all_stddevs', all_stddevs
    print 'means', means
    print 'stddevs', stddevs
   
    #exit()
    regression_matrix[2*max_lag:] = (regression_matrix[2*max_lag,:]-means['z'])/stddevs['z']
    return (regression_matrix, regression_mask, means, stddevs)

def timeSeriesToMatrixCsv(regression_matrix_csv, time_series, masks, max_lag):
    """ Convert a 2 row time series into a 
    regression matrix """
    regression_matrix,regression_mask, means, stddevs = timeSeriesToMatrixArray(time_series, masks, max_lag)
    header_x = ['x[%0d]' % i for i in range(-max_lag,0)]
    header_y = ['y[%0d]' % i for i in range(-max_lag,1)]
    header = header_x + header_y
    regression_data = [[str(regression_matrix[i,j]) if regression_mask[i,j] else '?' 
                        for j in range(regression_matrix.shape[1])]
                            for i in range(regression_matrix.shape[0])]
    # Eliminate rows with no output 
    regression_data = [x for x in regression_data if not x[len(x)-1] == '?']
    print regression_data[0]
        
    csv.writeCsv(regression_matrix_csv, regression_data, header)
    return (means, stddevs) 
    
regex_node = re.compile(r'[x,y]\[-?\d+\]')
regex_node_letter = re.compile(r'[x,y]')
regex_node_number = re.compile(r'-?\d+')

def parseNodeName(name): 
    def getCpt(name, regex):
        s = regex.search(name)
        return s.group() if s else ''
    
    if regex_node.search(name):
        letter = getCpt(name, regex_node_letter)
        index = int(getCpt(name, regex_node_number))
        return (letter, index)
    
    print 'parseNodeName(%s) "%s"' % (name, name), 'does not exist'
    raise Exception  # Should never happen
  
          
def applyCoefficients(coefficients, means, stddevs, x, y, N):
    """ Apply regression coefficients to two columns of time series data to predict 
        y value of next time
        x is length N [0,-N-1], y is length N-1 [-1,-N-2]
        For a N row data matrix return a 1D vector of length N """
        
    sigmoids = {}
    for node in coefficients['Sigmoid']:
        val = -node['threshold']  
        weights = node['attribs']  
        for k in weights.keys():
            #print k
            letter, i = parseNodeName(k)
            #print 'k =', k, ', letter =', letter, ', i =', i, ', i+N =', i+N 
            assert(i+N >= 0)
            z = x if letter == 'x' else y 
            val = val + weights[k] * (z[i+N]-means[letter][i])/stddevs[letter][i]
        #print "int(node['number']) = ", int(node['number'])
        #print 'val =', val
        sigmoids[int(node['number'])] = logit(val)
  
    node = coefficients['Linear'][0]
    val = -node['threshold'] 
    weights = node['attribs']     
    assert(len(sigmoids) == len(weights))
   # print 'sigmoids = ', sigmoids
    #print 'weights = ', weights
    for k in weights.keys():
        # print k
        val = val + weights[k]*sigmoids[int(k)]
    return logit(val)*stddevs['z'] + means['z']
   
    
 
def predictTimeSeries(coefficients, means, stddevs, t, x, y, n_start, max_lag, mask):
    """ Make predictions of numpy array time series x,y vs t with specified mask.
        Returns predicted y values
        x: 1 2 3 4 5 6 7 8 9
        y: a b c 
    """
    print 't', t.shape
    print 'x', x.shape
    print 'y', y.shape
    assert(t.shape[0] == y.shape[0])  
    assert(t.shape[0] == x.shape[0])  
    yret = NP.zeros(y.shape[0])
    yret[:] = y[:]
    num_predictions = y.shape[0] - n_start
    for i in range(n_start, yret.shape[0]):
        xs = x[i-max_lag:i+1]
        ys = yret[i-max_lag:i]
        yret[i] = applyCoefficients(coefficients, means, stddevs,  xs, ys, max_lag)
        print i, yret[i]
    #exit() # !@#$
    return yret 

def describeNPVector(name, x):
    """ Describe a 1d numpy array """
    print '   ', name, x.shape[0], NP.mean(x)
    
def describeNPArray(name, x):
    """ Describe a 2d numpy array """
    print ' ', name, x.shape
    for i in range(x.shape[0]):
        describeNPVector('%s[%d]' % (name, i), x[i,:])
    
def analyzeTimeSeries(filename, max_lag, fraction_training):
    """ Main function. 
        Analyze time series in 'filename' (assumed to be a CSV for now)
        Create model with up to mag_lag lags
        Use the first fraction_training of data for training and the 
        remainder for testing
    """
    
    base_name = os.path.splitext(filename)[0]
    regression_matrix_csv = base_name + '.regression.csv'
    results_filename = base_name + '.results' 
    model_filename = base_name + '.model' 
    prediction_matrix_csv = base_name + '.prediction.csv'
    
    """ Assume input file is a CSV with a header row """
    time_series_data, header = csv.readCsvFloat2(filename, True)
    
    """ Assume a weekly pattern """
    number_training = (int(float(len(time_series_data))*fraction_training)//7)*7
    
    print 'number_training', number_training, 'fraction_training', fraction_training,'len(time_series_data)',len(time_series_data)
    assert(number_training > max_lag)
    
    time_series = NP.transpose(NP.array(time_series_data))
    describeNPArray('time_series', time_series)
        
    training_time_series = NP.transpose(NP.array(time_series_data[:number_training]))
    print 'training_time_series.shape', training_time_series.shape
    
    t = NP.arange(time_series.shape[1])
    training_t = NP.arange(training_time_series.shape[1])
    
    num_series = training_time_series.shape[0]
    num_rows = training_time_series.shape[1]
    
    days_to_keep = [getDaysOfWeekToKeep(training_time_series[i,:]) for i in range(num_series)]
    
    masks = [getDaysOfWeekMask(days_to_keep[i], time_series.shape[1]) for i in range(num_series)]
    training_masks = [getDaysOfWeekMask(days_to_keep[i], num_rows) for i in range(num_series)]
    
    trends = [getTrend(training_t, training_time_series[i,:], training_masks[i]) for i in range(num_series)]
    
    x = [removeTrend1D(trends[i], training_t, training_time_series[i], training_masks[i]) for i in range(num_series)]
    for i in range(num_series):
        describeNPVector('x[%0d]'%i, x[i])
    detrended_training_time_series = NP.zeros([num_series, x[0].shape[0]])
    print 'detrended_training_time_series.shape', detrended_training_time_series.shape
    for i in range(num_series):
        print 'x[%0d].shape'%i, x[i].shape
        detrended_training_time_series[i,:] = x[i]
    print 'detrended_training_time_series.shape', detrended_training_time_series.shape
    # filtered_time_series = NP.vstack([filterDaysOfWeek(training_time_series[i,:], days_to_keep[i]) for i in range(num_series)])
    # print 'filtered_time_series.shape', filtered_time_series.shape
   
    for i in range(num_series):
        describeNPVector('detrended_training_time_series[%0d]'%i, detrended_training_time_series[i])
        
    means, stddevs = timeSeriesToMatrixCsv(regression_matrix_csv, detrended_training_time_series, training_masks, max_lag)
    print 'means', means
    print 'stddevs', stddevs
    run_weka.runMLPTrain(regression_matrix_csv, results_filename, model_filename, True, '-H 4')
    coefficients = run_weka.getCoefficients(results_filename)
   
    print '--------------------------------------------'
    print 'coefficients', len(coefficients)
    print coefficients
    print '--------------------------------------------'
    print 'means', len(means)
    print means
    print '--------------------------------------------'
    print 'stddevs', len(stddevs)
    print stddevs
    print '--------------------------------------------'
    #exit()
    detrended_full_x = [removeTrend1D(trends[i], t, time_series[i], masks[i]) for i in range(num_series)]
    detrended_time_series = NP.zeros([num_series, detrended_full_x[0].shape[0]])
    print 'detrended_time_series.shape', detrended_time_series.shape
    for i in range(num_series):
        print 'full_x[%0d].shape'%i, detrended_full_x[i].shape
    detrended_predictions = predictTimeSeries(coefficients, means, stddevs, t, detrended_full_x[0], detrended_full_x[1], number_training, max_lag, masks)
    predictions = addTrend1D(trends[1], t, detrended_predictions, masks[1]) 
    print '--------------------------------------------'
    print 'predictions =', predictions.shape
    # print predictions
    full_x = [NP.array(time_series[i]) for i in range(num_series)]
    
    print 't.shape', t.shape
    print 'full_x[0].shape', full_x[0].shape
    print 'full_x[1].shape', full_x[1].shape
    print 'predictions.shape', predictions.shape
    
    predicted_time_series = NP.vstack([t, full_x[0], full_x[1], predictions])
    
    print 'predicted_time_series.shape', predicted_time_series.shape
    # retrend !@#$\\
    prediction_header = ['t', 'x', 'y', 'y_pred']
    predicted_time_series_data = [[str(predicted_time_series[i,j]) 
                                    for i in range(predicted_time_series.shape[0])]
                                        for j in range(predicted_time_series.shape[1])]
                            
    csv.writeCsv(prediction_matrix_csv, predicted_time_series_data, prediction_header)
    
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
    PL.xlabel('Time (days)')
    PL.ylabel('Downloads')
    PL.title('Dowlnoads over time')
    PL.show()   
    
def test3():
    coefficients = run_weka.getCoefficients(r'C:\dev\exercises\time_series_purchases_99_other_001_lag_05.results') 
    for i in range(len(coefficients)):
        print i, ':', coefficients[i]
        
def test4():
    for i in range(10):
        for sgn in [-1, 1]:
            x = float(sgn) * 10.0**float(i)
            y = logit(x)
            print 'logit(%.2f) = %.6f' % (x,y)
    for i in range(99,-1,-1):
        for sgn in [-1, 1]:
            x = float(sgn*i)/100.0
            y = logit(x)
            #print 'logit(%.2f) = %.6f' % (x,y)
    for x in [-4000.0, -4318.30612305]:
        print x
        y = logit(x)
        print 'logit(%.2f) = %.6f' % (x,y)
        
if __name__ == '__main__':
    test4()