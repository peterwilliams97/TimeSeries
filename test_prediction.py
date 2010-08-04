"""
Create many time series, run several fitting algorithms on them and take best algorithm

Created on 25/07/2010

@author: peter
"""
from __future__ import division
import os, action_pipeline, make_time_series

def makeTestPath(filename):
    """ Using a hardwired test directory for now """
    return os.path.join(r'C:\dev\exercises', filename)

def makeTestFiles(create_files):
    """ Make a set of test files for evaluating time series prediction  algorithms 
        If create_files is True then create the files
    """
    number_days = 2 * 365
    downloads_per_day = 1000
    test_file_names = []
    file_number = 0
    params_list = []
    
    for other_purchase_ratio_pc in (10,20,50,100,200):
        for purchases_per_download_pc in (80, 40, 20):
            for purchase_max_lag in (2,4,7,14,28):
                params_list.append((purchases_per_download_pc,other_purchase_ratio_pc,purchase_max_lag))
    
    params_list = [(99,1,5)] 
    print len(params_list), 'test files'    
        
    for purchases_per_download_pc,other_purchase_ratio_pc,purchase_max_lag in params_list:
        filename = 'time_series_purchases_%02d_other_%03d_lag_%02d.csv' \
            % (purchases_per_download_pc, other_purchase_ratio_pc, purchase_max_lag)
        purchases_per_download = purchases_per_download_pc/100.0
        other_purchase_ratio = other_purchase_ratio_pc/100.0
        if create_files:
            make_time_series.makeTimeSeriesCsv(makeTestPath(filename), purchase_max_lag, number_days, downloads_per_day, \
                          purchases_per_download, other_purchase_ratio * downloads_per_day)
        test_file_names.append(filename)
    return test_file_names
                
if __name__ == '__main__':
    create_files = True
    test_file_names = makeTestFiles(create_files)
    print len(test_file_names), 'test files'
    max_lag = 20
    fraction_training = 0.8
    for filename in test_file_names:
        if False:
            time_series.findAutoCorrelations(makeTestPath(filename), max_lag, fraction_training)
        if True:
            action_pipeline.analyzeTimeSeries(makeTestPath(filename), max_lag, fraction_training)
            exit()