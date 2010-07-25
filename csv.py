"""
manipulate .csv files
Clean advertisement detection trainging set file

Peter
16/05/2010
"""
import copy,os,time
from math import *
from operator import itemgetter

def validateMatrix(matrix):
    "Check that all rows in matrix and same length"
    assert(len(matrix) > 0)
    assert(len(matrix[0]) > 0)
    for i in range(1, len(matrix)):
        assert(len(matrix[i]) == len(matrix[0]))
        
def validateMatrix2(matrix):
    "Check that all rows in matrix and same length and non-empty"
    validateMatrix(matrix)
    for i,row in enumerate(matrix):
        for j,val in enumerate(row):
            if len(val) == 0:
                print 'empty cell', i, j        
  
def readCsvRaw(filename): 
    "Reads a CSV file into a 2d array"
    lines = file(filename).read().strip().split('\n')
    entries = [[e for e in l.strip().split(',')] for l in lines]
    print 'readCsvRaw:', filename, len(entries), len(entries[0])
    validateMatrix(entries)
    return entries

def readCsvFloat2(filename, has_header): 
    "Reads a CSV file into a header and 2d array of float"
    header = None
    entries = readCsvRaw(filename)
    if has_header:
        header = entries[0]
        matrix = [[float(e) for e in row] for row in entries[1:]]
        print 'readCsvFloat:', filename, len(entries[1:]), len(entries[1])
    else:
        matrix = [[float(e) for e in row] for row in entries]
    return (matrix, header)    

def readCsvFloat(filename): 
    "Reads a CSV file into a 2d array of float"
    matrix, header = readCsvFloat2(filename, False)
    return matrix

def writeCsv(filename, in_matrix, header = None):
    "Writes a 2d array to a CSV file"
    matrix = [header] + in_matrix if header else in_matrix
    print 'writeCsv:', filename, len(matrix), len(matrix[0])
    file(filename, 'w').write('\n'.join(map(lambda row: ','.join(map(str,row)), matrix)) + '\n')
    
def modifyCsvRaw(in_filename,out_filename,modify_func):
    "Read in_filename into a 2d array, apply modify_func to it and write result to out_filename"
    in_entries = readCsvRaw(in_filename)
    out_entries = modify_func(in_entries)
    writeCsv(out_filename,out_entries)    
    
def swapMatrixColumn(matrix, i, j):
    n = len(matrix[0])
    if i < 0: i = n + i
    if j < 0: j = n + j
    if i > j: i,j = j,i
    for v in matrix:
        x = v[i]
        for k in range(i+1, j+1):
            v[k-1] = v[k]
        v[j] = x
"""
 height: continuous. | possibly missing
   width: continuous.  | possibly missing
   aratio: continuous. | possibly missing
   local: 0,1.
   | 457 features from url terms, each of the form "url*term1+term2...";
   | for example:
   url*images+buttons: 0,1.
     ...
   | 495 features from origurl terms, in same form; for example:
   origurl*labyrinth: 0,1.
     ...
   | 472 features from ancurl terms, in same form; for example:
   ancurl*search+direct: 0,1.
     ...
   | 111 features from alt terms, in same form; for example:
   alt*your: 0,1.
     ...
   | 19 features from caption terms
   caption*and: 0,1.
     ...
"""       

def makeHeader():
    "Make a header row based on the above comments"
    h = ['' for i in range(1559)]
    h[0] = 'height'
    h[1] = 'width'
    h[2] = 'aratio'
    h[3] = 'local'
    n = 4
    def addRange(n, span, prefix):
        for i in range(span):
            h[n+i] = prefix + '%03d' % (i+1)
        print n, span, prefix, n+span    
        return n + span 
    n = addRange(n, 457, 'url')
    n = addRange(n, 495, 'org')
    n = addRange(n, 472, 'anc')
    n = addRange(n, 111, 'alt')
    n = addRange(n, 19,  'cap')
    assert(n == 1558)
    h[1558] = 'Advert'
    return h

def isMissingValue(e):
    "User defined function for detecting missing value"
    return e.strip() == '?'

def replaceMissingValues(matrix):
    "Replace missing values in a 2d matrix with average or mode"
    width = len(matrix[0])
    height = len(matrix)
    h = matrix[0]
    for i in range(width):
        num_missing = 0
        for v in matrix[1:]:
            if isMissingValue(v[i]):
                num_missing = num_missing +1
        if num_missing > 0:
            frac_missing = float(num_missing)/float(height)
            head = '"' + h[i] + '"'
            print 'column', '%3d'%i, '%3d'%num_missing, '%.2f'%frac_missing, '%8s'%head, 
            uniques = []
            for v in matrix[1:]:
                if not v[i] in uniques:
                    uniques.append(v[i])
            number_each = [0 for j in range(len(uniques))] 
            for j in range(len(uniques)):
                number_each[j] = 0        
                for v in matrix[1:]:
                    if uniques[j] == v[i]:
                        number_each[j] = number_each[j] + 1
            if len(uniques) <= 3:  # if there a few values then replace wih mode
                j = max(enumerate(number_each), key=itemgetter(1))[0]
                replacement = uniques[j]
                assert(not isMissingValue(v[i]))
            else:  # if there are many values then replace with average
                remaining = [float(v[i]) for v in matrix[1:] if not isMissingValue(v[i])]
                replacement = sum(remaining)/float(len(remaining))
            print 'replacement', replacement  
            for v in matrix[1:]: 
                if isMissingValue(v[i]):
                    v[i] = replacement          
    
#
# The data we are working on
#           
# Data directory
data_dir = 'C:\\dev\\5167assigment1'           
# Input data - Don't touch this    
raw_name = os.path.join(data_dir,'ad1.csv')
# Input data with header. Needs to be generated once
headered_name = os.path.join(data_dir,'ad1_header.csv')
#Input data with header and pre-processing
headered_name_pp = os.path.join(data_dir,'ad1_header_pp.csv')   
# PCA on headered_name_pp
headered_name_pca = headered_name_pp + '.pca.csv'  
# PCA data normalized to stdev == 1
headered_name_pca_norm = os.path.join(data_dir,'ad1_header.pp.pca.norm.csv')            
# PCA data normalized to stdev == 1 by correlation with outcome
headered_name_pca_corr = os.path.join(data_dir,'ad1_header.pp.pca.norm.corr_order.csv')  

def makePath(name):
    return os.path.join(data_dir, name)  

def makeCsvPath(name):
    return makePath(name + '.csv')    

def makeTempPath(base_name):
    count_fn = os.path.join(data_dir, 'temp', 'count.txt') 
    count = 0
    num_retries = 10
    
    for i in range(num_retries):
        try:
            contents = file(count_fn).read().strip()
            if len(contents) > 0:
                count = int(contents)
            break
        except IOError:
            time.sleep(0.1)
    
    try:
        os.mkdir(os.path.join(data_dir, 'temp'))
    except WindowsError:
        pass
    
    for i in range(num_retries):
        try:
            file(count_fn, 'w').write(str(count+1))
            break
        except IOError:
            time.sleep(0.11)
            
    return os.path.join(data_dir, 'temp', base_name + ('%06d' % count))      
     
def prepareData():
    "Prepare data by adding a header row and replacing missing values"
    header = makeHeader()
    data = readCsvRaw(raw_name)
    
    hdata = [header] + data
    assert(len(hdata)==len(data)+1)
    validateMatrix(hdata)

    #swapMatrixColumn(data, 3, -1)
    writeCsv(headered_name, hdata)
    h2data = readCsvRaw(headered_name)
    
    replaceMissingValues(hdata)
    writeCsv(headered_name_pp, hdata)
    
    
if __name__ == '__main__':
    prepareData()    
   