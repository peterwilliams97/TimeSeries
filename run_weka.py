"""
A set of tools for running  the Weka MLP  and analyzing its output
The Weka MLP is weka.classifiers.functions.MultilayerPerceptron
"""
import shlex, subprocess, os, time, random, copy, shutil, csv

def dumpEnv():
	""" Dump all environment variables """
	for param in os.environ.keys():
		print "%20s %s" % (param,os.environ[param])
		
def checkExists(title, filename):
	""" Check that filename exists """	
	if not os.path.exists(filename):
		print title, filename, 'does not exist'
		exit()
			
def getAccuracy(filename):	
	""" Extract the accuracy from the stdout of a Weka classifer saved in a file 
	    called filename """
	results = file(filename,'r').read().strip().split('\n')
	found_cv = False
	for line in results:
		if line.find('Stratified cross-validation') >= 0:
			found_cv = True
		if found_cv:
			if line.find('Correctly Classified Instances') >= 0:
				terms = [s.strip() for s in line.split(' ') if not s == '']
				if False:
					for i, s in enumerate(terms):
						print i, ':', s
				accuracy = float(terms[4])
	return accuracy

cv_keys = [
	'Correlation coefficient',     
	'Mean absolute error',
	'Root mean squared error',
	'Relative absolute error',
	'Root relative squared error',
	'Total Number of Instances'
]   
def parseResultsLine(line):
	for key in cv_keys:
		start = line.find(key)
		if key >= 0:
			val = line[len(key):].strip()
			return (key,val)
	return None

def parseResults(filename):
	""" Parse results from the stdout of a Weka regression saved in a file 
	    called filename """
	result_lines = file(filename,'r').read().strip().split('\n')
	found_cv = False
	results = {}
	for line in result_lines:
		if line.find('Cross-validation') >= 0:
			found_cv = True
		if found_cv:
			key_val = parseResultsLine(line)
			if key_val:
				results[key_val[0]] = key_val[1]
	return results


def getPredictionsClassification(filename):	
	""" Extract the Weka prediction from results stored in filename """
	checkExists('Predictions file', filename)
	prediction_file = file(filename, 'r').read().strip().split('\n')
	found_header = False
	results = []
	for line in prediction_file:
		if found_header:
			terms = [s.strip() for s in line.split(' ') if not s == '']
			inst = int(terms[0])
			actual = terms[1]
			predicted = terms[2]
			if len(terms) > 4:
				error = True
				prediction = float(terms[4])
			else:
				error = False
				prediction = float(terms[3])
			r = {'inst':inst, 'actual':actual, 'predicted':predicted, 'error':error, 'prediction':prediction}
			if False:
				if r['error']:
					print r
			assert(r['error'] == (r['actual'] != r['predicted']))
			results.append(r)
		elif line.find('error prediction') >= 0:
			found_header = True
	return results

def getPredictionsRegression(filename):	
	""" Extract the Weka prediction from results stored in filename """
	checkExists('Predictions file', filename)
	prediction_file = file(filename, 'r').read().strip().split('\n')
	found_header = False
	results = []
	for line in prediction_file:
		if found_header:
			terms = [s.strip() for s in line.split(' ') if not s == '']
			inst = int(terms[0])
			actual = float(terms[1])
			predicted = float(terms[2])
			error = float(terms[3])
			r = {'inst':inst, 'actual':actual, 'predicted':predicted, 'error':error }
			results.append(r)
		elif line.find('inst#') >= 0:
			found_header = True
	return results

def getCoefficients(filename):	
	""" Extract the Weka prediction from results stored in filename """
	checkExists('Predictions file', filename)
	coeff_file = [x for x in file(filename, 'r').read().strip().split('\n') if len(x) > 0]
	all_nodes = []
	state = 0
	for line in coeff_file:
		parts = [x for x in line.strip().split(' ') if len(x) > 0]
		#print parts
		if parts[0] == 'Class':
			break
		if len(parts) > 1 and parts[1] == 'Node':
			assert(state == 0 or state == 3)
			node = {'type':parts[0], 'number':int(parts[2])}
			all_nodes.append(node)
			state = 1
		elif parts[0] == 'Inputs':
			assert(state == 1)
			state = 2
		elif parts[0] == 'Threshold':
			assert(state == 2)
			state = 3
			node['threshold'] = float(parts[1])
			node['attribs'] = {}
		elif state == 3:
			key = parts[1]
			val = float(parts[2])
			#print key, val
			node['attribs'][key] = val
	nodes = {}
	for type in ['Linear', 'Sigmoid']:
		nodes[type] = [n for n in all_nodes if n['type'] == type]
	return nodes

		
""" 
	You need to the environment variable 'WEKA_ROOT' to the location of 
	the Weka installation on your computer. 
"""

weka_root = 'undefined'
weka_jar = 'undefined'
weka_mlp = 'weka.classifiers.functions.MultilayerPerceptron'
# mlp_opts = ' -H "a,2" -x 4'
mlp_opts = '-H "a" -x 4'
# http://old.nabble.com/WEKA-CLI:-Problems-with-flags-td23670055.html
weka_cost = 'weka.classifiers.meta.CostSensitiveClassifier -cost-matrix "[0.0 1.0; 10.0 0.0]" -S 1 -W '
#def init():
#	global weka_root, weka_jar
try:
	weka_root = os.environ['WEKA_ROOT']	
except:
	print 'You must create an environment variable WEKA_ROOT and set it to your Weka path'
	exit()
weka_jar = os.path.join(weka_root, 'weka.jar')
weka_mlp = 'weka.classifiers.functions.MultilayerPerceptron'
# mlp_opts = ' -H "a,2" -x 4'
mlp_opts = '-H "a" -x 4'
# http://old.nabble.com/WEKA-CLI:-Problems-with-flags-td23670055.html
weka_cost = 'weka.classifiers.meta.CostSensitiveClassifier -cost-matrix "[0.0 1.0; 10.0 0.0]" -S 1 -W '
	
def outnameToModelname(out_fn):	
	base_name = os.splitext(out_fn)[0]
	return base_name + '.model'
	
def runWekaClass(out_fn, weka_cmds):
	""" Run the Weka class weka_cmds
		Write data to file out_fn
		See http://docs.python.org/library/subprocess.html
	"""
	checkExists('Weka jar',  weka_jar) 
	out = open(out_fn, 'w')	
	err = open('stderr.txt', 'w')		
	cmd = 'java -cp ' + weka_jar  + ' ' + weka_cmds
	print cmd
	p = subprocess.Popen(cmd, stdout=out, stderr=err)	
	if not p == 0:
		err.close()
		print file('stderr.txt').read()
	return p.wait()
		
def runMLPTrain(data_filename, results_filename, model_filename, is_regression, opts = mlp_opts):
	""" Run the Weka MultilayerPerceptron with options mlp_opts on the data in data_filename
		Write data to file results_filename
	"""
	checkExists('Data file', data_filename) 
	retcode = runWekaClass(results_filename, weka_mlp + ' ' + opts + ' -t ' + data_filename + ' -d ' + model_filename) 
	return parseResults(results_filename) if is_regression else getAccuracy(results_filename)

def runMLPPredict(data_filename, model_filename, predictions_filename):
	""" Run the Weka MultilayerPerceptron with model model_filename on
		data_filename.
		Write data to predictions_filename
	"""
	checkExists('Data file', data_filename) 
	checkExists('Model file', model_filename) 
	runWekaClass(predictions_filename, weka_mlp + ' -p 0 -T ' + data_filename + ' -l ' + model_filename) 

def testMatrixMLP(matrix, columns, opts = mlp_opts):
	""" Run MLP on attributes with index in columns """
	c_x = columns + [-1]      # include outcome
	sub_matrix = [[row[i] for i in c_x] for row in matrix]
	temp_base = csv.makeTempPath('subset'+('%03d'%len(columns))+'_')
	temp_csv = temp_base + '.csv'
	temp_results = temp_base + '.results'
	csv.writeCsv(temp_csv, sub_matrix)
	accuracy = runMLPTrain(temp_csv, temp_results, opts)
	return (accuracy, temp_csv, temp_results)

def mapToWekaOptions(options_map):
	"""	Convert a map with keys and values corresponding to Weka options 
		to a Weka options string
		e.g. {'M':0.5 'L':0.3, 'H':7 'x':5} => '-m 0.5 -L -0.3 -H 7 -x 5'
	"""
	option_strings = ['-' + k + ' ' + str(options_map[k]) for k in options_map.keys()]
	return ' '.join(option_strings)

def spaceSeparatedLine(arr):
	return ' '.join(map(str,arr))

def makeWekaOptions(learning_rate, momentum, number_hidden, num_cv, costs = None):
	""" Return Weka option string for specified values """
	options_map = {'M':momentum, 'L':learning_rate, 'H':number_hidden, 'x':num_cv}
	if costs:
		cost_matrix_path = csv.makeTempPath('cost') + '.cost'
		options_map['m'] = cost_matrix_path
		cost_matrix = ['%% Rows	Columns',
					   spaceSeparatedLine([2,2]),
					   '%% Matrix elements',
					   spaceSeparatedLine([0.0,  costs['True']]),
					   spaceSeparatedLine([costs['False'], 0.0])]	
		file(cost_matrix_path, 'w').write('\n'.join(cost_matrix))
	return mapToWekaOptions(options_map)
	
