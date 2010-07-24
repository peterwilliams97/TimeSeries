Time Series Analysis
--------------------

0. Parse config and command line parameters

1. Read data. 
	Possibly handle a variety of formats
	Convert to numpy array
	
2. Select training set	
	
3. Remove outliers
	Create a rule.
	Rule+data =>mask.
	Leave data intact

4. Remove trend
	Linear interpolation of training data
	
5. Fit cleaned training data
	MLP 
	10 fold CV
	
6. This gives model
	data =[outlier rule]=> mask
	masked data =[de-trend]=> de-trended data  (do not recalculate trend)
	apply model to de-trended data => predicted de-trended data
	re-trend data => predicted masked data
	unmask => data with gaps where mask was

6. Test model on test data (=input data - training data)
	Apply model to test data
	Compare predictions to actual on unmasked items
	
