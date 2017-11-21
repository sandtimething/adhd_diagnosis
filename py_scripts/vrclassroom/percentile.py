"""
	calcPercentile
	
	Calculate Percentile of test caseids based on train caseids, and write to db. 
	
	Origin from `percentile_witherrorhandling_2.py`. - Tom 08/31/2017

"""

import numpy as np
import pdb
import pandas as pd
# sys.path.insert(0,'../')
from py_scripts.utilities.loadData import db_manager, get_training_ids, get_model_input
from py_scripts.utilities.modelStruct import modelIO


def percentileCalc(trainData, testData):
	"""
		Calculate percentile according to TRAINING SET
		
        Find unique values in train and test list, then loop through to insert test samples into the training scale.
		
        trainData: training cases to build the model
        testData: test cases to run the model with
		
        only return percentile for testCaseIds
		"""
	uni_data,uni_count = np.unique(trainData,return_counts=True)
	length = len(trainData)    # number of training samples
	uni_test = np.unique(testData)
	uni_test = [x for x in uni_test if x not in uni_data]    # unique test samples different from training set
	uni_count = np.cumsum(uni_count)
	uni_count = [0]+list(uni_count[:-1])

	uni_data = uni_data.tolist()
	ind = 0 # index in training
	i = 0 # index in testing
	test_append = []
	test_count_append = []
	while (ind < len(uni_data)) & (i < len(uni_test)):
		x = uni_test[i]
		if x > uni_data[ind]:
			ind += 1
		else:
			test_append.append(x)
			test_count_append.append(uni_count[ind])
			i += 1
	if ind == len(uni_data):
		test_append += uni_test[i:]
		test_count_append += [len(uni_data)]*(len(uni_test)-i)
		
	uni_data = uni_data+test_append
	uni_count = uni_count+test_count_append


	uni_dic = {uni_data[i]: uni_count[i]/length for i in range(len(uni_data))}
	# dic = {testCaseIds[i]: uni_dic[testData[i]] for i in range(len(testCaseIds))}
	

	return [uni_dic[testdata] for testdata in testData]
	
	
	
def main(features,train_features,trainCaseIds,test_features,testCaseIds, use_training_cases = False):
	""" calculate percentile for incoming test cases, based on training set
	
	in:
		features:n dimension array, features to train and test
		trainCaseIds: m dimension array
		train_features: m*n dimension arry, trainCases feature
		
		testCaseIds: p dimension array
		test_features: p*n dimension array

		use_training_cases:if true, training features are the used for testing as well (append to the end).

	"""
	
	# define field names for input and output


	output_fields = ['Per'+x for x in features]
	output_fields += ['CaseId', 'BlockNum']
	

	if use_training_cases:
		testCaseIds += trainCaseIds
		test_features+=train_features
		
	# calculate percentile for all fields
	percentiles = []
	for field in features:
		percentiles.append(percentileCalc(train_features[field],test_features[field]))

	percentiles.append(testCaseIds)
	percentiles.append(np.zeros(len(testCaseIds)))

	percentiles = np.stack(percentiles,axis=1)

	result = pd.DataFrame(percentiles,columns=output_fields,index=testCaseIds)


	return result.to_dict('records')
	
	