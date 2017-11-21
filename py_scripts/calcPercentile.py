"""
	calcPercentile
	
	Calculate Percentile of test caseids based on train caseids, and write to db. 
	
	Origin from `percentile_witherrorhandling_2.py`. - Tom 08/31/2017

"""

import sys
import pymysql
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
	
def insert_percentile(df, mydb):
	""" 
		Insert percentile results into db
	"""
	mydb.insert_table(df,'percentile',index_name='CaseId',del_row_if_exist = True)
	
	
def main(testCaseIds,traindb_name = 'vrclassroomdata', testdb_name='webtest', use_training_cases = False, write_db = False, train_features=None, test_features=None):
	""" calculate percentile for incoming test cases, based on training set
	
	Args:
		testCaseIds: array-like. Features for case ids in the array must be found in testdb. 
		traindb_name: name of the training database to fetch training set. Should contain features for training cases
		testdb_name: name of the testing database to store percentile. Should contain features for test cases.
		use_training_cases: if true, training features are the used for testing as well (append to the end). ``traindb_name`` and ``testdb_name`` must be the same for this to take effect.
		write_db: if true, write result to database (percentile for testCaseIds)
		train_features: if specified, use the input data instead of fetching from traindb_name
		test_features: if specified, use the input data instead of fetching from testdb_name
	"""
	
	# input check
	if use_training_cases and (traindb_name != testdb_name):
		raise ValueError("``traindb_name`` and ``testdb_name`` must be the same")
		
	# initiate db connection
	mydb_train = db_manager(traindb_name)
	mydb_test = db_manager(testdb_name)
	
	# define field names for input and output
	input_tables = ['head_features','cpt_output_results']
	input_fields = [['PathLen', 'TimeActive', 'NumRot', 'TotalDeg'], ['OmissionErrors', 'CommissionErrors', 'TargetsRT', 'TargetsRtVariability', 'CommissionErrorsRT', 'CommissionErrorsRtVariability']]
	input_where = ['','where block=0']
	input_primarykey = ['CaseId','CaseId']
	output_fields = []
	for input_field in input_fields:
		output_fields += ['Per'+x for x in input_field]
	output_fields += ['CaseId', 'BlockNum']
	
	# get training caseids
	trainCaseIds = get_training_ids(mydb_train)
	if len(trainCaseIds) == 0:
		raise RuntimeError('cannot get training ids from training set table')
	
	if use_training_cases:
		testCaseIds += trainCaseIds
	
	# fetch features for training & testing
	modIO = modelIO(input_tables,input_fields,input_where,input_primarykey)
	if train_features is None:
		train_features = get_model_input(modIO,trainCaseIds,mydb_train)
	if test_features is None:
		test_features = get_model_input(modIO,testCaseIds,mydb_test)
		
		
	# calculate percentile for all fields
	percentiles = []
	for field in modIO.getAllFields():
		percentiles.append(percentileCalc(train_features[field],test_features[field]))
	percentiles.append(testCaseIds)
	percentiles.append(np.zeros(len(testCaseIds)))
	percentiles = np.stack(percentiles,axis=1)
	result = pd.DataFrame(percentiles,columns=output_fields,index=testCaseIds)

	# insert to db
	if write_db:
		insert_percentile(result,mydb_test)
	return result
	
	