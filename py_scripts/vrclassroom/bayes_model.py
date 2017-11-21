
# coding: utf-8

# # Creation of Bayes Model:
# This is meant to be run after the SignalDetection python script that creates the signal detections. We use a naive bayes model for our predictions. We use the internally stored training_set table to build the model, so one only needs to pass in the ids the we wish to use this model to predict.

#When running this script, pass in only the caseIds that we wish to predict that have complete data.

#   Example:
#	   python signal_detection_vrclassroom.py 69
#	   python signal_detection_vrclassroom.py 71 92
#   You can even pass in nothing and the script will just run and insert the values in the training_set:
#		python signal_detection_vrclassroom.py
#   However, you can't give this script caseIds which are already in the training_set table.
#

# ### Necessary imports:

# In[1]:

import numpy as np
import math
from scipy import stats
import pandas as pd
import pymysql
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import json
import collections
import pdb

#from __future__ import division # Not neccessary in Python 3 and later
import scipy
from math import exp,sqrt

from py_scripts.utilities.loadData import db_manager, get_model_input
from py_scripts.utilities.modelStruct import modelIO, modelParams, bayesModel

# ### 2) Functions to Estimate Probability Cutoffs

# In[3]:


#Assuming normal distributions we estimate their moments,
#then calculate raw probabilities for each subjects data point,
#finally returning the true estimated probability.

def generate_probability(data, adhd, isTrainData, beta_distribution = False, beta_params = None):
	"""
		Generate probabilities for each features and overall probability based on Bayes model.
		
		:Todo:
			Separate training and testing
				Store model params from training
				Do prediction with training results
			In ``combineProbs``, we should do \SUM(log(prob_i)) instead of \PROD(prob_i) to improve numerical stability
	"""
	adhdTrainData = adhd[np.where(isTrainData == 1)]
	data = data.astype(float)

	if not beta_distribution:
		trainData = data[np.where(isTrainData == 1)]
		#calculating moments:
		adhdMu = scipy.mean(trainData[adhdTrainData == 1])
		adhdSD = scipy.std(trainData[adhdTrainData == 1])
		healthyMu = scipy.mean(trainData[adhdTrainData == 0])
		healthySD = scipy.std(trainData[adhdTrainData == 0])
	else:
		if beta_params != None:
			data[data >= beta_params[1]] = beta_params[1] -1e-3
			data[data <= beta_params[0]] = beta_params[0] + 1e-3
			trainData = data[np.where(isTrainData == 1)]
			adhdAlpha, adhdBeta, adhdLoc, adhdScale = scipy.stats.beta.fit(trainData[adhdTrainData == 1], floc = beta_params[0], fscale = beta_params[1])
			healthyAlpha, healthyBeta, healthyLoc, healthyScale = scipy.stats.beta.fit(trainData[adhdTrainData == 0], floc = beta_params[0], fscale = beta_params[1])
		else:
			adhdAlpha, adhdBeta, adhdLoc, adhdScale = scipy.stats.beta.fit(trainData[adhdTrainData == 1])
			healthyAlpha, healthyBeta, healthyLoc, healthyScale = scipy.stats.beta.fit(trainData[adhdTrainData == 0])
						




	adhdADHDProbs = np.zeros(np.shape(data)[0])
	healthyProbs = np.zeros(np.shape(data)[0])

	# for i in np.arange(np.shape(data)[0]):
	if not beta_distribution:
		adhdADHDProbs = scipy.stats.norm.pdf(data, loc = adhdMu, scale = adhdSD)
		healthyProbs = scipy.stats.norm.pdf(data, loc = healthyMu, scale = healthySD)
	else:
		if True:#beta_params == None:
			adhdADHDProbs = scipy.stats.beta.pdf(data, a = adhdAlpha, b = adhdBeta, loc = adhdLoc, scale = adhdScale)
			healthyProbs = scipy.stats.beta.pdf(data, a = healthyAlpha, b = healthyBeta, loc = healthyLoc, scale = healthyScale)
		else:
			adhdADHDProbs = scipy.stats.beta.pdf(data, a = adhdAlpha, b = adhdBeta, loc = beta_params[0], scale = beta_params[1])
			healthyProbs = scipy.stats.beta.pdf(data, a = healthyAlpha, b = healthyBeta, loc = beta_params[0], scale = beta_params[1])

			
	return adhdADHDProbs/(adhdADHDProbs + healthyProbs), 1 - (adhdADHDProbs/(adhdADHDProbs + healthyProbs))


# calculate mean and covariance matrix of adhd/health group in training set
# and produce probability for all cases of adhd vs health
def generate_probability_mvg(data, adhd, isTrainData):
	data = data.transpose()
	trainData = data[:,np.where(isTrainData == 1)[0]]
	adhdTrainData = np.asarray(adhd[np.where(isTrainData == 1)])
	# pdb.set_trace()
	adhdMu = np.mean(trainData[:,adhdTrainData == 1],axis=1)
	adhdCov = np.cov(trainData[:,adhdTrainData == 1])
	healthyMu = np.mean(trainData[:,adhdTrainData == 0],axis=1)
	healthyCov = np.cov(trainData[:,adhdTrainData == 0])
	# pdb.set_trace()
   
	adhd_rv = stats.multivariate_normal(adhdMu,adhdCov)
	healthy_rv = stats.multivariate_normal(healthyMu,healthyCov)
	
	adhdADHDProbs = adhd_rv.pdf(data.T)
	healthyProbs = healthy_rv.pdf(data.T)
	
	return adhdADHDProbs/(adhdADHDProbs + healthyProbs), 1 - (adhdADHDProbs/(adhdADHDProbs + healthyProbs))
	




# In[4]:

def combineProbs(probs,method):
	""" Combine probabilities from multiple probability model
	Args:
		probs: np.ndarray, each column corresponds to a model, each row is a case
		method: string, can be 'multiply', 'mean', 'median'
	
		"""
	return eval("np.%s(probs,axis=1)"%method)


# In[5]:

# In[6]:

"""
This is the meat of these three scripts. The function that takes all
we have done up until now and calculates the features individual and cumulative prediction of adhd
We use a naive bayes methodology inspired by this:
https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Constructing_a_classifier_from_the_probability_model

Were we to seek to improve this model, we may want to switch out normal distributions for betas of the
omissions and commission errors, and include gamma distributions for TargetsRTV
Some caveats:
	-From Yiming, naive bayes works poorly with highly correlated features, several of which ours are.
It may be good to return and check the true correlations as we get more data.
	-Naive Bayes is known for for not giving particularily accurate probability estimates, although their
classification is usually right (IE the probabilities are on the right side of the midpoint). Because
we are advertising the probability as showing where one is on the spectrum, we should revisit this as
we get more data.
"""
def get_probabilities(data, isTrainData, model_params = None):
	from_model = False
	if (model_params != None) & (isinstance(data,pd.DataFrame)):
		adhd = np.asarray(data.iloc[:, 0])
		from_model = True
		params = model_params.params
		probs = ()
		for field,distribution,dist_param in zip(model_params.fields,params['distribution'],params['dist_param']):
			probs += generate_probability(np.asarray(data[field]), adhd, isTrainData, distribution=='beta',dist_param)
		probs_adhd = probs[::2]
		probs_healthy = probs[1::2]
		totalRawProbabilityADHD = combineProbs(np.asarray(probs_adhd).T,params['method'])
		totalRawProbabilityHealthy = combineProbs(np.asarray(probs_healthy).T,params['method'])

		totalProbabilityADHD = np.divide(totalRawProbabilityADHD,totalRawProbabilityADHD+totalRawProbabilityHealthy)
		priorsADHD = np.zeros((np.shape(data)[0]))
		
		count = 0
		for i in np.where(isTrainData==1)[0]:
			if (np.round(totalProbabilityADHD[i]) == adhd[i]):
				count = count + 1
		print("training accuracy: ", 100 * count/np.sum(isTrainData), "%")
		
		probs = np.asarray(probs+(totalRawProbabilityADHD,totalRawProbabilityHealthy,priorsADHD,totalProbabilityADHD)).T
		# probs_df = pd.DataFrame(probs,model_IO.output_fields,index=data.index)
		return probs
	else:
		adhd = data[:, 0]
		#Here we are initializing everything to have the correct dimension so we can
		#concatenate them in the future.
		omissionADHDProbs = np.zeros((np.shape(data)[0], 1))
		commissionADHDProbs = np.zeros((np.shape(data)[0],1))
		targetRTVADHDProbs = np.zeros((np.shape(data)[0],1))
		dPrimeADHDProbs = np.zeros((np.shape(data)[0],1))
		betaADHDProbs = np.zeros((np.shape(data)[0],1))
		priorsADHD = np.zeros((np.shape(data)[0],1))
		totalRawProbabilityADHD = np.zeros((np.shape(data)[0],1))

		omissionHealthyProbs = np.zeros((np.shape(data)[0], 1))
		commissionHealthyProbs = np.zeros((np.shape(data)[0],1))
		targetRTVHealthyProbs = np.zeros((np.shape(data)[0],1))
		dPrimeHealthyProbs = np.zeros((np.shape(data)[0],1))
		betaHealthyProbs = np.zeros((np.shape(data)[0],1))
		priorsHealthy = np.zeros((np.shape(data)[0],1))
		totalRawProbabilityHealthy = np.zeros((np.shape(data)[0],1))

		#generating the individual probabilities:
		omissionADHDProbs[:, 0], omissionHealthyProbs[:, 0] = generate_probability(data[:, 1], adhd, isTrainData, True,[-1e-6,52])
		commissionADHDProbs[:, 0], commissionHealthyProbs[:, 0] = generate_probability(data[:, 2], adhd, isTrainData, True,[-1e-6,520])
		targetRTVADHDProbs[:, 0], targetRTVHealthyProbs[:, 0] = generate_probability(data[:, 3], adhd, isTrainData)
		dPrimeADHDProbs[:, 0], dPrimeHealthyProbs[:, 0] = generate_probability(data[:, 4], adhd, isTrainData)
		betaADHDProbs[:, 0], betaHealthyProbs[:, 0] = generate_probability(data[:, 5], adhd, isTrainData)

		#Getting the total raw probability by multiplying all features
		totalRawProbabilityADHD = multiply_5_arrays(omissionADHDProbs,
														  commissionADHDProbs,
														  targetRTVADHDProbs,
														  dPrimeADHDProbs,
														  betaADHDProbs)
		totalRawProbabilityHealthy = multiply_5_arrays(omissionHealthyProbs,
															commissionHealthyProbs,
															targetRTVHealthyProbs,
															dPrimeHealthyProbs,
															betaHealthyProbs)
		# pdb.set_trace()
															
		# totalRawProbabilityADHD,totalRawProbabilityHealthy = generate_probability_mvg(data[:,1:6], adhd, isTrainData)
		# totalRawProbabilityADHD = totalRawProbabilityADHD[:,np.newaxis]
		# totalRawProbabilityHealthy = totalRawProbabilityHealthy[:,np.newaxis]
		
		#Finding the priors using an identical method as the features on SNAP scores. It could really be called a
		#feature itself but ~\_('_')_/~
		# priorsADHD[:, 0], priorsHealthy[:, 0] = generate_probability(data[:, 6], adhd, isTrainData)

		#getting the total probability
		totalProbabilityADHD = np.zeros((np.shape(data)[0],1))
		# pADHD = np.multiply(priorsADHD, totalRawProbabilityADHD)
		# pHealthy = np.multiply(priorsHealthy, totalRawProbabilityHealthy)
		pADHD = totalRawProbabilityADHD
		pHealthy = totalRawProbabilityHealthy
		pTotal = pADHD + pHealthy
		totalProbabilityADHD = np.divide(pADHD,pTotal)


		#Just for fun, here is our training accuracy:
		# Only report accuracy on training set
		count = 0
		for i in np.where(isTrainData==1)[0]:
			if (np.round(totalProbabilityADHD[i]) == adhd[i]):
				count = count + 1
		print("training accuracy: ", 100 * count/np.sum(isTrainData), "%")

		return np.concatenate((omissionADHDProbs,
							   omissionHealthyProbs,
							   commissionADHDProbs,
							   commissionHealthyProbs,
							   targetRTVADHDProbs,
							   targetRTVHealthyProbs,
							   dPrimeADHDProbs,
							   dPrimeHealthyProbs,
							   betaADHDProbs,
							   betaHealthyProbs,
							   totalRawProbabilityADHD,
							   totalRawProbabilityHealthy,
							   priorsADHD,
							   totalProbabilityADHD),
							  axis = 1)



# In[7]:

#helper function for readability
def multiply_5_arrays(a,b,c,d,e):
	holder = np.multiply(a, b)
	holder = np.multiply(holder, c)
	holder = np.multiply(holder, d)
	holder = np.multiply(holder, e)
	return holder





def get_snap(caseIds, db, cursor):
	data = []

	for i in caseIds:
		sql = """SELECT CaseId, AnswerContent, QuestionnaireId FROM questionnaryanswer WHERE CaseId = """ + str(i)
		cursor.execute(sql)
		# Fetch all the rows in a list of lists.
		results = cursor.fetchall()
		if len(results) == 0: #if there's not someone's data in database
			raise Exception(7)
		if i > 20: #ignores first 20 CaseIds with corrupted Json strings
			diction = json.loads(results[0][1])
			keys = []
			vals = []
			for key, value in diction.items():
				key = str(key)
				if len(key) < 10: #needs strings to have the same number of characters for proper sorting
					key = key[:8] + "0" + key[8:]
				keys.append(key)
				vals.append(int(value[0]))
			data.append(vals)

	data = np.asarray(data)
	data = data[:, np.argsort(keys)] #dict objects shuffle stuff randomly at times, so we need to resort it\
	sumSNAP = np.zeros((np.shape(data)[0], 1))

	sumSNAP[:, 0] = np.apply_along_axis(func1d = sum, axis = 1, arr = data[:, :19])

	return sumSNAP

# In[16]:



# ### Main Function:

# In[18]:

#main function
def main(featureFieldsObj,
	train_features,trainCaseIds,trainCasesAdhd,
	test_features,testCaseIds,
	model_params,use_training_cases=False):
	"""
	Bayes Model Train and Test

	In:

	featureFieldsObj :instance of featureFields
	trainCaseIds: train cases id
	train_features: features data get by featureFieldsObj
	trainCasesAdhd: train cases id adhdtype 
	test_features: features data get by featureFieldsObj
	testCaseIds: train cases id
	model_params: instance of modelParams
	use_training_cases: 

	"""

	isTrainingSet = np.ones((len(trainCaseIds) + len(testCaseIds)))
	if use_training_cases == False:
		isTrainingSet[len(trainCaseIds):] = np.repeat(0, len(testCaseIds))

	finalIds = np.concatenate((trainCaseIds, testCaseIds))


	caseIds = np.stack((finalIds, isTrainingSet), 1)

	print('\n\n')
	print(caseIds)
	print('\n\n')

	#merge all features data of trainCases and testCases


	trainCasesAdhd=pd.DataFrame(list(trainCasesAdhd))
	trainCasesAdhd.loc[trainCasesAdhd['ADHDDiagnose']>0,('ADHDDiagnose')]=1#todo ignore ADHD Type Now

	trainPatients=pd.merge(trainCasesAdhd,train_features,left_on='Id',right_on='CaseId')

	testCasesAdhd=pd.DataFrame([{'Id':testCase,'ADHDDiagnose':0} for testCase in testCaseIds])

	testPatients=pd.merge(testCasesAdhd,test_features,left_on='Id',right_on='CaseId')
	
	allPatients=trainPatients.append(testPatients)
	del allPatients['CaseId']
	print(allPatients)

		
	# print("calculating probabilities")
	probabilities = get_probabilities(allPatients, caseIds[:, 1], model_params = model_params)
		
	# print("inserting data")
	result = None
	if use_training_cases:
		isTestingSet = isTrainingSet.astype(bool)
	else:
		isTestingSet = isTrainingSet==0

	# if write_db:
	# 	probabilities_df = pd.DataFrame(np.concatenate([probabilities,caseIds[:,0:1]],axis=1),columns=model_fields.output_fields+['CaseId'])
	# 	insert_bayes_probabilities_df(probabilities_df.iloc[isTestingSet,:], caseIds[isTestingSet,0])

	result = pd.DataFrame(probabilities[isTestingSet,:],caseIds[isTestingSet,0],columns = featureFieldsObj.get_all_output_fields())
	
	return result,None



