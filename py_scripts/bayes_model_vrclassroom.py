
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
try:
	def main(CaseIDs,traindb_name = 'vrclassroomdata', testdb_name='webtest',write_db=False, use_df=True, config_path='../config/modelConfig', use_training_cases = False, modelobj = None, return_input=False,input_data=None):
		try:
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
		except Exception as e:
			raise Exception(1)


		# In[2]:





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
			if method == 'multiply':
				return np.prod(probs,axis=1)
			elif method == 'mean':
				return np.mean(probs,axis=1)
			elif method == 'median':
				return np.median(probs,axis=1)


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
		def get_probabilities(data, isTrainData, model_params = None, model_IO = None):
			from_model = False
			if (model_params != None) & (model_IO != None) & (isinstance(data,pd.DataFrame)):
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


		# ### 3. DB connecting, pull, & insert

		# In[8]:

		#initializes connection info for database
		def connect(db_name='vrclassroomdata'):
			try:
				return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
									   database = db_name,
									   user='cognitiveleap',
									   password= 'QWE@123456')
			except Exception as e:
				raise Exception(3)


		# In[9]:

		#Getting data from DB
		# Set label of testing set to 0
		def get_individual_data(CaseIds,modelIOStruct = None, tableNames=None,fieldNames=None,whereClauses=None, primary_keys = None):
			isTrainingSet = CaseIds[:,1]
			numTest = np.sum(isTrainingSet==0)
			CaseIds = CaseIds[:,0]

			try:
				if modelIOStruct != None:
					tableNames,fieldNames,whereClauses, primary_keys = modelIOStruct.getInput()
				
				if (tableNames == None) and (fieldNames == None):
					# prepare a cursor object using cursor() method
					dataCPT = get_cpt(CaseIds, isTrainingSet, cursor_rnd, cursor_web)
					dataSignal = get_signal(CaseIds, isTrainingSet, cursor_rnd, cursor_web)
					# dataSNAP = get_snap(CaseIds, db, cursor)
					dataSNAP = np.zeros((len(CaseIds),1))
					# dataADHD = get_adhd([CaseIds[i] for i in np.where(isTrainingSet==1)[0]], db_rnd, cursor_rnd, cross_db = True)
					# testADHD = np.concatenate((np.asarray([CaseIds[i] for i in np.where(isTrainingSet==0)[0]]).reshape(numTest,1),np.zeros((numTest,1))),axis=1)

					# dataADHD = np.concatenate((dataADHD,testADHD),axis=0)
					dataADHD = get_adhd(CaseIds, isTrainingSet, cursor_rnd, cursor_web, cross_db=True)
					result = np.concatenate((dataADHD, dataCPT, dataSignal, dataSNAP), axis = 1)
				elif len(tableNames) != len(fieldNames):
					print('tableNames and fieldNames must be the same length')
					raise Exception('tableNames and fieldNames must be the same length')
				else:
					mydb = db_manager()
					results = []
					for dbname,isTrain in zip((traindb_name,testdb_name),[1,0]):
						mydb.connect(dbname)
						df = get_model_input(modelIOStruct,CaseIds[isTrainingSet==isTrain],mydb)
						results.append(df)
					result = pd.concat(results,axis=0)
					
			except Exception as e:
				raise Exception(4)


			return result
			



		# In[10]:

		def get_cpt(caseIds, isTrain, cursor_train, cursor_test):

			dataPatient = []
			for i,isTr in zip(caseIds,isTrain):
				sql = """SELECT OmissionErrors, CommissionErrors, TargetsRtVariability
						 FROM cpt_output_results WHERE Block = 0 AND CaseId = """ + str(i)
				if isTr:
					cursor = cursor_train
				else:
					cursor = cursor_test
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) == 0: #if there's no someone's data in database
					print((i,isTr))
					raise Exception(7)
				dataPatient.append(results)
			dataPatient = np.concatenate(np.asarray(dataPatient))
			return dataPatient



		# In[11]:

		def get_signal(caseIds, isTrain, cursor_train, cursor_test):

			dataSignal = []
			for i,isTr in zip(caseIds,isTrain):
				sql = """SELECT DPrime, Beta FROM signal_detection WHERE Block = 0 AND CaseId = """ + str(i)
				if isTr:
					cursor = cursor_train
				else:
					cursor = cursor_test
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) == 0: #if there's no someone's data in database
					raise Exception(7)
				dataSignal.append(results)
			dataSignal = np.concatenate(np.asarray(dataSignal))
			return dataSignal


		# In[12]:

		def get_adhd(caseIds, isTrain, cursor_train, cursor_test, cross_db = False):

			dataADHD = []
			for i,isTr in zip(caseIds,isTrain):
				if not cross_db:
					sql = """SELECT Id, ADHDDiagnose FROM patient WHERE Id = """ + str(i)
				else:
					sql = "select a.Id, b.ADHDDiagnose  from `case` a, patient b where b.Id=a.SubjectId and a.Id="+str(i)
				if isTr:
					cursor = cursor_train
					cursor.execute(sql)
					# Fetch all the rows in a list of lists.
					results = np.asarray(cursor.fetchall())
					if len(results) == 0: #if there's not someone's data in database
						raise Exception(7)
					if results[0][1] > 0: # ignoring severity for now
						results[0][1] = 1
				else:
					results = np.asarray([[i,0]])
				dataADHD.append(results)
			dataADHD = np.concatenate(np.asarray(dataADHD))
			return dataADHD


		# In[13]:




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


		def has_existing_CaseId(caseid):


			# Prepare SQL query to INSERT a record into the database.
			try:
				sql = """SELECT * FROM bayes_probabilities WHERE CaseId = %s""" % (caseid)
				cursor_web.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor_web.fetchall())
				if len(results) != 0: #if there's someone's data in database
					return True
				else: return False
			except Exception as e:
				raise Exception(4)


		def insert_bayes_probabilities(data, caseIds):
			count = 0
			try:
				# search through the table and delete already existed caseids
				tb = 'bayes_probabilities'
				try:
					for caseid in np.unique(caseIds):
						if has_existing_CaseId(caseid):
							sql = """DELETE FROM %s WHERE CaseId = """%tb + str(caseid)
							cursor_web.execute(sql)
							count += 1
				except Exception as e:
					raise Exception(8)
				print ('%d caseids got deleted from %s'%(count,tb))
				count = 0

				for i in range(len(data)):
					count += 1
					sql = "INSERT INTO %s"%tb
					sql += "(CaseId,"
					sql += "OmissionRawProbabilityADHD, "
					sql += "OmissionRawProbabilityHealthy, "
					sql += "CommissionRawProbabilityADHD, "
					sql += "CommissionRawProbabilityHealthy, "
					sql += "TargetRTVRawProbabilityADHD, "
					sql += "TargetRTVRawProbabilityHealthy, "
					sql += "DPrimeRawProbabilityADHD, "
					sql += "DPrimeRawProbabilityHealthy, "
					sql += "BetaRawProbabilityADHD, "
					sql += "BetaRawProbabilityHealthy, "
					sql += "totalRawProbabilityADHD, "
					sql += "totalRawProbabilityHealthy, "
					sql += "priors, "
					sql += "finalProbabilityOfADHD) "
					sql += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (caseIds[i],
																									 data[i, 0],
																									 data[i, 1],
																									 data[i, 2],
																									 data[i, 3],
																									 data[i, 4],
																									 data[i, 5],
																									 data[i, 6],
																									 data[i, 7],
																									 data[i, 8],
																									 data[i, 9],
																									 data[i, 10],
																									 data[i, 11],
																									 data[i, 12],
																									 data[i, 13])
					# Execute the SQL command
					cursor_web.execute(sql)
				print ('%d caseids got inserted or updated from %s'%(count,tb))
			except Exception as e:
				raise Exception(5)
				
		def insert_bayes_probabilities_df(df, caseIds):
			"""
				Insert probabilities dataframe to database, for caseIds.
				"""
			mydb = db_manager(testdb_name)
			mydb.insert_table(df,'bayes_probabilities',index_name='CaseId',del_row_if_exist = True)
			


		# ### For testing:

		# In[17]:

		def delete_Ids(caseIds):
			try:
				for i in caseIds:
					sql = """delete from bayes_probabilities where CaseId = """ + str(i)
					# Execute the SQL command
					cursor.execute(sql)
			except Exception as e:
				raise Exception(5)



		def get_training_ids():
			sql = """SELECT CaseID FROM training_set"""
			cursor_rnd.execute(sql)
			# Fetch all the rows in a list of lists.
			results = np.asarray(cursor_rnd.fetchall())[:, 0]
			if len(results) == 0: #if there is no data in DB
				raise Exception (7)
			return results

		# ### Main Function:

		# In[18]:

		#main function
		def main_bayes_boundary(caseIdsTest, write_db = False, use_df = False, config_path = '../config/modelConfig',modelobj = None, return_input=None,input_data=None):

			# get model object
			if modelobj == None:
				model = bayesModel.json_decoder(config_path)
			else:
				model = modelobj
			model_fields = model.modIO
			model_params = model.modParams

			# get input data
			if input_data == None:
				#manipulation of data
				caseIdsTrain = get_training_ids()
				# for i in caseIdsTest:
					# if i in caseIdsTrain:
						# print("Error: user input CaseId contained in training_set! Quitting now...")
						# return
				isTrainingSet = np.ones((len(caseIdsTrain) + len(caseIdsTest)))
				if use_training_cases == False:
					isTrainingSet[len(caseIdsTrain):] = np.repeat(0, len(caseIdsTest))
				finalIds = np.zeros((len(caseIdsTrain) + len(caseIdsTest)))
				finalIds = np.append(caseIdsTrain, caseIdsTest)

				caseIds = np.stack((finalIds, isTrainingSet), 1)
			
				# print("getting data")
				if not use_df:
					allPatients = get_individual_data(caseIds)
				else:
					allPatients = get_individual_data(caseIds, modelIOStruct=model_fields)
			else:
				caseIds = input_data[0]
				isTrainingSet = caseIds[:,1]
				allPatients = input_data[1]
				print('using input_data for analysis')
				
			# if return_input == True, return necessary data and exit
			if return_input:
				return None,(caseIds,allPatients)
				
			# print("calculating probabilities")
			probabilities = get_probabilities(allPatients, caseIds[:, 1], model_params = model_params, model_IO = model_fields)
				
			# print("inserting data")
			result = None
			if use_training_cases:
				isTestingSet = isTrainingSet.astype(bool)
			else:
				isTestingSet = isTrainingSet==0
			if write_db:
				probabilities_df = pd.DataFrame(np.concatenate([probabilities,caseIds[:,0:1]],axis=1),columns=model_fields.output_fields+['CaseId'])
				insert_bayes_probabilities_df(probabilities_df.iloc[isTestingSet,:], caseIds[isTestingSet,0])
			result = pd.DataFrame(probabilities[isTestingSet,:],caseIds[isTestingSet,0],columns = model_fields.output_fields)
			
			return result,None





		# In[19]:
		#Uncomment lines below for testing
		#delete_Ids(np.append(np.arange(21, 36, 1), np.arange(41, 47, 1)))

		#allPatients = get_individual_data(caseIds)
		#probabilities = get_probabilities(allPatients[:, 1:])
		#caseIds = allPatients[:, 0]
		#insert_bayes_probabilities(probabilities, caseIds)
		#return allPatients

		#calls main from here:
		#caseIds = np.append(np.arange(21, 36, 1), np.arange(41, 47, 1))
		#connect to db
		db_rnd = connect(traindb_name)
		# prepare a cursor object using cursor() method
		cursor_rnd = db_rnd.cursor()
		db_web = connect(testdb_name)
		# prepare a cursor object using cursor() method
		cursor_web = db_web.cursor()	
		
		if not use_df:
			print('use_df=False is deprecated. use_df=True by default.')
			use_df = True

		data,input_data = main_bayes_boundary(CaseIDs,write_db = write_db, use_df = use_df, config_path=config_path, modelobj=modelobj, return_input=return_input,input_data=input_data)

		db_rnd.commit()
		db_rnd.close()
		db_web.commit()
		db_web.close()
		
		if return_input:
			return input_data
		else:
			return data
	
	if __name__ == '__main__':
		import sys
		try:
			CaseIDs = sys.argv[ 1: ]
			CaseIDs = [int(i) for i in CaseIDs]
		except Exception as e:
			raise Exception(2)
		main(CaseIDs)
	
except Exception as e:
	try:
		print('Error code: '+str(e))
		if 'db_web' in dir():
			db_web.close()
			print('db close in error handling')
		if 'db_rnd' in dir():
			db_rnd.close()
			print('db_rnd close in error handling')	
	except:
		print('about to exit. db close error.')

	try:
		if isinstance(e,str):
			print('uncaught error.')
			sys.exit(10)
		sys.exit(e)
	except Exception as e:	# otherwise sys.exit will be caught
		print('sys.exit error')


