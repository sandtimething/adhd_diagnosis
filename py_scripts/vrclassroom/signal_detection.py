# coding: utf-8

# # Creation of Signal Detection Data
#
# This script uses signal detection theory and pulls from the table cpt_output_results to create the features dprime, beta, and C for each
# individual based off their omission/commission errors, storing them in the table signal_detection.

# Usage:
#   python signal_detection_vrclassroom.py caseids
#   Example:
#	   python signal_detection_vrclassroom.py 1
#	   python signal_detection_vrclassroom.py 1 2


# ### 1. Necessary Imports:
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
from math import exp,sqrt

# In[1]:
def main(cpt_data,CaseId):
	"""
	In: cpt_data:[dict(Block, TargetsRtVariability, OmissionErrors, CommissionErrors, TargetCount, CaseId)]

	Out: [dict(Block, DPrime, Beta, C, CaseID)]
	"""

	# ### 2. Functions to estimate D', Beta, and C

	Z = stats.norm.ppf

	#This function calculates D', Beta, and C
	def dPrime(hits, misses, fas, crs):
		# Floors an ceilings are replaced by half hits and half FA's
		halfHit = 0.5/(hits+misses)
		halfFa = 0.5/(fas+crs)

		# Calculate hitrate and avoid d' infinity
		hitRate = hits/(hits+misses)
		if hitRate == 1: hitRate = 1-halfHit
		if hitRate == 0: hitRate = halfHit

		# Calculate false alarm rate and avoid d' infinity
		faRate = fas/(fas+crs)
		if faRate == 1: faRate = 1-halfFa
		if faRate == 0: faRate = halfFa

		# Return d', beta, c and Ad'
		out = {}
		dPrizime = Z(hitRate) - Z(faRate)
		beta = exp((Z(faRate)**2 - Z(hitRate)**2)/2)
		c = -(Z(hitRate) + Z(faRate))/2
		Ad = stats.norm.cdf(dPrizime/sqrt(2))
		return dPrizime, beta, c


	# In[4]:

	#This function processes ommission and commission errors,
	#and send them out to estimate d', beta, C function
	def generate_signal_detection(omissionErrors, commissionErrors, trueSigs, falseSigs):

		hits = trueSigs - omissionErrors
		misses = omissionErrors
		falseAlarms = commissionErrors
		correctRejections = falseSigs - commissionErrors

		length = np.shape(omissionErrors)[0]
		dP = np.zeros(length)
		beta = np.zeros(length)
		C = np.zeros(length)

		for i in np.arange(np.shape(omissionErrors)[0]):
			dP[i], beta[i], C[i] = dPrime(hits[i], misses[i], falseAlarms[i], correctRejections[i])
		return np.column_stack((dP, beta, C))


	# In[5]:

	#Needs major editing and changes!!!

	#This funtion slices up raw table into requisite chunks

	def separate_blocks(data):

		#creating blocks:
		block1 = data[data[:, 0] ==1, :]
		block1 = block_to_signal(block1, 173) #not accurate, need to double check

		block2 = data[data[:, 0] ==2, :]
		block2 = block_to_signal(block2, 173) #not accurate, need to double check

		block3 = data[data[:, 0] ==3, :]
		block3 = block_to_signal(block3, 174) #not accurate, need to double check

		total = data[data[:, 0] ==0, :]
		total =  block_to_signal(total, 520 - 52) #not accurate, need to double check
		#merging blocks into final table:
		finalData = np.array((block1[0, :], block2[0, :], block3[0, :], total[0, :]))

		for i in np.arange(1, np.shape(total)[0]):
			holder = np.array((block1[i, :], block2[i, :], block3[i, :], total[i, :]))
			finalData = np.concatenate((finalData, holder))

		return finalData


	# In[6]:

	#Needs major editing and changes!!!

	#This function serves as an intermediary between separate-blocks and generate_signal_detection
	def block_to_signal(block, falseSigs):
		trueSigs = block[:, 4]
		blockSignal = generate_signal_detection(block[:, 2], block[:, 3], block[:, 4], falseSigs)
		block = np.concatenate((block, blockSignal), axis = 1)
		return block


	# In[9]:

	#Inserts created features into signal detection tables

	def generate_signal(datas):
		# do insertion
		result=[]
		for row in datas:
			result.append(dict(Block=np.int(row[0]),DPrime=row[6],Beta=row[7],C=row[8],CaseId=row[5])) 
		return result
		# try:

		# 	for i in range(len(data)):
		# 		sql = "INSERT INTO %s"%tb
		# 		sql += "(Id, Block, DPrime, Beta, C, CaseID, Description)"
		# 		sql += "VALUES (%s, %s, %s, %s, %s, %s, %s)" % (np.int(maxPreviousId + i + 1),
		# 																		np.int(data[i, 0]),
		# 																		data[i, 6],
		# 																		data[i, 7],
		# 																		data[i, 8],
		# 																		data[i, 5],
		# 																		"Null")
		# 		#sql = "SHOW COLUMNS FROM "
		# 		# Execute the SQL command
		# 		cursor.execute(sql)
		# 		count += 1
		# 	# Commit your changes in the database
		# 	# Commit only once - Tom
		# 	db.commit()
		# 	print('%d rows inserted into database %s'%(count,tb))

		# except Exception as e:
		# 	raise Exception(5)



	# ### 4. Main Function:

		#Gets CPT data from database
	def get_data(cpt_data):
		return np.asarray([[cpt["Block"],cpt["TargetsRtVariability"]
		,cpt["OmissionErrors"],cpt["CommissionErrors"]
		,cpt["TargetCount"],cpt["CaseId"]] for cpt in cpt_data])


	#main function
	def main_bayes(caseIds):

		allPatients = get_data(cpt_data)

		allPatients = separate_blocks(allPatients)

		return allPatients



	# In[14]:

	#calls main function, stores data for testing
	allPatients = main_bayes(CaseId)
	returndict=generate_signal(allPatients)
	return returndict



