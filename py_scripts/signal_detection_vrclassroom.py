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
	
try:
	# ### 1. Necessary Imports:

	# In[1]:
	def main(CaseIDs,db_name='webtest', write_db=True):
		try:
			import sys #put import sys in the very beginning, so sys can handle error in the following import
			import numpy as np
			import math
			from scipy import stats
			import pandas as pd
			import pymysql



			#from __future__ import division # Not neccessary in Python 3 and later
			import scipy
			from math import exp,sqrt
		except Exception as e:
			raise Exception(1)



		# ### 2. Functions to estimate D', Beta, and C

		# In[3]:

		try:
			Z = stats.norm.ppf
		except Exception as e:
			raise Exception(9)

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


		# ### 3. DB connecting, pull, & insert

		# In[7]:

		#initializes connection info for database
		def connect(db_name = 'webtest'):
			try:
				return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
									   database = db_name,
									   user='cognitiveleap',
									   password= 'QWE@123456')
			except Exception as e:
				raise Exception(3)


		# In[8]:

		#Gets CPT data from database
		def get_data(CaseIds):




			try:

				data = []

				# Prepare SQL query to INSERT a record into the database.
				for i in CaseIds:
					sql = """SELECT Block, TargetsRtVariability, OmissionErrors, CommissionErrors, TargetCount, CaseId
							 FROM cpt_output_results WHERE CaseId = """ + str(i)
					cursor.execute(sql)
					# Fetch all the rows in a list of lists.
					results = np.asarray(cursor.fetchall())
					if len(results) == 0: #if there's not someone's data in database
						raise Exception(7)
					"""add an additional check in here for the case that there is already a signal detection output in the db"""
					data.append(results)
				concatenated = np.concatenate(np.asarray(data)) #put everything in the "try"
			except Exception as e:
				raise Exception(4)

			# db.close() delete all db.close() here; close at last for all

			return concatenated


		# In[9]:

		#Inserts created features into signal detection tables

		def insert_signal(data):



			maxPreviousId = get_max_previous_id()
			count = 0


			# search through the table and delete already existed caseids
			try:
				tb = 'signal_detection'
				for caseid in np.unique(data[:,5]):
					if has_existing_CaseId(caseid):
						sql = """DELETE FROM %s WHERE CaseId = """%tb + str(caseid)
						cursor.execute(sql)
						count += 1
			except Exception as e:
				raise Exception(8)
			print ('%d caseids got deleted from %s'%(count,tb))

			count = 0

			# do insertion
			try:

				for i in range(len(data)):
					sql = "INSERT INTO %s"%tb
					sql += "(Id, Block, DPrime, Beta, C, CaseID, Description)"
					sql += "VALUES (%s, %s, %s, %s, %s, %s, %s)" % (np.int(maxPreviousId + i + 1),
																					np.int(data[i, 0]),
																					data[i, 6],
																					data[i, 7],
																					data[i, 8],
																					data[i, 5],
																					"Null")
					#sql = "SHOW COLUMNS FROM "
					# Execute the SQL command
					cursor.execute(sql)
					count += 1
				# Commit your changes in the database
				# Commit only once - Tom
				db.commit()
				print('%d rows inserted into database %s'%(count,tb))

			except Exception as e:
				raise Exception(5)

	#		db.close() close at last for all


		# In[10]:

		#This gets the maximum previous ID and ensures that we start incrementing form the correct ID
		def get_max_previous_id():


			try:
				sql = """SELECT Id FROM signal_detection"""
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) == 0: #if there's no data in database
					results = 0
				maximum = np.max(np.asarray(results))

	#			db.close() close at last for all
			except Exception as e:
				raise Exception(4)

			return maximum


		# In[11]:

		#This function ensures that there are no repeated caseIds in the data we are trying to insert
		#Yiming changed it in the branch to detect whether each caseId is detected
		# Should return true whenever some rows are present -Tom
		def has_existing_CaseId(caseid):


			# Prepare SQL query to INSERT a record into the database.
			try:
				sql = """SELECT * FROM signal_detection WHERE CaseId = %s""" % (caseid)
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) != 0: #if there's someone's data in database
					return True
				else: return False
			except Exception as e:
				raise Exception(4)



		# ### 4. Main Function:

		# In[12]:

		#main function
		def main_bayes(caseIds):
			allPatients = get_data(caseIds)

			allPatients = separate_blocks(allPatients)
			if write_db:
				insert_signal(allPatients)
			return allPatients

		db = connect(db_name)
		cursor = db.cursor()



		# In[14]:

		#calls main function, stores data for testing
		allPatients = main_bayes(CaseIDs)
		db.close()
		return allPatients



		# In[ ]:

		#print(allPatients)


		# In[ ]:



		# In[ ]:

	if __name__ == '__main__':
		import sys
		try:
			CaseIDs = sys.argv[ 1: ]
			CaseIDs = [int(i) for i in CaseIDs]
		except Exception as e:
			raise Exception(2)
		main(CaseIDs)


		# In[ ]:
except Exception as e:
	try:
		print('Error code: '+str(e))
		if 'db' in dir():
			db.close()
			print('db close in error handling')
	except:
		print('about to exit. db close error.')

	try:
		if isinstance(e,str):
			print('uncaught error.')
			sys.exit(10)
		sys.exit(e)
	except Exception as e:	  # otherwise sys.exit will be caught
		print('sys.exit error')
