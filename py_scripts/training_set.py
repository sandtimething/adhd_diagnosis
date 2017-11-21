#This script can be optionally run before any of the Bayes scripts.
#Simply supply whatever CaseIds you wish to be included in the training set.

# Usage:
#	 python training_set.py caseids
#	 Example:
#			 python training_set.py 1
#			 python training_set.py 1 2
#		python training_set.py 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 41 42 43 44 45 46 47 49 50 51 52 61 62 63 64 65 66 67 68




try:
# ### 1. Necessary Imports:

# In[1]:

	def main(CaseIDs, db_name = 'vrclassroomdata', cross_db = True):
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


		# ### 2. Code to get CaseIds from command line

		# In[2]:

		

		#initializes connection info for database
		def connect():
			try:
				return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
									 database = db_name,
									 user='cognitiveleap',
									 password= 'QWE@123456')
			except Exception as e:
				raise Exception(3)



		def insert(data):
			# search through the table and delete already existed caseids
			count = 0
			try:
			# truncate the whole table
				sql = 'TRUNCATE TABLE training_set'
				cursor.execute(sql)
				# for caseid in np.unique(CaseIDs):
					# if has_existing_CaseId(caseid):
						# sql = """DELETE FROM training_set WHERE CaseID = """ + str(caseid)
						# cursor.execute(sql)
						# count += 1
			except Exception as e:
				raise Exception(8)
			# print ('%d caseids got deleted'%cursor.fetchone())

			count = 0

			# do insertion
			try:

				for i in range(len(data)):
					tb = 'training_set'
					sql = "INSERT INTO %s"%tb
					sql += "(CaseId, Diagnosis, Severity)"
					sql += "VALUES (%s, %s, %s)" % (data[i, 0],
													data[i, 2],
													data[i, 1])
					#sql = "SHOW COLUMNS FROM "
					# Execute the SQL command
					cursor.execute(sql)
					count += 1

				print('%d rows inserted into database %s'%(count,tb))

			except Exception as e:
				raise Exception(5)


		# In[11]:

		#This function ensures that there are no repeated caseIds in the data we are trying to insert
		#Yiming changed it in the branch to detect whether each caseId is detected
		# Should return true whenever some rows are present -Tom
		def has_existing_CaseId(caseid):


			# Prepare SQL query to INSERT a record into the database.
			try:
				sql = """SELECT * FROM training_set WHERE CaseId = %s""" % (caseid)
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) != 0: #if there's someone's data in database
					return True
				else: return False
			except Exception as e:
				raise Exception(4)

		def get_data(caseIds):

			dataADHD = []

			for i in caseIds:
				if not cross_db:
					sql = """SELECT Id, ADHDDiagnose FROM patient WHERE Id = """ + str(i)
				else:
					sql = "select a.Id, b.ADHDDiagnose  from `case` a, patient b where b.Id=a.SubjectId and a.Id="+str(i)
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
				if len(results) == 0: #if there's not someone's data in database
					raise Exception(7)
				dataADHD.append(results)
			dataADHD = np.concatenate(np.asarray(dataADHD))
			data = np.zeros((np.shape(dataADHD)[0], 3))
			data[:, 0:2] = dataADHD[:, 0:2]
			for i in np.arange(len(caseIds)):
				if dataADHD[i, 1] > 0:
					data[i, 2] = 1
				else:
					data[i, 2] = 0
			return data

		# ### 4. Main Function:

		# In[12]:





		db = connect()
		cursor = db.cursor()

		# In[14]:

		#calls main function, stores data for testing
		training_set = get_data(CaseIDs)
		insert(training_set)

		db.commit()

		db.close()

	
	if __name__ == '__main__':
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
	except:
		print('sys.exit error')