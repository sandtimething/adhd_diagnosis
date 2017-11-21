# Read RotY FROM hmd_data, compute log histogram of 360 bins
# Write 360 rows for each caseId into head_rot

# Usage:
# python rose_witherrorhandling.py caseids
#   Example:
#	   python rose_witherrorhandling.py 1
#	   python rose_witherrorhandling.py 1 2
try:
	def main(CaseIds, db_name='webtest', write_db = True):
		try:
			import pymysql
			import numpy as np
			import math
			import sys
		except Exception as e:
			raise Exception (1)

		# try:
			# CaseIds = sys.argv[1:]
			# CaseIds = [int(i) for i in CaseIds]
		# except Exception as e:
			# raise Exception (2)

		def connect(db_name):
			return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",database = db_name, user='cognitiveleap', password= 'QWE@123456')

		def logged_histogram(data):
			result = [0] * 360
			for d in data:
				result[int(d)] += 1
			result = [math.log(r + 1) for r in result]
			return result

		try:
			db = connect(db_name)
			# prepare a cursor object using cursor() method
			cursor = db.cursor()
		except Exception as e:
			raise Exception (3)

		for i in CaseIds:
			try:
				sql = """SELECT COUNT(*) FROM head_rot WHERE CaseId = """ + str(i)
				cursor.execute(sql)
				count = np.asarray(cursor.fetchall())[0][0]
			except Exception as e:
				raise Exception (4)

			try:
				tb = 'head_rot'
				if (count != 0) & (write_db):
					# caseid existed and corrupted, delete rows of this caseid and re-calculate
					sql = """DELETE FROM %s WHERE CaseId = """%tb + str(i)
					cursor.execute(sql)
					print('delete %d lines for caseid %d from %s'%(count,i,tb))
			except Exception as e:
				raise Exception (8)

			try:
				sql = """SELECT RotY FROM hmd_data WHERE CaseId = """ + str(i) + """ ORDER BY TimeLog"""
				cursor.execute(sql)
				# Fetch all the rows in a list of lists.
				results = np.asarray(cursor.fetchall())
			except Exception as e:
				raise Exception (4)

			matrix = np.asmatrix(results)
			if len(results) == 0:
				raise Exception (7)
				
			try:
				his = logged_histogram(np.asarray(matrix.T)[0])
			except Exception as e:
				raise Exception(9)
				
			try:
				if write_db:
					for j in range(360):
						sql = "INSERT INTO %s (CaseId,BlockNum, BinNum,Value) VALUES (%s, %s, %s, %s)" % (tb, i, 0, j, his[j])
						cursor.execute(sql)
					# Commit your changes in the database
					db.commit()
					print('insert %d lines for caseid %d from %s'%(360,i,tb))

			except Exception as e:
				raise Exception (5)		
				
		db.close()
		# print('db normal close')
		
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
	except Exception as e:      # otherwise sys.exit will be caught
		print('sys.exit error')

