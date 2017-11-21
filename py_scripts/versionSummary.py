
import math
import json


import pymysql
import numpy as np


def main(version,configPath,traindb_name = 'vrclassroomdata', testdb_name='webdarintest'):

	#initializes connection info for database
	def connect(schemaName):
		try:
			return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
								 database = schemaName,
								 user='cognitiveleap',
								 password= 'QWE@123456')
		except Exception as e:
			raise Exception(3)


	def fetchAttensionTime(db):
		cursor = db.cursor()
		sql='''SELECT PercentageDistracted FROM head_features'''
		cursor.execute(sql)
		results = np.asarray(cursor.fetchall())
		cursor.close()
		return results

	def addOrUpdateVersionSummary(db,data):
		cursor=db.cursor()
		sql='''DELETE FROM versionsummary where version=%s'''%data[0]
		cursor.execute(sql)
		sql='''INSERT INTO 
				versionsummary
				(version,modelconfig,averpercentageDistracted) 
				VALUES(%s,%s,%s)'''%(data[0],data[1],data[2])
		cursor.execute(sql)
		cursor.close()


	#read Model
	modelJson=json.dumps(open(configPath,'r').read())
	
	try:
		traindb=connect(traindb_name)
		testdb=connect(testdb_name)


		attensionTimes=fetchAttensionTime(traindb)
		averageAttensionTime=np.mean(attensionTimes)
		print('avg:%s'%averageAttensionTime)
		addOrUpdateVersionSummary(traindb,(version,modelJson,averageAttensionTime))
		addOrUpdateVersionSummary(testdb,(version,modelJson,averageAttensionTime))

		traindb.commit()
		testdb.commit()
	except Exception as e:
		raise e
	finally:
		traindb.close()
		testdb.close()




if __name__ == '__main__':
				main(version=1,configPath='../config/modelConfig_0904')