"""
	Module for interfaces with database.
	
	For data object, ``pandas.DataFrame`` is used in the module. For database connection, ``sqlalchemy`` is used.
	
	Class definition of ``db_manager``, as well as utililies function ``get_adhd`` and ``get_model_input`` is included in the module.

"""
# coding: utf-8

# In[2]:

# Switch to sqlalchemy to accomodate dataframe.to_sql function
# not perfect with error handling and managing connection when close
import pandas as pd
import os
import numpy as np
import pymysql
from sqlalchemy import create_engine
from py_scripts.utilities.modelStruct import modelIO as modelIO


# In[30]:

# load data from database
# from pandas import DataFrame

class db_manager(object):
	def __init__(self,schema_name = 'null'):
		"""Initiate db_manager object. 
		
		Connect if schema_name is provided"""
		self.connected = False
		self.__schema_name = ''
		self.__db = ''
		self.__cursor = ''
		self.__engine = ''
		if schema_name != 'null':
			self.connect(schema_name)
		
			
	def close(self):
		"""Close the connection manually.
		
		Not recommended. sqlalchemy will take care of the engine.dispose automatically."""
		if self.connected:
#				 self.__db.close()
			self.__engine.dispose()
			self.connected = False
				
	def __del__(self):
		"""Destructor function. Manual dispose() is unneccessary"""
		# self.close()   
		self.connected = False
		print('db close in destructor')
		

	def connect(self,schema_name = ''):
		"""Connect to the specified schema. Return exception 3 if unsuccessful
		"""
		if schema_name == '':
			if self.__schema_name=='':
				print('Specify name of the schema/db first')
				return
			else:
				print('Use the previous schema_name %s'%self.__schema_name)
		else:
			if self.__schema_name == schema_name:
					print('Seem to be the same engine')
			self.__schema_name = schema_name
			
		if self.connected:
			self.close()
		try:
#			 self.__db = pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",\
#								  database = schema_name, user='cognitiveleap', password= 'QWE@123456')
			#self.__cursor = self.__db.cursor()
			self.__engine = create_engine("mysql+pymysql://cognitiveleap:QWE@123456@rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com/%s"%self.__schema_name)
		except Exception as e:
			raise Exception (3)
		self.connected = True
	
	def get_sql_str_select(self,table_name, field_names = ['*'],where_ind_name = None, where_ind_value = None, where_clause = None):
		"""Get a sql SELECT query string.
		
		:param string table_name: name of the table in db
		:param list(string) field_names: list of fields to fetch from the table. Optional. Fetch all fields by default (default ``['*']``).
		:param string where_ind_name,where_ind_value: Optional, equivalent to ``where where_ind_name = where_ind_value`` clause in sql
		:param string where_clause: Optional, directly converted to where clause
		:return: sql SELECT string 
		"""
		if (where_ind_name!=None) & (where_ind_value!=None):
			sql_str = 'SELECT %s FROM %s.%s WHERE %s = %s'%(','.join([str(x) for x in field_names]),													self.__schema_name,														table_name,														 where_ind_name,														 where_ind_value)
		else:
			sql_str = 'SELECT %s FROM %s.%s %s'%(','.join([str(x) for x in field_names]),													self.__schema_name,														table_name, where_clause)
		# print(sql_str)
		return sql_str
		
	def fetch_table(self,table_name, field_names = None,where_ind_name = None, where_ind_value = None, where_clause = None, primary_key = 'Id', return_array = False):
		"""Fetch specific table from the schema and return data in dataframe. 
		
		:param string table_name: name of the table in db
		:param list(string) field_names: list of fields to fetch from the table. Optional. Fetch all fields by default (default ``['*']``).
		:param string where_ind_name,where_ind_value: Optional, equivalent to ``where where_ind_name = where_ind_value`` clause in sql
		:param string where_clause: Optional, directly converted to where clause
		:param string primary_key: Optional, specify which field to index the result. Default 'Id'. Specify 'CaseId' to index with caseid.
		:param bool return_array: Optional, default False. If True, return np.array instead of pd.DataFrame
		:return: pd.DataFrame of the queried data
		
		Note that if *primary_key* (e.g. CaseId) is not present in *field_names*, then it is automatically appended and returned in the result. 
		
		Example::
		
			# fetch C and beta from signal_detection table
			mydb.fetch_table('signal_detection',['C','beta'],'Block',0)
			
			# fetch everything from bayes_probabilities, with CaseId as index
			mydb.fetch_table('bayes_probabilities',primary_key = 'CaseId')
		"""
		if field_names is None:
			field_names = ['*']
		field_names = field_names.copy()
		if (primary_key not in field_names) and (field_names != ['*']):
			field_names.append(primary_key)
		if self.connected:
			fetch_data = pd.read_sql_query(sql=self.get_sql_str_select(table_name,field_names,where_ind_name,where_ind_value,where_clause),con=self.__engine,index_col = primary_key)
			if not return_array:
				return fetch_data
			else:
				return np.asarray(fetch_data)
		else:
			print ('db not connected yet. Do connect first')
		
	def sql_query_fetch_list(self,sql):
		"""Do sql query fetch, and return results in np.array	 
		
		:param string sql: sql query to execute
		"""
		if self.connected:
#			 self.__cursor.execute(sql)
#			 results = np.asarray(self.__cursor.fetchall())
			cur = self.__engine.execute(sql)
			results = np.asarray(cur.fetchall())
			cur.close()
			return results
		else:
			print ('db not connected yet. Do connect first')
	
	def sql_query_fetch_df(self,sql,primary_key = None):
		"""Do sql query fetch and return results in pd.dataframe
		
		Please include the field of primary_key in sql query if necessary
		
		:param string sql: sql query to execute
		:param string primary_key: name of field to index the dataframe
		
		"""
		if not self.connected:
			print ('db not connected yet. Do connect first')
			return
		results = pd.read_sql_query(sql,self.__engine,index_col = primary_key)
		return results
	
	def sql_nofetch(self,sql):
		"""Do sql without fetch (e.g. delete, truncate)
		
		:param string sql: sql query to execute
		"""
		if not self.connected:
			print ('db not connected yet. Do connect first')
			return
		cur = self.__engine.execute(sql)
		cur.close()
	
	def truncate_table(self,table_name):
		""""Truncate table
		
		:param string table_name: table_name to truncate
		
		"""
		sql = 'truncate table '+table_name
		self.sql_nofetch(sql)
		print('truncate table %s done'%table_name)
		
	
	def check_existed(self,table_name,field_name,ids):
		"""
		Check if ids in certain field existed in the db. Return list if ids is a list
		"""
		if not self.connected:
			print ('db not connected yet. Do connect first')
			return
		existed = []
		existed_count = []
		for ind in ids:
			try:
				sql = """SELECT COUNT(*) FROM %s.%s WHERE %s = %s"""%(self.__schema_name,																	  table_name,																	 field_name,																	 ind)
#				 self.__cursor.execute(sql)
#				 count = np.asarray(self.__cursor.fetchall())[0][0]
				cur = self.__engine.execute(sql)
				count = np.asarray(cur.fetchall())[0][0]
				cur.close()
			except Exception as e:
				raise Exception (4)
			existed.append(count!=0)
			existed_count.append(count)
		return np.asarray(existed),np.asarray(existed_count)
	
	def insert_table(self,df, table_name,index_name='Id',del_row_if_exist = True):
		"""Insert data frame into table
		
		Assume table existed.
		
		:param pd.DataFrame df: dataframe object to insert to db
		:param string table_name: name of table to insert
		:param string index_name: Optional, default None. Field name to check repeated rows in db. Not necessarily the actual index, could be e.g. 'CaseId'. If None, use original index.
		:param bool del_row_if_exist: Optional, default True. Specify whether to delete the existing rows (e.g. rows with the same caseid)
		
		"""
		if not self.connected:
			print ('db not connected yet. Do connect first')
			return
		if (index_name == 'id' ) or (index_name=='Id'):
			ind_list = np.asarray(pd.unique(df.index))
			ind_serie = df.index
		else:
			ind_list = np.asarray(pd.unique(df[index_name]))
			ind_serie = df[index_name]
		exist_list,exist_count_list = self.check_existed(table_name,index_name,ind_list)
		if del_row_if_exist & (len(exist_list)>0):
			for ind,exist_count in zip(ind_list[exist_list],exist_count_list[exist_list]):
				sql = """DELETE FROM %s.%s WHERE %s = %s"""%(self.__schema_name,																	  table_name,																	 index_name,																	 ind)
#				 self.__cursor.execute(sql)
				cur = self.__engine.execute(sql)
				print('delete %d rows for %s %d'%(exist_count,index_name,ind))
				cur.close()
		else:
			
			df = df[ind_serie.isin(ind_list[exist_list==False])]
			
		df.to_sql(name = table_name, con = self.__engine, schema=self.__schema_name, if_exists='append', index=False)	# do not insert index as in df, use the auto-incrementing id in db
		print('%d lines insertion done'%df.shape[0])
	   
	def get_engine(self):
		"""Return engine, not recommended"""
		return self.__engine
	   
	
def get_adhd(caseids,mydb=None):
	""" Do cross-table query and get adhd diagnosis for specified caseids.
	
	:param list(int) caseids: list of caseids to get data for
	:param db_manager mydb: Optional. db_manager object to connect to db. If not specified, connect to 'vrclassroomdata' by default.
	
	:return: pd.DataFrame with caseid as index and ADHDDiagnose as a column.
	"""
	if mydb is None:
		mydb = db_manager('vrclassroomdata')
	cids = ','.join([str(x) for x in caseids])
	this_df = mydb.sql_query_fetch_df("select a.Id, b.ADHDDiagnose  from `case` a, patient b where b.Id=a.SubjectId and a.Id in (%s) order by field (a.Id,%s)"%(cids,cids),primary_key='Id')
	this_df.ADHDDiagnose = this_df.ADHDDiagnose>0
	return this_df
	
def get_model_input(modIO,CaseIds,mydb=None):
	""" Get data according to modelIO struct.
	
		:param modelIO modIO: modelIO object, containing which fields to fetch
		:param list(int) CaseIds: list of caseids to get data for
		:return: pd.DataFrame object, indexed by caseid, columned by fields in modIO
	"""
	if mydb is None:
		mydb = db_manager('vrclassroomdata')
	if not isinstance(modIO,modelIO):
		raise ValueError('modIO must be an instance of modelIO')
	if len(CaseIds) == 0:
		return None
	dfs = []
	tableNames,fieldNames,whereClauses,primary_keys = modIO.getInput()
	for table,fields,where_c,p_key in zip(tableNames,fieldNames,whereClauses,primary_keys):
		if fields == ['CaseId','ADHDDiagnose'] :
			# if wants to fetch caseid and adhddiagnose, do cross table query 
			this_df = get_adhd(CaseIds,mydb)
		else:
			# get spcific fields from specific table
			this_df = mydb.fetch_table(table,fields,where_clause=where_c,primary_key=p_key)
		this_df = this_df.iloc[[np.where(this_df.index==cid)[0][0] for cid in CaseIds],:]	# reorder rows of caseids according to the caseids specified
		dfs.append(this_df)
	df = pd.concat(dfs,axis=1)
	return df
	

def get_training_ids(mydb):
	"""
		Get training caseids from training_set table
		
		:return: np.array for training caseids
	"""
	return list(mydb.fetch_table(table_name='training_set',field_names=['CaseId']).CaseId)
	
def get_hmd_data(CaseIds, mydb):
	""" Fetch data from db for each caseid in CaseIds
	"""
	data = []
	for cid in CaseIds:
		this_data = mydb.fetch_table(table_name='hmd_data',field_names=['PosX', 'PosY', 'PosZ', 'RotX', 'RotY', 'RotZ'],where_ind_name = 'CaseId', where_ind_value = cid)
		data.append(this_data)
	return data

# # Examples 

# ## copy tables between two db 

# In[9]:
if __name__ == '__main__':
	db_rnd = db_manager('vrclassroomdata')
	db_web = db_manager('webtest')
	CaseIds = [40]
	for cid in CaseIds:
		df = db_rnd.fetch_table(table_name = 'trial_data',where_ind_name = 'CasdId', where_ind_value = cid )
		db_web.insert_table(df, 'trial_data', 'CasdId')
		print(str(cid)+' done')

	db_rnd.close()
	db_web.close()


	# ## load data with fetch table

	# In[31]:

	mydb = db_manager('vrclassroomdata')
	# print(mydb.connected)
	hmd_data = mydb.fetch_table(table_name='hmd_data',						where_ind_name = 'CasdId',						where_ind_value = 68,						 field_names = ['Id','PosX', 'PosY', 'PosZ', 'RotX', 'RotY', 'RotZ'],						primary_key='Id')
	print(hmd_data.head())
	head_features_new = mydb.fetch_table(table_name='head_features_new',primary_key='id')
	print(head_features_new.head())


	# ## Insertion

	# In[32]:

	# some dummy operation, do insertion
	mydb.insert_table(table_name='head_features_new',df=head_features_new[head_features_new['CaseId']==68],index_name='CaseId',del_row_if_exist=True)


	# ## Other stuff

	# In[33]:

	mydb.check_existed(field_name='CaseId',ids=[1,2,3,40,100],table_name='head_features_new')


	# In[34]:

	mydb.sql_query_fetch_df(sql='SELECT * FROM vrclassroomdata.head_features_new WHERE CaseId = 40')


	# In[24]:

	mydb.connected #check connectivity
	mydb.close() # close engine
	mydb.check_existed(field_name='CaseId',ids=[1,2,3,40,100],table_name='head_features_new') # try a command
	mydb.connect()
	print(mydb.check_existed(field_name='CaseId',ids=[1,2,3,40,100],table_name='head_features_new')) # try again
	mydb.close()


# # Old stuff 

# In[2]:

# load csv from folders
# obsolete
def loadFiles(root="data/TAIWAN_RAW_DATA/ADHD"):
	"""
		Out-of-date function to load data from csv in folders
	"""
	data_rt = [] # realtime.csv
	data_trial = [] # trialdata.csv
	data_id = [] # caseid/subjectid
	RealTime = "A2RealTime_"
	TrialData = "A2TrialData_"
	folder_list = os.listdir(root) # list of subfolders in the root
	for folders in folder_list:
		folders_path = os.path.join(root,folders)
		if folders.find("pass") != -1:
			continue
			
		try:
			data_rt.append(pd.read_csv(os.path.join
								   (folders_path,
								   RealTime+folders[3:]+".csv")))
			data_trial.append(pd.read_csv(os.path.join
									  (folders_path,
									   TrialData+folders[3:]+".csv")))
			data_id.append(int(folders.split('_')[1]))
		except:
			print(os.path.join(folders_path,TrialData+folders[3:]+".csv"))
			
	return data_rt,data_trial,data_id,folder_list

