import numpy as np
import json
class featureFields(object):
	"""
	Serialized Format for database dynamic query

	tables:{"tables": [{"name": "cpt_output_results", "fields": ["OmissionErrors"], "whereclauses": ["block=0"], "wherekwargs": {}}]}
	"""
	def __init__(self,tableFieldsStruct=[],outputFieldsStruct=[]):
		self.inputTables=tableFieldsStruct
		self.outputTables=outputFieldsStruct

	def addTableFields(self,tableName,tableFields=None,*_clauses, **kwargs):
		self.inputTables+=[dict(name=tableName,fields=tableFields,whereclauses=_clauses,wherekwargs=kwargs)]

	def addSql(self,sqlString,tableFields=None):
		self.inputTables+=[dict(name='sqlString',value=sqlString)]


	def addOutputFields(self,tableName,tableFields=None):
		self.outputTables+=[dict(name=tableName,fields=tableFields)]

	def get_all_input_fields(self):
		"""
		Return All Fields
		
		Example:['PathLen' 'TimeActive' 'NumRot' 'TotalDeg' 'OmissionErrors'
 		'CommissionErrors' 'TargetsRT' 'TargetsRtVariability' 'CommissionErrorsRT'
 		'CommissionErrorsRtVariability']
		"""
		return np.concatenate([table["fields"] for table in self.inputTables])

	def get_all_output_fields(self):
		return np.concatenate([table["fields"] for table in self.outputTables])



