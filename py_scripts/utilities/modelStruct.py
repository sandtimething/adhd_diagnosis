"""
	Class definition to specify model IO and paramaters.
"""

import json
import os
class modelParams(object):
	""" Class to hold params of the model 
		
		To initiate a modelParams class
		
		:param list(string) fields: field names for input to the model
		:param string model_type: string or array-liek, type(s) of the model 
		:param dict params: dict, paramaters of the model
		:param dict dict_init: Optional. If specified as a dictitonary, initiate from dict directly
	"""
	def __init__(self,fields = [],model_type='',params={}, dict_init=None):
		""" Initiate a modelParams class
		Args:
			fields: field names for input to the model
			model_type: string or array-liek, type(s) of the model 
			params: dict, paramaters of the model
			dict_init: if specified as a dictitonary, initiate from dict directly
			"""
		if dict_init:
			self.fields=dict_init['fields']
			self.model_type = dict_init['model_type']
			self.params = dict_init['params']
		else:
			self.fields = fields
			self.model_type = model_type
			self.params = params
		
	def json_decoders(jsonstr):
		""" Decode modelParams object from json string
		
		:param string jsonstr: json string to decode from
		:return: modelParams object 
		"""
		obj = json.loads(jsonstr)
		return modelParams(obj['fields'],obj['model_type'],obj['params'])
	
	def json_decoder(path):
		""" Decode modelParams object from file specified as path.
		
		:param string path: path to a file to decode from
		:return: modelParams object 
		"""
		obj = json.load(open(path,'r'))
		return modelParams(obj['fields'],obj['model_type'],obj['params'])
		
	def toJson(self):
		""" Serialize self as json string
		
		:return: json string encoding the modelParams object 
		"""
		return json.dumps(self.__dict__,indent=4)

class modelIO(object):
	""" Specify where to get input to the model from the db, and where to store output from the model into the db.
	
	:param list(string) input_tableNames: list of tables to fetch from
	:param list(string) input_fieldNames: list of fields to fetch, must be same length as input_tableNames
	:param list(string) input_whereClauses: list of where clauses to pass into sql query, must be same length as input_tableNames
	:param list(string) primaryKeys: list of primary key names for each table, must be same length as input_tableNames
	:param string output_tableName: name of output table
	:param list(string) output_fieldNames: list of fields in output data
	:param dict dict_init: Optional. If specified, initiate from the dictitonary directly.
	"""
	def __init__(self,input_tableNames = [],input_fieldNames=[],input_whereClauses=[],primaryKeys=[],output_tableName = '',output_fieldNames = [], dict_init=None):
		if dict_init:
			self.input_tables = dict_init['input_tables']
			self.input_fields = dict_init['input_fields']
			self.input_where_clauses = dict_init['input_where_clauses']
			self.input_primary_key = dict_init['input_primary_key']
			self.output_table = dict_init['output_table']
			self.output_fields = dict_init['output_fields']
		else:
			self.input_tables = input_tableNames
			self.input_fields = input_fieldNames
			self.input_where_clauses = input_whereClauses
			self.input_primary_key = primaryKeys
			self.output_table = output_tableName
			self.output_fields = output_fieldNames
		
	def getInput(self):
		""" Get members related to data input.
		
			:return: input_tables, input_fields, input_where_clauses, input_primary_key
		"""
		return self.input_tables, self.input_fields, self.input_where_clauses, self.input_primary_key
		
	def getAllFields(self):
		""" Get all field names of input_fields in a list
			
			:return: list(string). Containing input field names.
		"""
		result = []
		for f in self.input_fields:
			result += f
		return result
		
	def json_decoders(jsonstr):
		""" Decode modelIO object from json string
		
		:param string jsonstr: json string to decode from
		:return: modelIO object		
		"""
		obj = json.loads(jsonstr)
		return modelIO(obj['input_tables'],obj['input_fields'],obj['input_where_clauses'],\
		obj['input_primary_key'],obj['output_table'],obj['output_fields'])
		
	def json_decoder(path):
		""" Decode modelIO object from file specified as path.
		
		:param string path: path to a file to decode from
		:return: modelIO object 
		"""
		obj = json.load(open(path,'r'))
		return modelIO(obj['input_tables'],obj['input_fields'],obj['input_where_clauses'],\
		obj['input_primary_key'],obj['output_table'],obj['output_fields'])
	
	def toJson(self):
		""" Serialize self as json string
		
		:return: json string encoding the modelIO object 
		"""
		return json.dumps(self.__dict__,indent=4)

class bayesModel(object):
	""" Class containing modelParams and modelIO
	
		Be careful that we have to assign a new empty object to each argument, otherwise the 'append' operation will mess up the default.
		
		
		Example::
		
			# load basic setup from modelConfig.example
			# use multiply method by default
			model = bayesModel.json_decoder('../config/modelConfig.example',True)
			
			# add features 
			model.addBayesFeature('DPrime')
			model.addBayesFeature('OmissionErrors','beta',[-1e-6,52],out_f=['OmissionRawProbabilityADHD','OmissionRawProbabilityHealthy'])
			model.addBayesFeature('CommissionErrorsRtVariability','beta',[-1e-6,1])
			model.addBayesFeature('C')
			
			# add four last fields
			model.addBayesExtra()
			
			# change method. Currently known methods are 'multiply', 'mean' and 'median'.
			model.changeMethod('mean')
			
			# use the model
			result_new = bayes_model_vrclassroom.main(caseIds,modelobj=newmodel)
	"""
	def __init__(self,modIO = None,modParams = None):
		""" Have to assign a new empty object to each argument,
			Otherwise the 'append' operation will mess up the default
			"""
		if modIO is None:
			# instead of modIO = modelIO()
			modIO = modelIO(input_tableNames = [],input_fieldNames=[],input_whereClauses=[],primaryKeys=[],output_tableName = '',output_fieldNames = [])
		if modParams is None:
			# instead of modParams = modelParams()
			modParams = modelParams(fields = [],model_type='Naive_Bayes',params={'dist_param':[],'distribution':[],'method':''})
		self.modParams = modParams
		self.modIO = modIO
		
	def toJson(self,path=None):
		""" Serialize the object to json string.
		
			If path is given, write to file and return 0; otherwise return string
			
			:return: json string encoding the modelParams object or 0
		"""
		if path is not None:
			json.dump(self, open(path,'w'), default=lambda o: o.__dict__, sort_keys=True, indent=4)
			return 0
		else:
			return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
	
	def json_decoder(path, only_input = False):
		""" Decode modelParams object from file specified as path.
		
			:param string path: path to file to decode from
			:param bool only_input: Optional, default False. If True, only load the input fields from the file. This is handy with addBayesFeature function to add features interactively.
			
			:return: bayesModel object
			
			Usage::
				modelobj = bayesModel.json_decoder('../config/modelConfig')
		"""
		print('looking for config file at '+os.path.abspath(path))
		obj = json.load(open(path,'r'))
		if only_input:
			obj['modIO']['output_fields'] = []
			obj['modParams'] = {'fields':[], 'model_type':'Naive_Bayes', \
								'params':{'dist_param':[],'distribution':[],'method':obj['modParams']['params']['method']}}
			return bayesModel(modIO = modelIO(dict_init=obj['modIO']),modParams=modelParams(dict_init=obj['modParams']))
		else:
			return bayesModel(modIO = modelIO(dict_init=obj['modIO']),modParams=modelParams(dict_init=obj['modParams']))
		
	def addBayesFeature(self,feat,distribution = 'gauss', dist_param = [],out_f = None):
		""" Add features to this bayes model, including modIO.output_fields, modParams.fields and modParams.params. 
		
			By default, new output_fields will be ``feat + 'RawProbabilityADHD'`` and ``feat + 'RawProbabilityHealthy'``
		
			:param string feat: name of the feature. This will be appended to modParams.fields
			:param string distribution: name of the distribution for this feature. 
			:param list dist_param: list of paramaters of the distribution.
			:param list(string) out_f: Optional. If specified, append this to modIO.output_fields. Must be None or list of length 2.
			
		"""
		if not isinstance(out_f,list):
			self.modIO.output_fields += [feat + 'RawProbabilityADHD']
			self.modIO.output_fields += [feat + 'RawProbabilityHealthy']
		elif len(out_f) != 2:
			raise ValueError('out_f should be list of length 2')
		else:
			self.modIO.output_fields += out_f
		self.modParams.fields += [feat]
		self.modParams.params['dist_param'] += [dist_param]
		self.modParams.params['distribution'] += [distribution]
		
	def rmBayesFeature(self,feat):
		""" Remove feature from this bayes model. 
		
			By default, remove modIO.output_fields, modParams.fields and modParams.params.
		
		"""
		if feat in self.modParams.fields:
			ind = self.modParams.fields.index(feat)
			self.modParams.fields.pop(ind)
			self.modIO.output_fields.remove(feat + 'RawProbabilityADHD')
			self.modIO.output_fields.remove(feat + 'RawProbabilityHealthy')
			self.modParams.params['dist_param'].pop(ind)
			self.modParams.params['distribution'].pop(ind)
		else:
			print(feat+' is not present in modelParams.fields')
			
	def addBayesExtra(self):
		""" Append to the modIO.output_fields extra fields to stay consistant with database.
		
		"""
		self.modIO.output_fields += ["TotalRawProbabilityADHD",
            "TotalRawProbabilityHealthy",
            "Priors",
            "FinalProbabilityOfADHD"]
			
	def changeMethod(self,method):
		""" Change ``method`` of modParams.params['method']
		
			:param string method: name of the method. Currently known methods are 'multiply', 'mean' and 'median'.
		"""
		self.modParams.params['method'] = method