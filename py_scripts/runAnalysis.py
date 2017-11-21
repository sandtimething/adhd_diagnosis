# script for running the scripts in one to pack into exe
try:
	import sys
	import os
	from py_scripts import rose_witherrorhandling as rose_witherrorhandling
	from py_scripts import percentile_witherrorhandling_2 as percentile_witherrorhandling_2
	from py_scripts import signal_detection_vrclassroom as signal_detection_vrclassroom
	from py_scripts import bayes_model_vrclassroom as bayes_model_vrclassroom
	from py_scripts import calcHeadFeature as calcHeadFeature
	from py_scripts import calcPercentile as calcPercentile
	import numpy as np
	import argparse
	import warnings
except Exception as e:
	raise Exception (1)
	
useTrain = False
testmode = False
trainDb = 'vrclassroomdata'
testDb = 'webdarintest'
CaseIds = []
configPath='modelConfig'
def getPath():
	path = sys.path[0]+'\\'
	print (sys.path)
	if os.path.isdir(path):
		return path
	elif os.path.isfile(path):
	  	return os.path.dirname(path)
def main(CaseIds,traindb,testdb, testmode=False, configPath='../config/modelConfig'):
	print('rose started\n')
	rose_witherrorhandling.main (CaseIds,testdb, write_db=not testmode)
	print('\n')
	print('percentile started\n')
	hfs = calcHeadFeature.main(testCaseIds=list(CaseIds),testdb_name=testdb,write_db=True)
	percentiles = calcPercentile.main(CaseIds,traindb_name = traindb, testdb_name=testdb,\
                                  use_training_cases = useTrain, write_db = not testmode)
	print('\n')
	print('signal detection started\n')
	signal_detection_vrclassroom.main  (CaseIds, testdb, write_db=not testmode)
	print('\n')
	print('bayes started\n')
	t = bayes_model_vrclassroom.main  (CaseIds, traindb, testdb, write_db=not testmode, use_df=True, config_path=configPath)
	if testmode:
		print ('\nresult from percentile: \n')
		print(percentiles)
		print ('\nresult from bayes: \n')
		print(t)
	return percentiles,t
	
if __name__ == '__main__':
	curdir = getPath()
	print('running from path '+curdir)
	try:
		parser = argparse.ArgumentParser(description='run analysis on the new test case')
		parser.add_argument('caseids',nargs='*',type=int, help='list of caseids to test')
		parser.add_argument('dummy',nargs='*')
		parser.add_argument('--traindb',nargs=1,type=str,default=['vrclassroomdata'], help='train db name, default: vrclassroomdata')
		parser.add_argument('--testdb',nargs=1,type=str,default = ['webdarintest'],help='test db name, default: webdarintest')
		parser.add_argument('--configpath',nargs=1,type=str,help='path to config file, default: ./modelConfig',default=['./modelConfig'])
		parser.add_argument('--testmode',action="store_true",help='add this to enter testmode')
		arg=parser.parse_known_args(sys.argv[1:])
		
		if arg[0].caseids:
			CaseIds = arg[0].caseids
			CaseIds = list(np.unique(CaseIds))
		if arg[0].traindb:
			trainDb = arg[0].traindb[0]
		if arg[0].testdb:
			testDb = arg[0].testdb[0]
		if arg[0].configpath:
			configPath = arg[0].configpath[0]
		if arg[0].testmode:
			testmode = arg[0].testmode
	except Exception as e:
		raise Exception (2)

	if testmode:
		print('running as test mode, will not write to db')
	warnings.filterwarnings("ignore")
	
	main(CaseIds,trainDb,testDb,testmode,curdir+configPath)
	print('\n')
	print('all done\n')
	sys.stdout.flush()
	sys.exit(0)
