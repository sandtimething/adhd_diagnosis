"""
	calcHeadFeature
	
	Calculate head features of test caseids and write to db. 
	
	Origin from `percentile_witherrorhandling_2.py`. - Tom 08/31/2017

"""
import sys
import pymysql
import numpy as np
import pdb
import pandas as pd
# sys.path.insert(0,'../')
from py_scripts.utilities.loadData import db_manager, get_hmd_data
from py_scripts.utilities.mathHelper import calcAngleDeg,euler2Vec,smooth, pi, ceil
		
#functions for feature extracting
threshold = 1 / 50 / 100
def percentage_over_threshold_head(data):
	x = data[ : , 0]
	y = data[ : , 1]
	z = data[ : , 2]
	diffx = np.diff(x)
	diffy = np.diff(y)
	diffz = np.diff(z)
	count = 0
	for i in range(0, len(diffx)):
		if (diffx[i]**2 + diffy[i]**2 + diffz[i]**2)**0.5 > threshold:
			count += 1
	return count / len(diffx)

def head_length(data):
	x = data[ : , 0]
	y = data[ : , 1]
	z = data[ : , 2]
	diffx = np.diff(x)
	diffy = np.diff(y)
	diffz = np.diff(z)
	l = 0;
	for i in range(0, len(diffx)):
		l += (diffx[i]**2 + diffy[i]**2 + diffz[i]**2)**0.5
	return l

def rot_sum_degrees(data):
	y = data[ : , 3]
	y = [item - 360 if item > 180 else item for item in y]
	diff = np.diff(y)
	diff_abs = [abs(i) for i in diff]
	return sum(diff_abs)



low = -10# the right angle line that the kid cross, in degrees
high = -low # left one


def head_rot(data):
	y = data[ : , 4]
	y = [item - 360 if item > 180 else item for item in y]
	m = np.mean(y)
	count = 0
	at_right = False
	at_left = False
	for i in y:
		if i >= m + high and not at_right:
			count += 1
			at_right = True
			at_left = False
		if i <= m - low and not at_left:
			count += 1
			at_left = True
			at_right = False
	return count

def distracted(data):
	threshold = 15
	headEuler = data[:, 3:6]
	tarVec = np.tile([0,2.03,4.91],[len(headEuler),1])
	headPos = data[:, 0:3]
	tarVec = tarVec - headPos
	
	headVec = np.array([euler2Vec(angle) for angle in headEuler])
	deg = calcAngleDeg(tarVec,headVec)
	count_percentage = 0
	for i in deg:
		if i > threshold:
			count_percentage += 1
			
	count = 0
	thre_subtracted = [d - threshold for d in deg]
	for i in range(len(thre_subtracted) - 1):
		if thre_subtracted[i] * thre_subtracted[i + 1] < 0:
			count += 1
	#print('max deg %f. percentage %f'%(max(deg),count_percentage/ len(deg)))
	return count_percentage / len(deg), count / 2
	
def distracted_rotOnly(data):
	threshold_y = 10
	threshold_x = 20
	headEuler = data[:, 3:6]
	headEuler[headEuler[:,0]>180,0] -= 360
	headEuler[headEuler[:,1]>180,1] -= 360
	#headEuler[:,1] = np.asarray([x-360 for x in headEuler[:,1] if x > 180])
	distract_list = np.logical_or(np.abs(headEuler[:,0]+5)>threshold_x, np.abs(headEuler[:,1])>threshold_y)
	#pdb.set_trace()
	count_percentage = np.mean(distract_list)
	zero_crossing = len([ind for ind in range(1,len(distract_list)) if distract_list[ind]!=distract_list[ind-1]])
	return count_percentage,zero_crossing/2

def isMoveLarge(data):
	headPos = data[:,[0,2]]
	headPos_smooth = np.stack((smooth(headPos[:,0],15,'gaussian',4),smooth(headPos[:,1],15,'gaussian',4)),axis=1)
	# xrange = np.max(headPos_smooth[:,0]) - np.min(headPos_smooth[:,0])
	# zrange = np.max(headPos_smooth[:,1]) - np.min(headPos_smooth[:,1])
	xrange1 = np.max(headPos_smooth[:,0]) - headPos_smooth[0,0]
	zrange1 = np.max(headPos_smooth[:,1]) - headPos_smooth[0,1]
	xrange2 = headPos_smooth[0,0] - np.min(headPos_smooth[:,0])
	zrange2 = headPos_smooth[0,1] - np.min(headPos_smooth[:,1])
	#print('%f,%f '%(xrange,zrange))
	# isMoveLarge = (xrange > 0.4) | (zrange > 0.4)
	isMoveLarge = (xrange1 > 0.2) | (zrange1 > 0.2) | (xrange2 > 0.2) | (zrange2 > 0.2)
	return isMoveLarge

	
def process_feature(CaseIds, data):
	""" Process raw data in db (should be testdb), write features to db (testdb)"""
	results = []
	count = 0
	columns = ['BlockNum', 'PathLen', 'TimeActive', 'NumRot', 'TotalDeg', 'PercentageDistracted', 'IsMoveRangeLarge', 'CaseId']
	try:
		for i in range(len(CaseIds)):
			nparray = np.asarray(data[i])
			result = np.zeros(len(columns))
			result[1] = head_length(nparray)
			result[2] = percentage_over_threshold_head(nparray)
			result[3] = head_rot(nparray)
			result[4] = rot_sum_degrees(nparray)
			result[5] = distracted_rotOnly(nparray)[0]        # percentage time distracted
			# print(distracted(data[i]))
			# print(distracted_rotOnly(data[i]))
			result[6] = isMoveLarge(nparray)        # whether the kid is moving in larger range than normal
			result[7] = CaseIds[i]
			results.append(result)
		results = pd.DataFrame(np.asarray(results),columns=columns,index=CaseIds)
	except Exception as e:
		raise Exception(9)

	return results
	
def insert_features(df,mydb):
	""" Insert result dataframe to db
	"""
	mydb.insert_table(df,'head_features',index_name='CaseId',del_row_if_exist = True)
	

def main(testCaseIds,testdb_name='webdarintest', write_db = False):
	""" Calculate features for incoming test cases.
	
	Args:
		testCaseIds: list. Raw data for case ids in the array must be found in testdb. 
		testdb_name: name of the testing database to fetch raw data and store features. Should contain raw data (hmd_data) for test cases.
		write_db: if true, write result to database (features for testCaseIds)
	"""
	
	# obtain raw data
	mydb = db_manager(testdb_name)
	data = get_hmd_data(testCaseIds,mydb)
	
	# compute features
	features = process_feature(testCaseIds, data)
	
	# insert to db
	if write_db:
		insert_features(features,mydb)
	
	return features
	