"""
This code is deprecated. Use ``calcHeadFeature`` and ``calcPercentile`` instead.

Read from hmd_data, compute precentile data: PerPathLen, PerTimeActive, PerNumRot, PerTotalDeg
Write to head_features

Usage:
	python percentile_witherrorhandling_2.py caseids

Example:
  percentile_witherrorhandling_2.py 1
  percentile_witherrorhandling_2.py 1 2
	  """
try:
    try:
        import sys
        import pymysql
        import numpy as np
        import scipy.signal as signal
        import pdb
        from math import cos, sin, acos, pi, ceil
        import pandas as pd
    except Exception as e:
        raise Exception(1)
        
    def main(testCaseIds,traindb_name = 'vrclassroomdata', testdb_name='webtest', use_training_cases = False, write_db = False):
        """ 
		calculate features and percentile for incoming test cases
			
		Args:
			testCaseIds: array-like. Raw data for case ids in the array must be found in testdb. 
			traindb_name: name of the training database to fetch training set. Should already computed head_features table for training cases
			testdb_name: name of the testing database to store features and percentile. Should contain raw data (hmd_data) for test cases.
			use_training_cases: if true, trainCaseIds is the used as testing as well (append to the end)
			write_db: if true, write result to database (features and percentile for testCaseIds)
        """
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

        
        def radians (array):
                return array*pi/180

        # turn euler angle into vector
        def euler2Vec (vec3):
            rotx = vec3[0]
            roty = vec3[1]
            rotz = vec3[2]
            x = np.sin(radians(roty))*np.cos(radians(rotx))
            y = -np.sin(radians(rotx))
            z = np.cos(radians(roty))*np.cos(radians(rotx))
            return np.array([x,y,z])

        # calculate angle between two vector
        def calcAngleDeg (v1,v2):
            # v1*v2/(sqrt(v1*v1)*sqrt(v2*v2))
            costheta = np.sum(np.multiply(v1,v2),axis=1)/(
                np.sqrt(np.sum(np.multiply(v1,v1),axis=1))*
                np.sqrt(np.sum(np.multiply(v2,v2),axis=1)))
        #     if np.any(abs(costheta)>1):
        #         print("costheta value wrong:\n")
        #         print(costheta[(abs(costheta)>1)])
            return 180/pi*np.arccos([min(max(x,-1),1) for x in costheta])
    
        def smooth(y, box_pts = 7, smooth_type = 'box',sigma = 3):
            if smooth_type == 'box':
                box = np.ones(box_pts)/box_pts
            elif smooth_type == 'gaussian':
                box = signal.gaussian(box_pts,sigma)
                box = box/np.sum(box)
        #         print(box)
            y_smooth = np.convolve(y, box, mode='valid')
            return y_smooth
    
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

        #database connecting function
        def connect(db_name = 'webtest'):
            try:
                db = pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",database = db_name, user='cognitiveleap', password= 'QWE@123456')
                return db.cursor(), db
            except Exception as e:
                raise Exception(3)


        def get_data(CaseIds, raw, cursor):

            data = []

            try:
                for i in CaseIds:
                    if raw:
                        sql = """SELECT PosX, PosY, PosZ, RotX, RotY, RotZ FROM hmd_data WHERE CaseId =  """ + str(i) + """ ORDER BY TimeLog, TimeLogMillisecond"""
                        cursor.execute(sql)
                        # Fetch all the rows in a list of lists.
                        results = np.asarray(cursor.fetchall())
                        if len(results) == 0: #if there's no someone's data in database
                            raise Exception(7)
                        data.append(results)
                    else:
                        sql = "SELECT PathLen, TimeActive, NumRot, TotalDeg, CaseId FROM head_features WHERE CaseId = %s "% str(i)
                        cursor.execute(sql)
                        results = np.asarray(cursor.fetchall())
                        if (len(results) > 0):
                            data.append(results[0])
            except Exception as e:
                raise Exception(4)


            return np.asarray(data)

        def insert(CaseIds, data, cursor, db):
            count = 0
            try:
                tb = 'head_features'
                for i in range(len(data)):
                    sql = "INSERT INTO %s (PathLen,TimeActive, NumRot,TotalDeg, PercentageDistracted, IsMoveRangeLarge, CaseId) VALUES (%s, %s, %s, %s, %s, %s, %s)" % (tb,data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i,4], data[i,5], CaseIds[i])
                    # Execute the SQL command
                    cursor.execute(sql)
                    count += 1
                    # Commit your changes in the database
                db.commit()
                print('%d rows inserted into database %s'%(count,tb))
            except Exception as e:
                raise Exception(5)


        def update(CaseIds, data, cursor, db):
            count = 0
            try:
                tb = 'head_features'
                for i in range(len(data)):
                    sql = "UPDATE %s SET PerPathLen = %s, PerTimeActive = %s, PerNumRot = %s, PerTotalDeg = %s WHERE CaseId = %s" % (tb,data[i, 0], data[i, 1], data[i, 2], data[i, 3], CaseIds[i])
                    # Execute the SQL command
                    cursor.execute(sql)
                    # Commit your changes in the database
                    count += 1
                db.commit()
                print('%d rows updated in the database %s'%(count,tb))
            except Exception as e:
                raise Exception(5)


        num_features = 4
        def process_feature(CaseIds, cursor, db):
            """ Process raw data in db (should be testdb), write features to db (testdb)"""
            data = get_data(CaseIds,True,cursor)
            results = []
            ids  = []
            count = 0
            # search through the table and delete already existed caseids
            try:
                tb = 'head_features'
                if write_db:
                    for caseid in np.unique(CaseIds):
                        if calculated(caseid,cursor):
                            sql = """DELETE FROM %s WHERE CaseId = """%tb + str(caseid)
                            cursor.execute(sql)
                            count += 1
            except Exception as e:
                raise Exception(8)
            print ('%d caseids got deleted from %s'%(count,tb))
            
            try:
                for i in range(len(CaseIds)):
                    result = [0, 0, 0, 0, 0, 0]
                    result[0] = head_length(data[i])
                    result[1] = percentage_over_threshold_head(data[i])
                    result[2] = head_rot(data[i])
                    result[3] = rot_sum_degrees(data[i])
                    result[4] = distracted_rotOnly(data[i])[0]        # percentage time distracted
                    # print(distracted(data[i]))
                    # print(distracted_rotOnly(data[i]))
                    result[5] = isMoveLarge(data[i])        # whether the kid is moving in larger range than normal
                    results.append(result)
                    ids.append(CaseIds[i])
                results = np.asarray(results)
            except Exception as e:
                raise Exception(9)

            if write_db:
                insert(ids, results, cursor, db)
            return pd.DataFrame(results,columns=['PathLen','TimeActive','NumRot','TotalDeg','PercentageDistracted','IsMoveRangeLarge'],index=ids)

        #check if the subject is already processed
        def calculated(CaseId,cursor):
            data = get_data([CaseId], False, cursor)
            return (len(data)) != 0

        # calculate percentile according to TRAINING SET
        # find unique values in train and test list, then loop through to insert test samples into the training scale
        # trainCaseIds, trainData: training cases to build the model
        # testCaseIds, testData: test cases to run the model with
        # only return results for testCaseIds
        def percentage(trainCaseIds, testCaseIds, trainData, testData): #should be percentile lol
            uni_data,uni_count = np.unique(trainData,return_counts=True)
            length = len(trainCaseIds)    # number of training samples
            uni_test = np.unique(testData)
            uni_test = [x for x in uni_test if x not in uni_data]    # unique test samples different from training set
            uni_count = np.cumsum(uni_count)
            uni_count = [0]+list(uni_count[:-1])

            uni_data = uni_data.tolist()
            ind = 0 # index in training
            i = 0 # index in testing
            test_append = []
            test_count_append = []
            while (ind < len(uni_data)) & (i < len(uni_test)):
                x = uni_test[i]
                if x > uni_data[ind]:
                    ind += 1
                else:
                    test_append.append(x)
                    test_count_append.append(uni_count[ind])
                    i += 1
            if ind == len(uni_data):
                test_append += uni_test[i:]
                test_count_append += [len(uni_data)]*(len(uni_test)-i)
                
            uni_data = uni_data+test_append
            uni_count = uni_count+test_count_append


            uni_dic = {uni_data[i]: uni_count[i]/length for i in range(len(uni_data))}
            dic = {testCaseIds[i]: uni_dic[testData[i]] for i in range(len(testCaseIds))}
        
            return dic
            
        def get_training_ids(cursor):
            sql = """SELECT CaseID FROM training_set"""
            cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = np.asarray(cursor.fetchall())[:, 0]
            if len(results) == 0: #if there is no data in DB
                raise Exception (7)
            return results

        def update_percentage(trainCaseIds, testCaseIds, cursor_rnd, cursor_test, db_test):
            """ get features from rnd db and test db respectively, then compute percentile, and write percentile for test cases into testdb"""
            data = get_data(trainCaseIds, False, cursor_rnd)
            data_test = get_data(testCaseIds, False, cursor_test)
            try:
                length = len(testCaseIds)
                perpathlen = percentage(trainCaseIds, testCaseIds, data[:, 0],data_test[:, 0])
                pertimeactive = percentage(trainCaseIds, testCaseIds, data[:, 1],data_test[:, 1])
                pernumrot = percentage(trainCaseIds, testCaseIds, data[:, 2],data_test[:, 2])
                pertotaldeg = percentage(trainCaseIds, testCaseIds, data[:, 3],data_test[:, 3])
                results = np.zeros((length, num_features))
                for i in range(length):
                    results[i, 0] = perpathlen[testCaseIds[i]]
                    results[i, 1] = pertimeactive[testCaseIds[i]]
                    results[i, 2] = pernumrot[testCaseIds[i]]
                    results[i, 3] = pertotaldeg[testCaseIds[i]]
            except Exception as e:
                print(e)
                raise Exception(9)
                
            if write_db:
                update(testCaseIds, results, cursor_test, db_test)
            return pd.DataFrame(results,columns=['PerPathLen', 'PerTimeActive', 'PerNumRot', 'PerTotalDeg'],index=testCaseIds)



        cursor_test, db_test = connect(testdb_name)
        cursor_rnd, db_rnd = connect(traindb_name)
        trainCaseIds = get_training_ids(cursor_rnd)
        if use_training_cases == True:
            testCaseIds += list(trainCaseIds)

        df_feat = process_feature(testCaseIds, cursor_test, db_test) #calculate features for the ones that hasn't been calculated in CaseIds

        # get caseid for all rows
        # try:
            # sql = "SELECT DISTINCTROW(CaseId) FROM head_features"
            # cursor.execute(sql)
            # # Fetch all the rows in a list of lists.
            # results = list(cursor.fetchall())
            # results = [i[0] for i in results]
        # except:
            # raise Exception(4)
            

        
        df_percentile = update_percentage(trainCaseIds,testCaseIds, cursor_rnd, cursor_test, db_test)  #update percentile for everyone
        
        db_test.close()
        db_rnd.close()
        
        df = pd.concat((df_feat,df_percentile),axis=1)
        return df
            
    
    
    if __name__ == '__main__':
        import sys
        try:
            #get the case id
            CaseIds = sys.argv[1:]
            CaseIds = [int(i) for i in CaseIds]
        except Exception as e:
            raise Exception(2)
        main(CaseIds)
        
except Exception as e:
    try:
        print('Error code: '+str(e))    
        if 'db_test' in dir():
            db_test.close()
            print('db_test close in error handling')
        if 'db_rnd' in dir():
            db_rnd.close()
            print('db_rnd close in error handling')
    except:
        print('about to exit. db close error.')            
    try:
        if isinstance(e,str):
            print('uncaught error.')
            sys.exit(10)
        sys.exit(e)
    except Exception as e:      # otherwise sys.exit will be caught
        print('sys.exit error')
        

