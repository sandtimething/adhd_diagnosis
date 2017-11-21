# # Creation of Bayes Model (3/4):
# This script needs to be run after bayes_dist_vrclassroom. We create and store cutoffs for our plots.
# No imputs or outputs are required, as we only use data already entered in the DB

#When running this script no caseIDs need be passed in, as only the internal training set is used for calculations

# Usage:
#   python bayes_cutoffs_vrclassroom.py


# ### Necessary imports:

# In[1]:
try:

  try:
      import numpy as np
      import math
      from scipy import stats
      import pandas as pd
      import pymysql
      import matplotlib as mpl
      import matplotlib.pyplot as plt
      import sys
      import json
      import collections

      #from __future__ import division # Not neccessary in Python 3 and later
      import scipy
      from math import exp,sqrt
  except Exception as e:
      raise Exception(1)


  def connect():
      try:
          return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
                                 database = 'rnd_test',
                                 user='cognitiveleap',
                                 password= 'QWE@123456')
      except Exception as e:
          raise Exception(3)


  #intermediary function for finding the bayes boundary given two diferent distributions
  def get_distribution_cutoff(data):

      omissionErrorsCutoffs = find_boundary(data[:, 1], data[:, 2], data[:, 3])

      commissionErrorsCutoffs = find_boundary(data[:, 4], data[:, 5], data[:, 6])

      targetsRTVCutoffs = find_boundary(data[:, 7], data[:, 8], data[:, 9])

      dPrimeCutoffs = find_boundary(data[:, 10], data[:, 11], data[:, 12])

      betaCutoffs = find_boundary(data[:, 13], data[:, 14], data[:, 15])

      theWholePointOfThisFunction = np.concatenate((np.asarray(omissionErrorsCutoffs),
                                                np.asarray(commissionErrorsCutoffs),
                                                np.asarray(targetsRTVCutoffs),
                                                np.asarray(dPrimeCutoffs),
                                                np.asarray(betaCutoffs)))

      theWholePointOfThisFunction = np.ndarray.flatten(theWholePointOfThisFunction)

      return theWholePointOfThisFunction




  # In[5]:

  # helper function for finding the boundary
  def find_boundary(x, first, second):
      truths = first < second
      cutoffs = []
      for i in np.arange(len(truths) - 1):
          if truths[i] != truths[i + 1]:
              cutoffs.append(x[np.int(i) + 1])
      if len(cutoffs) < 2:
          cutoffs.append(None)
      return cutoffs



  #getting data from plotting vector tables
  def get_probability_data():
      try:
          data = []


          sql = """SELECT * FROM bayes_bound_prob_plot"""
          cursor.execute(sql)
          # Fetch all the rows in a list of lists.
          results = np.asarray(cursor.fetchall())
          if len(results) == 0: #if there's no someone's data in database
              raise Exception(7)
          data = np.asarray(results)



      except Exception as e:
          raise Exception(4)

      return data



  #Inserting created cutoofs for plots
  def insert_cutoffs(data):
      data[np.equal(data,None)] = -666

      try:
          cursor = db.cursor()

          #Clearing DB:
          sql = "truncate table bayes_cutoffs"
          # Execute the SQL command
          cursor.execute(sql)

          sql = "INSERT INTO bayes_cutoffs (Id,"
          sql += "OmissionErrorsCutoffOne,"
          sql += "OmissionErrorsCutoffTwo,"
          sql += "CommissionErrorsCutoffOne,"
          sql += "CommissionErrorsCutoffTwo,"
          sql += "TargetRTVCutoffOne, "
          sql += "TargetRTVCutoffTwo, "
          sql += "DPrimeCutoffOne, "
          sql += "DPrimeCutoffTwo, "
          sql += "BetaCutoffOne, "
          sql += "BetaCutoffTwo )"
          sql += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (0,
                                                                           data[0],
                                                                           data[1],
                                                                           data[2],
                                                                           data[3],
                                                                           data[4],
                                                                           data[5],
                                                                           data[6],
                                                                           data[7],
                                                                           data[8],
                                                                           data[9])
              # Execute the SQL command
          cursor.execute(sql)

      except Exception as e:
          raise Exception(5)


  def main_bayes_boundary():

      probData = get_probability_data()

      cutoffs = get_distribution_cutoff(probData)

      insert_cutoffs(cutoffs)

      return cutoffs


  db = connect()
  # prepare a cursor object using cursor() method
  cursor = db.cursor()

  data = main_bayes_boundary()
  # Commit your changes in the database
  db.commit()
  db.close()

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
  except Exception as e:    # otherwise sys.exit will be caught
    print('sys.exit error')

