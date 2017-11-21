
# coding: utf-8

# #  Creation of Bayes Distribution
# This is meant to be run after the SignalDetection python script that creates the signal detections. This script creates a series of vectors to be stored in their own table which can be automatically plotted when qurneried. Uncomment code at bottom for plotting results.

#When running this script no caseIDs need be passed in, as only the internal training set is used for calculations

# Usage:
#   python bayes_dist_vrclassroom.py


# ### 1. Necessary Imports:

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

      #from __future__ import division # Not neccessary in Python 3 and later
      import scipy
      from math import exp,sqrt
  except Exception as e:
      raise Exception(1)


  # In[2]:


  """
  try:
      CaseIDs = sys.argv[ 1: ]
      CaseIDs = [int(i) for i in CaseIDs]
  except Exception as e:
      raise Exception(2)
  """



  # ### 2) Functions to Estimate Probability Distributions

  # In[3]:

  #helper function for calculating pdfs and x coordinates:
  def generate_normal_probabilities(data, adhd):
      #calculating moments:
      adhdMu = scipy.mean(data[adhd == 1])
      adhdSD = scipy.std(data[adhd == 1])
      healthyMu = scipy.mean(data[adhd == 0])
      healthySD = scipy.std(data[adhd == 0])



      #calculating x  values for the pdfs
      minimum = np.amin((adhdMu-3*adhdSD, healthyMu - 3*healthySD))
      if minimum < 0:
          minimum = 0
      x = np.linspace(minimum,
                      np.max((adhdMu + 3*adhdSD, healthyMu + 3*healthySD)),
                      200)

      adhdProbs = np.zeros(np.shape(x)[0])
      healthyProbs = np.zeros(np.shape(x)[0])

      #calculating pdfs for each x value
      for i in np.arange(np.shape(x)[0]):
          adhdProbs[i] = scipy.stats.norm.pdf(x[i], loc = adhdMu, scale = adhdSD)
          healthyProbs[i] = scipy.stats.norm.pdf(x[i], loc = healthyMu, scale = healthySD)

      return x, adhdProbs, healthyProbs



  # In[4]:

  def get_probability_distributions(data):
      rows = np.shape(data)[0]
      cols = np.shape(data)[1]
      adhd = data[:, 0]

      data = data[:, 1:cols]


      #I should modularize this, but I think this is marginally more readable
      xOE, adhdOEProbs, healthyOEProbs = generate_normal_probabilities(data[:, 0], adhd)
      xCE, adhdCEProbs, healthyCEProbs = generate_normal_probabilities(data[:, 1], adhd)
      xRTV, adhdRTCProbs, healthyRTVProbs = generate_normal_probabilities(data[:, 2], adhd)
      xDP, adhdDPProbs, healthyDPProbs = generate_normal_probabilities(data[:, 3], adhd)
      xB, adhdBProbs, healthyBProbs = generate_normal_probabilities(data[:, 4], adhd)

      return  np.transpose(np.array((xOE, adhdOEProbs, healthyOEProbs,
                          xCE, adhdCEProbs, healthyCEProbs,
                          xRTV, adhdRTCProbs, healthyRTVProbs,
                          xDP, adhdDPProbs, healthyDPProbs,
                          xB, adhdBProbs, healthyBProbs)))



  # In[5]:

  #initializes connection info for database
  def connect():
      try:
          return pymysql.connect(host = "rm-j6cluj4576jdi6n6oo.mysql.rds.aliyuncs.com",
                                 database = 'rnd_test',
                                 user='cognitiveleap',
                                 password= 'QWE@123456')
      except Exception as e:
          raise Exception(3)


  # In[6]:

  #Gets pertinent data from database

  def get_data(CaseIds):

      try:

          dataPatient = []
          dataSignal = []
          dataADHD = []
          adhd = np.zeros((len(CaseIds), 1))
          # Prepare SQL query to INSERT a record into the database.
          for i in CaseIds:
              sql = """SELECT OmissionErrors, CommissionErrors, TargetsRtVariability
                       FROM cpt_output_results WHERE Block = 0 AND CasdId = """ + str(i)
              cursor.execute(sql)
              # Fetch all the rows in a list of lists.
              results = np.asarray(cursor.fetchall())
              if len(results) == 0: #if there's no someone's data in database

                  raise Exception(7)
              dataPatient.append(results)

              sql = """SELECT DPrime, Beta FROM signal_detection WHERE Block = 0 AND CaseId = """ + str(i)
              cursor.execute(sql)
              # Fetch all the rows in a list of lists.
              results = np.asarray(cursor.fetchall())
              if len(results) == 0: #if there's no someone's data in database
                  raise Exception(7)
              dataSignal.append(results)


              sql = """SELECT ADHDDiagnose FROM patient WHERE Id = """ + str(i)
              cursor.execute(sql)
              # Fetch all the rows in a list of lists.
              results = np.asarray(cursor.fetchall())
              if len(results) == 0: #if there's not someone's data in database
                  raise Exception(7)
              if results > 0: # ignoring severity for now
                  results = 1
              dataADHD.append(results)


      except Exception as e:
          raise Exception(4)



      dataPatient = np.concatenate(np.asarray(dataPatient))
      dataSignal = np.concatenate(np.asarray(dataSignal))
      adhd[:, 0] = np.asarray(dataADHD)
      return np.concatenate((adhd, dataPatient, dataSignal), axis = 1)


  # In[7]:

  #Inserts created vectors into plotting table

  def insert_bayes(data):

      try:
          cursor = db.cursor()
          #Clearing DB:
          sql = "truncate table bayes_bound_prob_plot"
          # Execute the SQL command
          cursor.execute(sql)


          for i in range(len(data)):
              #weird syntax for readability
              sql = "INSERT INTO bayes_bound_prob_plot"
              sql += "(Id,"
              sql += " XCoordinateOmissionErrors,"
              sql += " HealthyOmissionErrorsProbability,"
              sql += " ADHDOmissionErrorsProbability,"
              sql += " XCoordinateComissionErrors,"
              sql += " HealthyCommissionErrorsProbability,"
              sql += " ADHDCommissionErrorsProbability,"
              sql += " XCoordinateTargetsRTVariability,"
              sql += " HealthyTargetsRTVariabilityProbability,"
              sql += " ADHDTargetsRTvariabilityProbability,"
              sql += " XCoordinateDPrime,"
              sql += " HealthyDPrimeProbability,"
              sql += " ADHDDPrimeProbability,"
              sql += " XCoordinateBeta,"
              sql += " HealthyBetaProbability,"
              sql += " ADHDBetaProbability)"
              sql += " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" % (np.int(i) + 1,
                                                                                                   data[i, 0],
                                                                                                   data[i, 1],
                                                                                                   data[i, 2],
                                                                                                   data[i, 3],
                                                                                                   data[i, 4],
                                                                                                   data[i, 5],
                                                                                                   data[i, 6],
                                                                                                   data[i, 7],
                                                                                                   data[i, 8],
                                                                                                   data[i, 9],
                                                                                                   data[i, 10],
                                                                                                   data[i, 11],
                                                                                                   data[i, 12],
                                                                                                   data[i, 13],
                                                                                                   data[i, 14])

              # Execute the SQL command
              cursor.execute(sql)
      except Exception as e:
          raise Exception(5)

  def get_training_ids():
      sql = """SELECT CaseID FROM training_set"""
      cursor.execute(sql)
      # Fetch all the rows in a list of lists.
      results = np.asarray(cursor.fetchall())[:, 0]
      if len(results) == 0: #if there is no data in DB
          raise Exception (7)
      return results


  # ### 4. Main Function:

  # In[8]:

  #Main function does errythang

  def main_bayes():
      caseIds = get_training_ids()
      allPatients = get_data(caseIds)
      bayesData =  get_probability_distributions(allPatients)

      """#plotting code. Uncomment if you want to error check plots.
      plt.plot(bayesData[:, 0], bayesData[:, 1],'b') # plotting t,b separately
      plt.plot(bayesData[:, 0], bayesData[:, 2],'g') # plotting t,c separately
      plt.title("Omission Errors")
      plt.show()

      plt.plot(bayesData[:, 3], bayesData[:, 4],'b') # plotting t,b separately
      plt.plot(bayesData[:, 3], bayesData[:, 5],'g') # plotting t,c separately
      plt.title("Commission Errors")
      plt.show()

      plt.plot(bayesData[:, 6], bayesData[:, 7],'b') # plotting t,b separately
      plt.plot(bayesData[:, 6], bayesData[:, 8],'g') # plotting t,c separately
      plt.title("RTV")
      plt.show()

      plt.plot(bayesData[:, 9], bayesData[:, 10],'b') # plotting t,b separately
      plt.plot(bayesData[:, 9], bayesData[:, 11],'g') # plotting t,c separately
      plt.title("DPrime")
      plt.show()

      plt.plot(bayesData[:, 12], bayesData[:, 13],'b') # plotting t,b separately
      plt.plot(bayesData[:, 12], bayesData[:, 14],'g') # plotting t,c separately
      plt.title("Beta")
      plt.show()"""


      insert_bayes(bayesData)

      return bayesData



  # In[9]:

  #caseIds = np.append(np.arange(1, 53, 1), np.arange(61, 69, 1))
  #connect to db
  db = connect()
  # prepare a cursor object using cursor() method
  cursor = db.cursor()

  bayesData = main_bayes()
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
