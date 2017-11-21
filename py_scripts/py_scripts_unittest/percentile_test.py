import unittest
import json

from py_scripts.database import db_manager

from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.Application import rndAnalysisDatabase

from py_scripts.database import db_manager
from py_scripts.vrclassroom import percentile
from py_scripts.vrclassroom.featureStruct import featureFields


from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING



class percentile_test(unittest.TestCase):
    """
    Test For Percentile
    """
    def setUp(self):

        self.db = db_manager(CONNECTION_STRING, engine_kwargs={
                             'echo': True})  # connect database
        self.vrclassroom = vrclassroomFetchDatabase(self.db)
        self.rnd = rndAnalysisDatabase(self.db)
        # get features

        self.features = featureFields([],[])

        self.features.addTableFields(
            'head_features', ['PathLen', 'TimeActive', 'NumRot', 'TotalDeg'])
        self.features.addTableFields('cpt_output_results',
                                     ['OmissionErrors', 'CommissionErrors', 'TargetsRT',
                                      'TargetsRtVariability', 'CommissionErrorsRT', 'CommissionErrorsRtVariability'],
                                     'block=0')


    def tearDown(self):
        pass

    def testfeatureStruct(self):
        print(self.features.get_all_input_fields())
        print(json.dumps(self.features.__dict__))
        pass


    def testmain(self):

        # get trainCaseIds
        trainCaseIds = self.rnd.get_training_ids()

        # get train CaseIds feature data
        trainFeaturesData = self.rnd.get_features(self.features, trainCaseIds)

        testCaseIds = [21]
        # get test CaseIds feature data
        testFeaturesData = self.rnd.get_features(self.features, testCaseIds)

        calcdatas=percentile.main(self.features.get_all_input_fields(),trainFeaturesData,trainCaseIds,testFeaturesData,testCaseIds)
        print(calcdatas)






if __name__ == '__main__':
    unittest.main()
