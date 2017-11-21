import unittest
import json

from py_scripts.Application import rndAnalysisDatabase
from py_scripts.database import db_manager
from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING
from py_scripts.vrclassroom.featureStruct import featureFields


class rndAnalysisDatabase_test(unittest.TestCase):
    """
    Test Class For rndAnalysisDatabase
    """

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={'echo': True})
        self.app = rndAnalysisDatabase(self.db)

    def tearDown(self):
        pass

     # Rose
    def testget_head_rot(self):
        datas = self.app.get_head_rot([1])
        count = 0
        for data in datas:
            count += 1
        print(count)

    def testdel_head_rot(self):
        self.app.del_head_rot([1, 2])
        pass

    def testinsert_head_rot(self):
        #self.app.insert_head_rot([dict(BlockNum=0,BinNum=0,Value=8,CaseId=1),dict(BlockNum=0,BinNum=0,Value=8,CaseId=2)]*2)
        pass

    def testinsert_or_update_head_rot(self):
        self.app.insert_or_update_head_rot(dict(BlockNum=0,BinNum=0,Value=7,CaseId=1))


    # head_features

    def testget_head_features(self):
        pass

    def testdel_head_features(self):
        pass

    def testinsert_head_features(self):
        pass

    def testinsert_or_update_head_features(self):
        pass

    # percentile

    def testget_percentile(self):
        pass

    def testdel_percentile(self):
        pass

    def testinsert_percentile(self):
        pass

    def testinsert_or_update_percentile(self):
        pass

    # signal_detection

    def testget_signal(self):
        pass

    def testdel_signal(self):
        pass

    def testinsert_signal(self):
        pass

    def testinsert_or_update_signal(self):
        pass

    # bayes_probabilities
    def testget_bayes_probabilities(self):
        pass

    def testdel_bayes_probabilities(self):
        pass

    def testinsert_bayes_probabilities(self):
        pass

    def testinsert_or_update_bayes_probabilities(self):
        pass


    def testget_training_ids(self):
        caseIds=self.app.get_training_ids()
        print(caseIds)

    def testget_features(self):
        CaseIds=[1,2,3]

        features=self.app.get_features(self.featureStruct(),CaseIds)



    def featureStruct(self):
        featureStruct=featureFields()
        featureStruct.addTableFields('head_features',['PathLen', 'TimeActive', 'NumRot', 'TotalDeg'])
        featureStruct.addTableFields('cpt_output_results',
            ['OmissionErrors', 'CommissionErrors', 'TargetsRT', 
            'TargetsRtVariability', 'CommissionErrorsRT', 'CommissionErrorsRtVariability'],
            'block=0')
        return featureStruct

if __name__ == '__main__':
    unittest.main()
