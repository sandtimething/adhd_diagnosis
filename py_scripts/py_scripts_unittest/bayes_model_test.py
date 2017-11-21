import unittest
import json

from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.Application import rndAnalysisDatabase
from py_scripts.database import db_manager
from py_scripts.vrclassroom import bayes_model
from py_scripts.vrclassroom.featureStruct import featureFields

from py_scripts.utilities.modelStruct import modelParams, bayesModel

from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING


class bayes_model_test(unittest.TestCase):

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={
                             'echo': True})  # connect database
        self.vrclassroom = vrclassroomFetchDatabase(self.db)
        self.rnd = rndAnalysisDatabase(self.db)

        self.features = featureFields([],[])

        self.features.addTableFields(
            'head_features', [
                "TimeActive",
                "NumRot",
                "PercentageDistracted",
                "TotalDeg",
                "PathLen"
            ])

        self.features.addTableFields('cpt_output_results',
                                     ['OmissionErrors', 'CommissionErrors', 'TargetsRT',
                                      'TargetsRtVariability', 'CommissionErrorsRT', 'CommissionErrorsRtVariability'],
                                     'block=0')
        self.features.addTableFields("signal_detection",[
                "DPrime",
                "Beta",
                "C"
            ],'block=0')

        self.features.addOutputFields("bayes_probabilities",[
            "DPrimeRawProbabilityADHD",
            "DPrimeRawProbabilityHealthy",
            "OmissionRawProbabilityADHD",
            "OmissionRawProbabilityHealthy",
            "CommissionErrorsRtVariabilityRawProbabilityADHD",
            "CommissionErrorsRtVariabilityRawProbabilityHealthy",
            "CRawProbabilityADHD",
            "CRawProbabilityHealthy",
            "TotalDegRawProbabilityADHD",
            "TotalDegRawProbabilityHealthy",
            "CommissionRawProbabilityADHD",
            "CommissionRawProbabilityHealthy",
            "TargetRTVRawProbabilityADHD",
            "TargetRTVRawProbabilityHealthy",
            "TotalRawProbabilityADHD",
            "TotalRawProbabilityHealthy",
            "Priors",
            "FinalProbabilityOfADHD"
        ])

    def tearDown(self):
        pass

    def testfeatureStruct(self):
        # print(json.dumps(self.features.__dict__,indent=4))
        pass


    def testmain(self):
        config_path="D:/CLProject/rnd_analysis/config/modelConfig_0904"
        model = bayesModel.json_decoder(config_path)
        model_params = model.modParams

        print(model_params)
        # get trainCaseIds
        trainCaseIds = self.rnd.get_training_ids()

        # get train CaseIds feature data
        trainFeaturesData = self.rnd.get_features(self.features, trainCaseIds)

        testCaseIds = [21]
        # get test CaseIds feature data
        testFeaturesData = self.rnd.get_features(self.features, testCaseIds)
        
        trainCasesAdhd=self.vrclassroom.getADHDType(trainCaseIds)

        bayes_model.main(self.features,
            trainFeaturesData,trainCaseIds,trainCasesAdhd,
            testFeaturesData,testCaseIds,
            model_params)

        pass


if __name__ == '__main__':
    unittest.main()
