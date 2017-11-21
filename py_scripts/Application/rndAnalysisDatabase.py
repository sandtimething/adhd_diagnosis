import pandas as pd


class rndAnalysisDatabase(object):
    """rndAnalysisDatabse"""

    def __init__(self, db_manager):
        self.db = db_manager

     # Rose
    def get_head_rot(self, CaseIds, fields=None):
        return self.db.find("head_rot", CaseId=CaseIds)

    def del_head_rot(self, CaseIds):
        return self.db.delete("head_rot", CaseId=CaseIds)

    def insert_head_rot(self,datadics):
        return self.db.insertMany("head_rot", datadics)

    def insert_or_update_head_rot(self, datadic):
        """
        CaseId In Datadic is required. Single Record Supported only
        """
        self.db.upsert("head_rot", datadic,["CaseId"])

    # head_features

    def get_head_features(self, CaseIds, fields=None):
        return self.db.find("head_features", CaseId=CaseIds)

    def del_head_features(self, CaseIds):
        return self.db.delete("head_features", CaseId=CaseIds)

    def insert_head_features(self,datadics):
        return self.db.insertMany("head_features", datadics)

    def insert_or_update_head_features(self, CaseIds, datadics):
        self.db.delete("head_features", CaseId=CaseIds)
        self.db.insertMany("head_features", datadics)

    # percentile

    def get_percentile(self, CaseIds, fields=None):
        return self.db.find("percentile", CaseId=CaseIds)

    def del_percentile(self, CaseIds):
        return self.db.delete("percentile", CaseId=CaseIds)

    def insert_percentile(self,datadics):
        return self.db.insertMany("percentile", datadics)

    def insert_or_update_percentile(self, CaseIds, datadics):
        self.db.delete("percentile", CaseId=CaseIds)
        self.db.insertMany("percentile", datadics)

    # signal_detection

    def get_signal(self, CaseIds, fields=None):
        return self.db.find("signal", CaseId=CaseIds)

    def del_signal(self, CaseIds):
        return self.db.delete("signal", CaseId=CaseIds)

    def insert_signal(self,datadics):
        return self.db.insertMany("signal", datadics)

    def insert_or_update_signal(self, CaseIds, datadics):
        self.db.delete("signal", CaseId=CaseIds)
        self.db.insertMany("signal", datadics)


    # bayes_probabilities

    def get_bayes_probabilities(self, CaseIds, fields=None):
        return self.db.find("bayes_probabilities", CaseId=CaseIds)

    def del_bayes_probabilities(self, CaseIds):
        return self.db.delete("bayes_probabilities", CaseId=CaseIds)

    def insert_bayes_probabilities(self,datadics):
        return self.db.insertMany("bayes_probabilities", datadics)

    def insert_or_update_bayes_probabilities(self, CaseIds, datadics):
        self.db.delete("bayes_probabilities", CaseId=CaseIds)
        self.db.insertMany("bayes_probabilities", datadics)



    def get_training_ids(self):
        """
        Fetch All Training CaseId From Training_set
        """
        return [data["CaseId"] for data in self.db.fetchAll("training_set")]


    def get_features(self,featureFieldsObj,CaseIds):
        """
        Get Features according to featureFieldsObj
        return featureFieldsObj.get_all_fields() dict

        Example: 
            featureFieldsObj:{"tables": [{"name": "cpt_output_results", "fields": ["OmissionErrors"], "whereclauses": ["block=0"], "wherekwargs": {}}]}

        """

        for table in featureFieldsObj.inputTables:
            if(table["name"].lower()=='sqlstring'):
                tabledata=self.db.sqlQuery(table["value"])
                tabledata=pd.DataFrame(list(tabledata))
            else:
                tabledata=self.db.find(table["name"],CaseId=CaseIds,*table["whereclauses"],**table["wherekwargs"])
                tabledata=pd.DataFrame(list(tabledata))
                if(len(table["fields"])>0):
                    tabledata=tabledata[table["fields"]+["CaseId"]]

            if('data' in dir()):
                data=pd.merge(data,tabledata,on='CaseId')
            else:
                data=tabledata
        
        #tabledatas:[dict(),dict()]==>join by =CaseId
        return data



    def __updateTable(self,tableName,datadic,keys):
        return self.db.update(tableName,datadic,keys)

