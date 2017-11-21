from py_scripts.database import whereInSql

class vrclassroomFetchDatabase(object):

    """docstring for vrclassroomFetchDatabase"""

    def __init__(self, db_manager):
        self.db = db_manager

    def getCPT(self, CaseIds, blocks=None, fields=None):
        """
        Get CPT
        """
        if(blocks==None):
            return self.db.find("cpt_output_results",CaseId=CaseIds)
        else:
            return self.db.find("cpt_output_results",CaseId=CaseIds,Block=blocks)

    def getADHDType(self, CaseIds):
        """
        Get ADHDType of Cases
        [Id ,ADHDDiagnose]
        """
        sql = "select a.Id, b.ADHDDiagnose  from `case` a, patient b where b.Id=a.SubjectId and a.Id In (%s)"
        sql = sql % whereInSql(CaseIds)
        return self.db.sqlQuery(sql)

    def getSNAP(self, CaseIds, fields=None):
        pass

    def getHMD(self, CaseIds, fields=None):
    	"""
    	Get Hmd_data. Slow Function
    	"""    
    	return self.db.find("hmd_data",CaseId=CaseIds)

