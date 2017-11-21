import unittest
import json
from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.database import db_manager
from py_scripts.unittest.connectionstring import CONNECTION_STRING


class vrclasroomFetchDatabse_test(unittest.TestCase):

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={'echo': True})
        self.vrclassroom = vrclassroomFetchDatabase(self.db)

    def tearDown(self):
        pass

    def testgetCPT(self):
        datas = self.vrclassroom.getCPT([1, 2, 3])
        count = 0
        for data in datas:
            count += 1
        print(count)

    def testgetADHDType(self):
        datas = self.vrclassroom.getADHDType([1, 2, 3])
        count = 0
        for data in datas:
            count += 1
        print(count)

    def testgetHMD(self):
        datas = self.vrclassroom.getHMD([1])
        count = 0
        for data in datas:
            count += 1
        print(count)


if __name__ == '__main__':
    unittest.main()
