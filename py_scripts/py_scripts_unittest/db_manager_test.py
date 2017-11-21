import unittest
from py_scripts.database import db_manager
from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING

class db_manager_test(unittest.TestCase):

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING,engine_kwargs={'echo': True})
        pass

    def tearDown(self):
        pass

    def testconnect(self):
        self.db.connect(CONNECTION_STRING)
        #self.assertEqual(myclass.sum(1, 2), 2, 'test sum fail')


    def testTableObj(self):
        table=self.db.tableObj("case")
        table.find(table.table.columns["Id"]<2)

    def testfetchAll(self):
        casetable=self.db.fetchAll('case')
        for case in casetable:
            print(case)

    def testfind(self):
        datas=self.db.find("case",Id=1)
        datas=list(datas)
        for data in datas:
            print(data)

if __name__ == '__main__':
    unittest.main()
