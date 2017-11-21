import unittest
from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.database import db_manager
from py_scripts.vrclassroom import signal_detection
from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING


class signal_dection_test(unittest.TestCase):


    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={'echo': True})
        self.vrclassroom = vrclassroomFetchDatabase(self.db)


    def tearDown(self):
        pass

    def testmain(self):
        cpt_data=self.vrclassroom.getCPT([21])
        datas=signal_detection.main(cpt_data,21)
        count = 0
        for data in datas:
            count += 1
            print(data)


if __name__ == '__main__':
    unittest.main()
