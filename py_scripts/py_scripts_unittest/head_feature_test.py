import unittest
from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.database import db_manager
from py_scripts.vrclassroom import head_feature
from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING


class rose_test(unittest.TestCase):

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={'echo': True})
        self.vrclassroom = vrclassroomFetchDatabase(self.db)


    def tearDown(self):
        pass

    def testmain(self):
        hmd_data=self.vrclassroom.getHMD([21,22])
        datas=head_feature.main(hmd_data,[21,22])
        print(datas)


if __name__ == '__main__':
    unittest.main()
