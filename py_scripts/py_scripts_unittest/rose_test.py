import unittest
from py_scripts.Application import vrclassroomFetchDatabase
from py_scripts.database import db_manager
from py_scripts.vrclassroom import rose
from py_scripts.py_scripts_unittest.connectionstring import CONNECTION_STRING


class rose_test(unittest.TestCase):

    def setUp(self):
        self.db = db_manager(CONNECTION_STRING, engine_kwargs={'echo': True})
        self.vrclassroom = vrclassroomFetchDatabase(self.db)


    def tearDown(self):
        pass

    def testmain(self):
        hmd_data=self.vrclassroom.getHMD([21])
        datas=rose.main(hmd_data,21)
        count = 0
        for data in datas:
            count += 1
            print(data)
        self.assertEqual(count,360,'360 rose data fail')


if __name__ == '__main__':
    unittest.main()
