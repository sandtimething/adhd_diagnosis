# coding: utf-8


# https://github.com/pudo/dataset
# https://dataset.readthedocs.io/en/latest/
import dataset
from sqlalchemy.sql import and_, or_, not_


# In[30]:

# load data from database
# from pandas import DataFrame

class db_manager(object):

    # init with a connectionString
    def __init__(self, connectionString, engine_kwargs=None):
        """Initiate db_manager object.
        *engine_kwargs* will be directly passed to
            SQLAlchemy, e.g.  set *engine_kwargs={'pool_recycle': 3600}* will avoid `DB
            connection timeout`_.

        .. _SQLAlchemy Engine URL: http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine
        .. _DB connection timeout: http://docs.sqlalchemy.org/en/latest/core/pooling.html#setting-pool-recycle
         """
        self.connected = False
        self.engine_kwargs = engine_kwargs
        self.db = ''
        self.connect(connectionString)  # connect to db

    @property
    def database(self):
        """Return Databse Object"""
        return self.db

    def tableObj(self, tableName):
        """
        Load a table.

                This will fail if the tables does not already exist in the database. If the table exists, its columns will be reflected and are available on the Table object.
        """
        return self.db.load_table(tableName)

    def close(self):
        """Close the connection manually.

        Not recommended. sqlalchemy will take care of the engine.dispose automatically."""
        if self.connected:
            #                self.__db.close()
            self.db.commit()
            self.connected = False

    def __del__(self):
        """Destructor function. Manual dispose() is unneccessary"""
        self.close()

    def connect(self, connectionString):
        """Connect to the specified connectionString.
        """
        if connectionString == '':
            raise ValueError("connectionString cannot not be empty")

        if self.connected:
            self.close()
        try:
            self.db = dataset.connect(
                connectionString, engine_kwargs=self.engine_kwargs)
        except Exception as e:
            raise e
        self.connected = True
        return self.db

    def insert(self, tableName, datadic):
        """Inert Data Into table
           Example::     Insert('case',dic(Id=1,CPTVersion="1.0"))
                                                        """
        table = self.tableObj(tableName)
        table.insert(datadic)

    def insertMany(self, tableName, datadics):
        table = self.tableObj(tableName)
        table.insert_many(datadics)

    def update(self, tableName, datadic, keys):
        """
        Update table data
                Example::update('patient',dict(name='John Doe', age=47), ['name'])
        """
        table = self.tableObj(tableName)
        table.update(datadic, keys)

    def upsert(self, tableName, datadic, keys):
        """
        An UPSERT is a smart combination of insert and update.

                If rows with matching keys exist they will be updated, otherwise a new row is inserted in the table.
        """
        table = self.tableObj(tableName)
        table.upsert(datadic, keys)

    def delete(self, tableName, *clauses, **filters):
        table = self.tableObj(tableName)
        return table.delete(*clauses, **filters)

    def sqlQuery(self, sqlString):
        """Excute SQL Query"""
        return self.db.query(sqlString)

    def fetchAll(self, tableName):
        table = self.tableObj(tableName)
        return table.all()

    def find(self, tableName, *_clauses, **kwargs):
        table = self.tableObj(tableName)
        return table.find(*_clauses, **kwargs)

    def find_one(self, tableName, *_clauses, **kwargs):
        table = self.tableObj(tableName)
        return table.find_one(*_clauses, **kwargs)


def AndOp(query1, query2):
    """
    And Combine two querys
    """
    return and_(query1, query2)


def OrOp(query1, query2):
    """
    or Combine two querys
    """
    return or_(query1, query2)


def TableField(tableObj, fieldName):
    """
    Return Column Field
    """
    return tableObj.table.columns[fieldName]


def whereInSql(inlist):
    """
    Example:sql="select a.Id, b.ADHDDiagnose  from `case` a, patient b where b.Id=a.SubjectId and a.Id In (%s)"
            sql=sql % whereInSql(CaseIds)
    """
    return ', '.join(list(map(lambda x: str(x), inlist)))
