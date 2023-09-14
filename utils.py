import findspark
findspark.init()
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row, SQLContext
from time import sleep
def _initialize_spark() -> SparkSession:
    try:
        # Kiểm tra xem liệu SparkSession đã tồn tại hay chưa
        if 'spark' not in globals():
            spark = SparkSession.builder.getOrCreate()
            # SparkContext.setSystemProperty('spark.executor.memory', '12g')
            # SparkContext.setSystemProperty('spark.driver.memory', '12g')
            # Cau hinh de doc tap tin tu HDFS
            # SparkContext.setSystemProperty('spark.hadoop.dfs.client.use.datanode.hostname', 'true')
            conf = SparkConf().setAppName("2_Recommendation_System").setMaster("local")
            conf.set("spark.network.timeout", "365d")
            conf.set("spark.executor.heartbeatInterval", "100s")
            spark = SparkSession.builder.config(conf=conf).getOrCreate()
            sc = spark.sparkContext
            # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
            if 'sc' not in globals():
                sc = SparkContext.getOrCreate()
            else:
                sc = globals()['sc']
            app_id = sc.applicationId
            print(f"Spark is running. Application ID: {app_id}")
        else:
            spark = globals()['spark']
            sc = spark.sparkContext
            # Kiểm tra xem liệu SparkContext đã tồn tại hay chưa
            if 'sc' not in globals():
                sc = SparkContext.getOrCreate()
            else:
                sc = globals()['sc']
            # Get the application ID
            app_id = sc.applicationId
            print(f"Spark is running. Application ID: {app_id}")
        return spark, sc
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Retrying in 5 seconds...")
        sleep(5)
    