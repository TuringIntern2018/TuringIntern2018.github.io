# Logistic regression with Apache Spark

In this section we give a tutorial on how to run logistic regression in Apache Spark on the Airline data on the CrayUrika-GX. Here we interface with Spark through PySpark, the Python API, though Spark also offers APIs through Scala, Java and R. It's also recommended to use Jupyter notebook to run your Python code so the code can easily be run in stages which makes for easier error diagnosis. 

## Spark settings

```python
import numpy as np
import findspark
findspark.init()
import scipy as scipy
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession




spark = SparkSession.builder \
   .master("mesos://zk://zoo1:2181,zoo2:2181,zoo3:2181/mesos") \
   .appName("Airplane data") \
   .config("spark.cores.max", 360) \
   .config("spark.executor.cores", 36) \
   .config("spark.executor.memory", "96g") \
   .getOrCreate() \


sc = spark.sparkContext

print(sc._conf.getAll()) #print settings to double check

import time
time.sleep(10)
print('wait over')
```

Spark requires to be initialised through the SparkSession class, which defines important configurations which controls how the nodes are to be distributed. The 'master' method above takes the url of the Mesos master node (for the Cray-Urika GX, may be different for other systems), which is a cluster resource manager that allows data to be distributed across the cluster. Alternatively 'master' can be set to 'local[x]' to distribute data locally, where 'x' is the number of CPU cores you want to use (36 is the limit per node on the Cray-Urika GX).

spark.cores.max denotes the number of maximum CPU cores Spark is allowed to use for this session. On Cray-Urika GX this is capped to 324 (36 cores per node, 9 nodes). spark.executor.cores is the number of cores each executor (node) can use, up to 36 on Cray-Urika GX. spark.executor.memory controls how much memory can be allocated to each executor.

For fastest performance use all 324 cores, but if total memory exceeds around 1800gb Spark will reduce the number of cores as there isn’t enough memory. So memory per executor should be kept below 200gb.

In general (min(spark.cores.max, 324)/spark.executor.cores)*spark.executor.memory<=1800

If running on Mesos Spark takes a few seconds to initialise so it's a good idea to ask Python to wait 10 seconds before running any computation.
