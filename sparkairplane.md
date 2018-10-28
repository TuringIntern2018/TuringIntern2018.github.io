# Logistic regression with Apache Spark

In this section we give a tutorial on how to run logistic regression in Apache Spark on the Airline data on the CrayUrika-GX. Here we interface with Spark through PySpark, the Python API, though Spark also offers APIs through Scala, Java and R. It's also recommended to use Jupyter notebook to run your Python code so the code can easily be run in stages which makes for easier error diagnosis. 

## Spark settings

Spark requires to be initialised through the SparkSession class, which defines important configurations which controls how the nodes are to be distributed. The 'master' method below takes the url of the Mesos master node (for the Cray-Urika GX, may be different for other systems), which is a cluster resource manager that allows data to be distributed across the cluster. Alternatively 'master' can be set to 'local[x]' to distribute data locally, where 'x' is the number of CPU cores you want to use (36 is the limit per node on the Cray-Urika GX).

spark.cores.max denotes the number of maximum CPU cores Spark is allowed to use for this session. On Cray-Urika GX this is capped to 324 (36 cores per node, 9 nodes). spark.executor.cores is the number of cores each executor (node) can use, up to 36 on Cray-Urika GX. spark.executor.memory controls how much memory can be allocated to each executor.

For fastest performance use all 324 cores, but if total memory exceeds around 1800gb Spark will reduce the number of cores as there isn’t enough memory. So memory per executor should be kept below 200gb.

In general (min(spark.cores.max, 324)/spark.executor.cores)*spark.executor.memory<=1800

If running on Mesos Spark takes a few seconds to initialise so it's a good idea to ask Python to wait 10 seconds before running any computation.

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

## Data Processing

Here we read in the Airlines dataset, which is split by year between 1987 and 2008. First we read in the data from 1987, then iteratively append the data from each new year to our 1987 data until we reach 2008. A new column called 'Delay' is defined which is a binary variable that returns 1 if the flight was delayed and 0 otherwise.

```python
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

data1987=spark.read.csv(
    "file:///home/users/jwang/Airline/1987.csv", header=True, inferSchema=True)

for year in range(1988,2009):
    datayear=spark.read.csv(
    "file:///home/users/jwang/Airline/{}.csv".format(year), header=True, inferSchema=True)
    data1987=data1987.union(datayear)


print('start')
data1987=data1987.withColumn("Delay", when(data1987["ArrDelay"] > 0, 1).otherwise(0))
```

## Feature Engineering

Next we do some feature engineering. Here we treat 'Distance', and 'Year' as numerical variables, and we introduce higher polynomial terms of both into the data after standardising both columns, up to 3rd and 6th power for both respectively. It might be strange to treat 'Year' as a numerical variable rather than categorical but sometimes it's desirable to predict delays in future years based on data in previous years which would not be possible to do if 'Year' was a categorical variable. Furthermore, we add sine and cosine of the CRSDepTime and CRSArrTime (expected depart and arrival times) columns with period 2400 to reflect the periodic nature of time. In a sense this is a first order Fourier series regression of both columns, somewhat similar to how linear regression is a 'first order polynomial regression'

```python
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler

degree=4
degreeyear=7

from pyspark.sql.functions import mean as _mean, stddev as _stddev, col

data1987 = data1987.withColumn("Distance", data1987["Distance"].cast(DoubleType()))
data1987 = data1987.withColumn("CRSElapsedTime", data1987["CRSElapsedTime"].cast(DoubleType()))

df_stats = data1987.select(
    _mean(col('Distance')).alias('distmean'),
    _stddev(col('Distance')).alias('diststd')
).collect()

distmean = df_stats[0]['distmean']
diststd = df_stats[0]['diststd']

data1987 = data1987.withColumn("Distance", (col("Distance") - distmean) / diststd)


df_stats2 = data1987.select(
    _mean(col('Year')).alias('yearmean'),
    _stddev(col('Year')).alias('yearstd')
).collect()

yearmean = df_stats2[0]['yearmean']
yearstd = df_stats2[0]['yearstd']

data1987 = data1987.withColumn("Yearst", (col("Year") - yearmean) / yearstd)


for i in range(2,degree):
    data1987 = data1987.withColumn("Distance"+str(i), data1987["Distance"]**i)
    
for i in range(2,degreeyear):
    data1987 = data1987.withColumn("Yearst"+str(i), data1987["Yearst"]**i)
    
pi=math.pi

data1987 = data1987.withColumn("CRSDepTimeCosine", cos(data1987["CRSDepTime"]*2*pi/2400))

data1987 = data1987.withColumn("CRSDepTimeSine", sin(data1987["CRSDepTime"]*2*pi/2400))

data1987 = data1987.withColumn("CRSArrTimeCosine", cos(data1987["CRSArrTime"]*2*pi/2400))

data1987 = data1987.withColumn("CRSArrTimeSine", sin(data1987["CRSArrTime"]*2*pi/2400))
    
print('done')
```

We want to get rid of all rows which denote cancelled flights as cancelled flights have 'NA' for the flight delay columns and are generally a bit different from actual flight delays which we're trying to predict. Categorical variables in Spark can be a bit fiddly (compared to say R or Python), but it's basically the same as principle as one hot encoding in the standard Python sklearn library which turns categorical variables with say K categories into K-1 columns where the ith column returns 1 for the first category and 0 otherwise. .

```python

data1987=data1987.filter(data1987["Cancelled"]==0)


logisticdata=data1987.select("Delay","Year","Month","DayofMonth","DayofWeek","CRSDepTime","CRSArrTime","UniqueCarrier",
"CRSElapsedTime","Origin","Dest","CRSDepTimeCosine","CRSDepTimeSine","CRSArrTimeCosine","CRSArrTimeSine","Distance",
'Distance2', 'Distance3', 'Yearst', 'Yearst2', 'Yearst3', 'Yearst4', 'Yearst5','Yearst6')

categorical_variables=["Month","DayofMonth","DayofWeek","UniqueCarrier","Origin","Dest"]

for variable in categorical_variables:
    #converts string variables to numerical indices e.g. January to 1, February to 2 etc.
    indexer = StringIndexer(inputCol=variable, outputCol=variable+"index")
    logisticdata = indexer.fit(logisticdata).transform(logisticdata)
 
    #explodes the now numerical categorical variables into binary variables 
    encoder = OneHotEncoder(inputCol=variable+"index", outputCol=variable+"vec")
    logisticdata = encoder.transform(logisticdata)   

print('dropping')
logisticdata=logisticdata.na.drop()
print('finished dropping')    
    
print(logisticdata.take(2))

```

Furthermore, data has to be structured into a specific form before it can be passed to the LogisticRegression class in Spark. Specifically every feature column needs to be condensed into a 'DenseVector' column which we dub 'features' via the 'VectorAssembler' transformer. Here we also split the data into a training and test data sets in the ratio of 85:15.

```python

from pyspark.ml.feature import VectorAssembler

logisticdata2=logisticdata.select("Delay","Monthvec","DayofMonthvec","DayofWeekvec","CRSDepTime","CRSArrTime","UniqueCarriervec","CRSElapsedTime","Originvec"
                            ,"Destvec", "Year","CRSDepTimeCosine","CRSDepTimeSine","CRSArrTimeCosine","CRSArrTimeSine","Distance", 'Distance2', 'Distance3', 'Yearst', 'Yearst2', 'Yearst3', 'Yearst4', 'Yearst5','Yearst6')


assembler = VectorAssembler(
    inputCols=["CRSDepTime","CRSArrTime","CRSElapsedTime","Monthvec", "DayofMonthvec", "DayofWeekvec", "UniqueCarriervec", "Originvec", "Destvec", "CRSDepTimeCosine","CRSDepTimeSine","CRSArrTimeCosine","CRSArrTimeSine","Distance",'Distance2', 'Distance3','Yearst', 'Yearst2', 'Yearst3', 'Yearst4', 'Yearst5','Yearst6'],
    outputCol="features")

logisticdata2 = assembler.transform(logisticdata2)


logisticdata2=logisticdata2.select("Year","Delay","features")

logistictestdata, logistictrainingdata = logisticdata2.randomSplit(weights=[0.15, 0.85],  seed=12345)

```

## Model Fitting

The actual fitting of the logistic regression model is relatively straightforward. Key parameters to keep in mind are the regularisation parameters. 'elasticNetParam' controls the type of parameterisation, which when at 0 and 1 denotes ridge regression and LASSO regression respectively (latter of which we chose here) and everything between a linear combination of both. 'regParam' controls the degree of parametrisation. Here we also time how long the regression takes to train. 

```python

lrgen = LogisticRegression(labelCol="Delay", featuresCol="features", maxIter=100, regParam=0.001, elasticNetParam=1, standardization=True)
# Fit the data to the model
print('lrgen done')

import time
import datetime

start = time.time()

linearModelgen = lrgen.fit(logistictrainingdata)

end = time.time()

timetaken=end-start
print(timetaken)

```

## Model Predictions

After training the model, we would like to make some predictions and evaluate the model on the test set. This is relatively straightforward to do and we can write the resulting precision and recall of the model predictions to a csv file, which can be easily used to plot a precision-recall graph in Python or R for instance. 

```python

predictions = linearModelgen.evaluate(logistictestdata)

testSummary = predictions

print(type(testSummary))


print('start')
start1 = time.time()
prtest = testSummary.pr.toPandas()
end1 = time.time()
print('done')

timetaken1=end1-start1
print(timetaken1)

prtest.to_csv('airplanetestsummary.csv')

print(prtest['recall'])

print(prtest['precision'])

print(type(testSummary))

```

## Model Coefficients

The model coefficients can be retrieved but it's just given as a list of numbers. To map the coefficients back to the original features they correspond to is a little fiddly but basically the coefficients are still given in the correct order, just without the column names. So you can match the coefficients with the names by retrieving the original feature column names in the correct order from the dataset and simply stacking the column of coefficients next to it to form an numpy array with (coefficient, name) pairs. The coefficients (with names) are again printed to a csv file.

```python

modelcoefficients=np.array(linearModelgen.coefficients)

names=[x["name"] for x in sorted(logistictrainingdata.schema["features"].metadata["ml_attr"]["attrs"]["binary"]+
   logistictrainingdata.schema["features"].metadata["ml_attr"]["attrs"]["numeric"], 
   key=lambda x: x["idx"])]


matchcoefs=np.column_stack((modelcoefficients,np.array(names)))

import pandas as pd

matchcoefsdf=pd.DataFrame(matchcoefs)

matchcoefsdf.columns=['Coefvalue', 'Feature']

print(matchcoefsdf)

matchcoefsdf.to_csv('Airplanecoefspd1.csv')


```

To summarise the notable findings on coefficient analysis:

Months most likely to contain delays are: December, January, June, July. September and May least likely

Friday, Thursday are more likely to contain delays. Saturday and Sunday least likely.

Flights originating from Atlanta International Airport, Detroit Metro Airport, Chicago O'Hare International Airport are most likely to cause delays. Palmdale Regional Airport and Lihue Airport least likely.

Flights arriving at Newark Liberty International Airport, Seattle–Tacoma International Airport, Atlanta International Airport are most likely to cause delays. Guam International Airport and Palmdale Regional Airport.

Delays most likely between 15:00-20:00, least likely between 4:00-9:00

Delays are less likely to happen in recent years, and more likely to happen the longer the flight.


## Other models

Since the actual model training part is fairly straightforward to use, everything up to and including the Feature Engineering section can be easily reused for other models in Spark. For example, below we illustrate how to run a random forest model on the same data.

```python

# Random forest 

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="Delay", featuresCol="features", numTrees=100)

start3 = time.time()

rfmodel=rf.fit(logistictrainingdata)

end3 = time.time()

timetaken3=end3-start3
print(timetaken3)

print('done')

```

Spark's RandomForestClassifier unfortunately does not provide an automatic way to retrieve a spectrum of precision and recall values depending on the probability threshold of your prediction. However it's relatively straightforward to retrieve the precision and recall for the default threshold (>=50% probability predicts a Delay).

```python

predictions = rfmodel.transform(logistictestdata)

predictions=predictions.select('Delay','rawPrediction','probability','prediction')

print('start')
print(predictions.take(50))
print('done')

from pyspark.mllib.evaluation import MulticlassMetrics

predictions=predictions.withColumn("prediction", predictions["prediction"].cast(DoubleType()))

predictions=predictions.withColumn("Delay", predictions["Delay"].cast(DoubleType()))

results = predictions.select(['prediction', 'Delay'])
predictionAndLabels=results.rdd
metrics = MulticlassMetrics(predictionAndLabels)

cm=metrics.confusionMatrix().toArray()
accuracy=(cm[0][0]+cm[1][1])/cm.sum()
precision=(cm[0][0])/(cm[0][0]+cm[1][0])
recall=(cm[0][0])/(cm[0][0]+cm[0][1])
print("RandomForestClassifier: accuracy,precision,recall",accuracy,precision,recall)

```

However, the choice of classification models on Spark is still relatively more limited compared to more commonly used programming languages by Data Scientists such as R and Python. For example the SVM implementation in Spark does not provide nonlinear kernels, and Multilayer perceptron classifier is the only neural network algorithm available.

## Benchmarks

Here are some sample run times for the Airline data analysis:

| Algorithm           | Data distribution | Number of cores | Runtime (seconds) |
|---------------------|-------------------|-----------------|-------------------|
| Logistic Regression | Mesos             | 324             |                   |
| Logistic Regression | Local             | 36              | 1066.61           |
| Random Forest       | Mesos             | 324             | 411.20            |
| Random Forest       | Local             | 36              | 3057.40           |

