Introduction...

## Airline data

The data that we analyse in the following can be downloaded from http://stat-computing.org/dataexpo/2009/. The data contain records of all commericial flights within the USA,  from October 1987 to April 2008. The data can be downloaded as 22 separate csv files, each containing the data for one year. When unzipped, the files take up 12 GB. 

Each column in the csv files corresponds to one of the following covariates: `Year` comprised between 1987 as 2008, `Month`, `DayOfMonth`, `DayOfWeek` expressed as integers (for the days of the week, 1 is Monday), `DepTime` and `ArrTime` are the actual arrival and departure local times in the hhmm format, `CRSDepTime` and `CRSArrTime` are the scheduled arrival and departure local times in the hhmm format, `UniqueCarrier` is	the unique carrier code, `FlightNum` the flight number, `TailNum`	the plane tail number, `ActualElapsedTime` in minutes, `CRSElapsedTime`	in minutes, `ArrDelay`	arrival delay, in minutes, `DepDelay`	departure delay, in minutes, `Origin`	origin IATA airport code, `Dest`	destination IATA airport code, `Distance` in miles.

In what follows, we binarise the `DepDelay` column, setting each value to `True` if the `DepDelay` is greater than 0, and False otherwise. We use this variable as our response, and 

There are also other variables that we do not take into consideration, since they cannot used to predict delays: `AirTime`	in minutes, `TaxiIn`	taxi in time, in minutes, `TaxiOut`	taxi out time in minutes, `Cancelled`	was the flight cancelled?, `CancellationCode`	reason for cancellation (A = carrier, B = weather, C = NAS, D = security), `Diverted`	1 = yes, 0 = no, `CarrierDelay` in minutes, `WeatherDelay` in minutes, `NASDelay`	in minutes, `SecurityDelay`	in minutes, `LateAircraftDelay`	in minutes.

## Logistic regression with TensorFlow

We make use of Estimators, a high-level TensorFlow API that includes implementations of the most popular machine learning algorithms. You can lear more about Estimators here: https://www.tensorflow.org/guide/estimators.

In particular, in order to perform linear regression, we are going to use the LinearClassifier etimator. Instantiating, and training a LinearClassifier is really simple. Assuming to have defined a set of numeric columns `my_numeric_columns` and categorical columns `my_categorical_columns`, we can istantiate a LinearClassifier as follows:
```
import tensorflow as tf
classifier = tf.estimator.LinearClassifier(
            feature_columns=my_numeric_columns+my_categorical_columns)
```
For the airline data, the columns can be defined as:
```

year = fc.categorical_column_with_vocabulary_list('Year', ['1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008'])
month = fc.categorical_column_with_vocabulary_list('Month', ['1','2','3','4','5','6','7','8','9','10','11','12'])
dayofmonth = fc.categorical_column_with_vocabulary_list('DayofMonth', ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20', '21', '22', '23', '24', '25', '26', '27', '28', '29','30', '31'])
dayofweek = fc.categorical_column_with_vocabulary_list('DayOfWeek', ['1','2','3','4','5','6','7'])
deptime = fc.numeric_column('DepTime')
arrtime = fc.numeric_column('ArrTime')
uniquecarrier = fc.categorical_column_with_hash_bucket('UniqueCarrier', hash_bucket_size=1000)
flightnum = fc.categorical_column_with_hash_bucket('FlightNum', hash_bucket_size=10000)
arrdelay = fc.numeric_column('ArrDelay')
depdelay = fc.numeric_column('DepDelay')
origin = fc.categorical_column_with_hash_bucket('Origin', hash_bucket_size=1000)
dest = fc.categorical_column_with_hash_bucket('Dest', hash_bucket_size=1000)
distance = fc.numeric_column('Distance')
cancelled = fc.categorical_column_with_vocabulary_list('Cancelled',['0','1'])
diverted = fc.categorical_column_with_vocabulary_list('Diverted', ['0','1'])
```

```
classifier.train(train_inpf)
```
where we have defined 
You can read more about Estimators here https://www.tensorflow.org/guide/estimators 


## Pros and cons of using TensorFlow

