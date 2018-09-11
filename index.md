*(Introduction common to both...)*

# Airline data

The data that we analyse in the following can be downloaded from http://stat-computing.org/dataexpo/2009/. The data contain records of all commericial flights within the USA,  from October 1987 to April 2008. The data can be downloaded as 22 separate csv files, each containing the data for one year. When unzipped, the files take up 12 GB. 

Each column in the csv files corresponds to one of the following covariates: `Year` comprised between 1987 as 2008, `Month`, `DayOfMonth`, `DayOfWeek` expressed as integers (for the days of the week, 1 is Monday), `DepTime` and `ArrTime` are the actual arrival and departure local times in the hhmm format, `CRSDepTime` and `CRSArrTime` are the scheduled arrival and departure local times in the hhmm format, `UniqueCarrier` is the unique carrier code, `FlightNum` the flight number, `TailNum` the plane tail number, `ActualElapsedTime` in minutes, `CRSElapsedTime` in minutes, `ArrDelay` arrival delay, in minutes, `DepDelay` departure delay, in minutes, `Origin` origin IATA airport code, `Dest` destination IATA airport code, `Distance` in miles.

In what follows, we binarise the `DepDelay` column, setting each value to `True` if the `DepDelay` is greater than zero, and False otherwise. We use this variable as our response, and 

There are also other variables that we do not take into consideration, since they cannot used to predict delays: `AirTime` in minutes, `TaxiIn`	taxi in time, in minutes, `TaxiOut`	taxi out time in minutes, `Cancelled` was the flight cancelled?, `CancellationCode`	reason for cancellation (A = carrier, B = weather, C = NAS, D = security), `Diverted` 1 = yes, 0 = no, `CarrierDelay` in minutes, `WeatherDelay` in minutes, `NASDelay`	in minutes, `SecurityDelay` in minutes, `LateAircraftDelay` in minutes.

# Logistic regression with TensorFlow

We make use of Estimators, a high-level TensorFlow API that includes implementations of the most popular machine learning algorithms. You can lear more about Estimators here: https://www.tensorflow.org/guide/estimators. Here, in order to perform linear regression, we use the LinearClassifier estimator. 

## Training
Instantiating and training a LinearClassifier is very simple. Assuming to have defined a set of numeric columns `my_numeric_columns` and categorical columns `my_categorical_columns`, we can istantiate a LinearClassifier as follows:
```python
import tensorflow as tf
classifier = tf.estimator.LinearClassifier(
            feature_columns=my_numeric_columns+my_categorical_columns)
```
For the airline data, the columns can be defined as:
```python
import tensorflow.feature_column as fc

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
Note that we have used three types of columns: `fc.numeric_colimn`, for continuous variables, `fc.categorical_column_with_vocabulary_list` for categorical variables for which all the classes are known and can be easily enumerated, `fc.categorical_column_with_hash_bucket` for categorical variables with a high number of classes (such as `FlightNum`). The parameter `hash_bucket_size` is an upper bound on the number of categories. More information about the different types of feature columns available for TensorFlow estimators can be found at https://www.tensorflow.org/guide/feature_columns.

For clarity of exposition, we divide them into numeric and categorical columns:
```python
my_numeric_columns = [deptime, arrtime, distance] #depdelay
my_categorical_columns = [year, month, dayofmonth, dayofweek, uniquecarrier,
            flightnum, origin, dest, cancelled, diverted]
```
Once the Estimator has been instantiated, it can be easily trained with the `train` method:
```python
classifier.train(train_inpf)
```
where `train_inpf` is the input function that feeds the data into the function. 

Before moving to the next section, note that if you train the LinearClassifier as it is (as of 11 September 2018), this will print a warning:
```
WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
```
To avoid this problem, you can define an alternative function to calulate the area under the curve
```
def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }
```
and add it to your classifier
```
classifier = tf.contrib.estimator.add_metrics(classifier, metric_auc)
```
Since the new metric has the same name of the existing one, the latter will be overwritten.

## Defining an input function

The `train_inpf` is defined in three steps. 

### Defining input format 

First, we need to define the names of the columns in the dataset `CSV_COLUMNS`, the corresponding default values `DEFAULTS`, and the name of the response variable `LABEL_COLUMN`.
```python
CSV_COLUMNS = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'UniqueCarrier', 'FlightNum',  'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'Cancelled', 'Diverted']
DEFAULTS = [[""], [""], [""], [""], [0], [0], [""], [""], [0.], [0.],[""], [""], [0], [""],[""]]
LABEL_COLUMN = 'ArrDelay'
```
Note that we have chosen `[""]` as a default for the categorical variables, `[0]` for the integer variables, and `[0.]` for the continuous variables. Defaults also define the type of the input column to be loaded from file, so it is important that they match the variable type. 

### Parsing csv files

To parse the csv files, we first need to be matched by the input files. In this case, after downloading the data in csv format in a dedicated folder, we can easily indicate to our parser that we want to use *all* the data in that folder during the training step. 
```
train_file = "*.csv"
```

Now we can define the parser:
```python
def parse_csv(value):
      tf.logging.info('Parsing {}'.format(data_file))
      columns = tf.decode_csv(value, record_defaults=DEFAULTS, select_cols = [0, 1, 2, 3, 4, 6, 8, 9, 14, 15, 16, 17, 18, 19, 21], na_value="NA")
      features = dict(zip(CSV_COLUMNS, columns))
      labels = features.pop('ArrDelay')
      classes = tf.greater(labels, 0)  # binary classification
      return features, classes
```

### 
```python
def input_fn(data_file, num_epochs, shuffle, batch_size, buffer_size=1000):
      # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(data_file)
      # Read lines from text files
      textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
      # Parse text lines as comma-separated values (CSV)
      dataset = textlines_dataset.map(parse_csv)
      if shuffle:
          dataset = dataset.shuffle(buffer_size=buffer_size)
      # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
      dataset = dataset.repeat(num_epochs)
      dataset = dataset.batch(batch_size)
      return dataset
```

### Wrapper 
Finally, since the arguments of `classifier.train` cannot take any input, we have to wrap our input functions into a new function that does not take any argument:
```python
train_inpf = functools.partial(input_fn, train_file, num_epochs=1, shuffle=True, batch_size=100)
```

# How to parallelise TensorFlow code


# Pros and cons of using TensorFlow

