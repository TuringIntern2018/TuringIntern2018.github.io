# Logistic and linear regression with TensorFlow

We make use of Estimators, a high-level TensorFlow API that includes implementations of the most popular machine learning algorithms. Here, in order to perform linear regression, we use the LinearClassifier estimator. You can learn more about Estimators on the TensorFlow official website: https://www.tensorflow.org/guide/estimators.

## Training
Instantiating and training a LinearClassifier is very simple. Assuming to have defined a set of numeric columns `my_numeric_columns` and categorical columns `my_categorical_columns`, we can istantiate a LinearClassifier as follows:
```python
import tensorflow as tf
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)
```
For example, for the airline data, the columns can be defined as:
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
```
Note that we have used three types of columns: `fc.numeric_column`, for continuous variables, `fc.categorical_column_with_vocabulary_list` for categorical variables for which all the classes are known and can be easily enumerated, `fc.categorical_column_with_hash_bucket` for categorical variables with a high number of classes (such as `FlightNum`). The parameter `hash_bucket_size` is an upper bound on the number of categories. More information about the different types of feature columns available for TensorFlow estimators can be found at https://www.tensorflow.org/guide/feature_columns.

For clarity of exposition, we divide them into numeric and categorical columns:
```python
my_numeric_columns = [deptime, arrtime, distance] #depdelay
my_categorical_columns = [year, month, dayofmonth, dayofweek, uniquecarrier, flightnum, origin, dest, cancelled, diverted]
```
Once the Estimator has been instantiated, it can be easily trained with the `train` method:
```python
classifier.train(train_inpf)
```
where `train_inpf` is the input function that feeds the data into the function. 

## Defining an input function

The input function `train_inpf` is defined in four steps. 

### 1. Defining input format 

First, we need to define the names of the columns in the dataset `CSV_COLUMNS`, the corresponding default values `DEFAULTS`, and the name of the response variable `LABEL_COLUMN`.
```python
CSV_COLUMNS = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'UniqueCarrier', 'FlightNum',  'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'Cancelled', 'Diverted']
DEFAULTS = [[""], [""], [""], [""], [0], [0], [""], [""], [0.], [0.],[""], [""], [0], [""],[""]]
LABEL_COLUMN = 'ArrDelay'
```
Note that we have chosen `[""]` as a default for the categorical variables, `[0]` for the integer variables, and `[0.]` for the continuous variables. Defaults also define the type of the input column to be loaded from file, so it is important that they match the variable type. 

### 2. Parsing csv files

To parse the csv files, we first need to be matched by the input files. In this case, after downloading the data in csv format in a dedicated folder, we can easily indicate to our parser that we want to train the model on the year 2006, 
```python
train_file = "2006.csv"
```
or that we want to use *all* the data in that folder during the training step. 
```python
train_file = "*.csv"
```

Now we can define the parser:
```python
def parse_csv(value):
      tf.logging.info('Parsing {}'.format(data_file))
      columns = tf.decode_csv(value, record_defaults=DEFAULTS, select_cols = [0, 1, 2, 3, 4, 6, 8, 9, 14, 15, 16, 17, 18, 19, 21], na_value="NA")
      features = dict(zip(CSV_COLUMNS, columns))
      labels = features.pop('ArrDelay')
      # Define the two classes for logistic regression
      # If the DepDelay is greater than 0, than the label is True (i.e. the flight was delayed)
      # Otherwise, 
      classes = tf.greater(labels, 0) 
      return features, classes
```

### 3. Defining the input function

In the input function, we need to 
* create a list of file names that match the pattern given in the file name `data_file`
* parse the text files 
* shuffle the data 
* choose the number of times that the stochastic gradient descent algorithm is going to go through the dataset (number of epochs)
* get a batch of data

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
      # Get a batch of data of size bathc_size
      dataset = dataset.batch(batch_size)
      return dataset
```

### 4. Defining a wrapper for the input function 
Finally, since the arguments of `classifier.train` cannot take any input, we have to wrap our input functions into a new function that does not take any argument:
```python
train_inpf = functools.partial(input_fn, train_file, num_epochs=1, shuffle=True, batch_size=100)
```

## Testing and prediction

The wrappers for the input functions of the evaluation and prediction steps can be defined similarly to before: 
```python
eval_inpf = functools.partial(input_fn, predict_file, num_epochs=1, shuffle=False, batch_size=100)
predict_inpf = functools.partial(input_fn, predict_file, num_epochs=1, shuffle=False, batch_size=100)
```

Just like training, all you need for the evaluation of an Estimator is encapsulated in one function:
```python
result = classifier.evaluate(test_inpf)
```

The output of the evaluation is a set of metrics that can be displayed 
```python
for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))
```
Similarly for prediction:
```python
pred_results = classifier.predict(input_fn=predict_inpf)
for i in range(10):
    print(next(pred_results))
```
Before moving to the next section, note that if you train the LinearClassifier (as of 11 September 2018), this will print a warning:
```
WARNING:tensorflow:Trapezoidal rule is known to produce incorrect PR-AUCs; please switch to "careful_interpolation" instead.
```
To avoid this problem, you can define an alternative function to calulate the area under the curve:
```python
def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }
```
and add it to your classifier:
```python
classifier = tf.contrib.estimator.add_metrics(classifier, metric_auc)
```
Since the new metric has the same name of the existing one, the latter will be overwritten.

## Retrieving the regression coefficients

Finally, if you want to retrieve the regression coefficients, you can use the following function, that returns the weight names and the corresponding set of coefficients:
```python
def get_flat_weights(model):
   weight_names = [
       name for name in model.get_variable_names()
       if "linear_model" in name and "Ftrl" not in name]
   for name in model.get_variable_names():
       print(name)
       print(model.get_variable_value(name))
   weight_values = [model.get_variable_value(name) for name in weight_names]
   weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)
   return weight_names, weights_flat
```
The full code of this tutorial can be found at http://acabassi.github.com/linear-regression-tensorflow/...

## Linear regression

If instead we wanted to predict exactly the flight delays in minutes, we could have done exactly the same as above, replacing `LinearClassifier` with `LinearRegressor` and not binarising the `ArrDelay` variable in the input function.

The full code to perform linear regression on the airline data the can be found at http://acabassi.github.com/linear-regression-tensorflow/...
