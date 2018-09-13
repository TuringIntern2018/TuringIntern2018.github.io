# Airline data

In the following, we analyse the airline dataset, publicly available for download from http://stat-computing.org/dataexpo/2009/. The data contains records of all commericial flights within the USA,  from October 1987 to April 2008. It can be downloaded as 22 separate csv files, each containing the data for one year. When unzipped, the files take up 12 GB. 

Each column in the csv files corresponds to one of the following covariates. Among others:`Year` comprised between 1987 and 2008, `Month`, `DayOfMonth`, `DayOfWeek` expressed as integers (for the days of the week, 1 is Monday), `CRSDepTime` and `CRSArrTime` the expected arrival and departure local times in the hhmm format,`UniqueCarrier` the unique carrier code, `FlightNum` the flight number, `TailNum` the plane tail number, `CRSElapsedTime` expected flight time in minutes, `ArrDelay` arrival delay, in minutes, `DepDelay` departure delay in minutes, `Origin` origin IATA airport code, `Dest` destination IATA airport code, `Distance` in miles.

Using this information, we want to see if it is possible to predict whether a flight will be delayed or not, making use of the information available before the departure. Therefore, in what follows, we binarise the `ArrDelay` column, setting each value to True if the `ArrDelay` is greater than zero, and False otherwise. Using this variable as our response, we perform logistic regression on the other covariates. The goal is to be able to do out-of-sample prediction and identify which variables influence delays the most.

<!--- (There are also other variables that we do not take into consideration, since they cannot used to predict delays:  `CRSDepTime` and `CRSArrTime` the scheduled arrival and departure local times in the hhmm format, `AirTime` in minutes, `TaxiIn` taxi in time, in minutes, `TaxiOut` taxi out time in minutes, `Cancelled` was the flight cancelled?, `CancellationCode`	reason for cancellation (A = carrier, B = weather, C = NAS, D = security), `Diverted` 1 = yes, 0 = no, `CarrierDelay` in minutes, `WeatherDelay` in minutes, `NASDelay`	in minutes, `SecurityDelay` in minutes, `LateAircraftDelay` in minutes.) -->

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
The full code can be found at http://acabassi.github.com/linear-regression-tensorflow/...

## Linear regression

If instead we wanted to predict exactly the flight delays in minutes, we could have done exactly the same as above, replacing `LinearClassifier` with `LinearRegressor` and not binarising the `ArrDelay` variable in the input function.

The full code can be found at http://acabassi.github.com/linear-regression-tensorflow/...

# How to parallelise TensorFlow code
The guidelines given here to run linear and logistic regression with TensorFlow in a distributed setting are specific to the Urika-GX platform. However, they can be easily be generalised to any other platform by replacing the commands to start a job and the node names as needed.

## Setting up the cluster
From the TensorFlow website: *A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.*

Therefore, in order to run TensorFlow in parallel, you will need to define a set of workers and one or more parameter servers. With the Urika-GX platform, each worker or parameter server can be set up through `mrun`. Here is the bash script needed to run one parameter server and two workers on three different nodes (indicated by `node1`, `node2`, `node3`):
```shell-script
mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=ps \
            --task_index=0 \
            > output-parameter-server.txt &

mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node2 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-first-worker.txt &

mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node3 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=1 \
            > output-second-worker.txt &
```
On Urika-GX, each job must be launched by `mrun`. The `-n 1` and `-N 1` flags indicate the number of jobs to be started and the number of nodes to be used. `--cpus-per-task` sets the number of CPUs to be used for each job and `--nodelist` is used to specify the name of the node on which the job will run. Here, for simplicity, we assume to have three nodes named `node1`, `node2`, and `node3`.
The rest of the script can be used on any platform, as it is simply calling the Python script . 

In this example we allocate 36 CPUs to each parallel job. TensorFlow automatically detects all the CPUs available on the same node and decides which part of the job to allocate to each one. 
However, we found that in practice the stochastic gradient descent algorithms implemented in TensorFlow for linear and logistic regression only use one or two CPUs. Therefore, it is more efficient to run multiple jobs on the same node, by assigning multiple jobs to the same node, with different ports. Here we give a simple example with one parameter server and two workers, all running on node one. 

```shell-script
mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=ps \
            --task_index=0 \
            > output-parameter-server.txt &

mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-first-worker.txt &

mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-second-worker.txt &
```

## Split up the tasks between the parameter servers and the workers
The `linear-classifier-parallel.py` file will be a Python script containing:
* a parser, to process the input options such as the job name, task index, etc. 
* a `main` function, in which the TensorFlow cluster is set up and different tasks are assigned to the parameter servers and workers
* the code presented in the "Logistic regression with TensorFlow" section. The only difference is that, if more than one worker is set up, you need to make sure that each worker loads a different batch of data. To this end, you can use the `dataset.shard()` method as explained below. 

The parser can be defined as follows:
```python
import argparse
import sys
FLAGS = None

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.register("type", "bool", lambda v: v.lower() == "true")
      parser.add_argument("--ps_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
      parser.add_argument("--worker_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
      parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers")
      parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
      parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
      parser.add_argument("--l1", type=float, default=0.0, help="L1 regularisation strength")
      parser.add_argument("--l2", type=float, default=0.0, help="L2 regularisation strength")
      parser.add_argument("--batch_size", type=int, default=500, help="Batch size")
      FLAGS, unparsed = parser.parse_known_args()
      tf.logging.set_verbosity(tf.logging.WARN)
      tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```
Notice that it can also be used to provide other optional inputs to the main function, such as the L1 and L2 regularisation strength or the batch size. 

The `main` function is called by the parser and is used to set up the cluster, divide the work between the parameter server and the workers and then run the main part of your code.
```python
def main(_):
      ps_hosts = FLAGS.ps_hosts.split(",")
      worker_hosts = FLAGS.worker_hosts.split(",")

      # Create a cluster from the parameter server and worker hosts.
      cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

      # Create and start a server for the local task.
      server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
      
      # Split up the tasks between the parameter servers and the workers
      if FLAGS.job_name == "ps":
            server.join()
      elif FLAGS.job_name == "worker":
            with tf.device(tf.train.replica_device_setter(
                  worker_device="/job:worker/task:%d" % FLAGS.task_index,
                  cluster=cluster)):
                  # Training and prediction go here
                  # ...
            
                  if FLAGS.task_index==0:
                        # Print output with only one worker
                        
```

With Estimators, if the data is read from file, the only thing that changes between the code for sequential and parallel jobs is how the data is loaded by each worker. In order to specify which batch of data should be read by each worker, you can use the `dataset.shard` method as described below. This will make sure that the first batch of data will be read by worker 0, the second one by worker 1, and so on.
```python
def input_fn(data_file, num_epochs, shuffle, batch_size, buffer_size=1000):
      # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(data_file)
      # Read lines from text files
      textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
      # Parse text lines as comma-separated values (CSV)
      dataset = textlines_dataset.map(parse_csv)
      # Assign different batches to each worker
      dataset = dataset.shard(FLAGS.num_workers, FLAGS.task_index)
      if shuffle:
          dataset = dataset.shuffle(buffer_size=buffer_size)
      # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
      dataset = dataset.repeat(num_epochs)
      # Get a batch of data of size batch_size
      dataset = dataset.batch(batch_size)
      return dataset
```
The full code can be found at http://acabassi.github.com/linear-regression-tensorflow/...

# TensorFlow for statisticians

More and more statisticians nowadays have huge, rich datasets available which unfortunately are too large to be analysed with classical tools for statistical analysis such as R. In these cases, machine learning tools can come in handy, since they are specifically developed for large datasets. One of the most popular machine learning libraries at the moment is TensorFlow, developed and maintained by Google which has recently become open source. It is written in C++ and Python and has several different APIs. In the what follows, we explain a few things that we think any statistician should know before starting to use TensorFlow for two of the most used statistical techniques: linear and logistic regression.

The great thing about using TensorFlow for linear and logistic regression is that both algorithms are implemented in the high-level Estimators API under the names of LinearRegressor and LinearClassifier respectively. Estimators are wrappers that contain everything that is needed in order to instantiate a machine learning model, and to train it and evaluate it on any type of device (or set of devices): CPUs, GPUs, and also the Tensor Processing Units (TPUs) recently developed by Google and tailored for TensorFlow. 

Therefore, instantiating and training such models is very easy: the code needed to create a model and train it on the data fits in just two lines! On top of that, for each Estimator the user can specify a set of parameters to choose the optimisation algorithm and the strength of the L1 and L2 penalties.  Indeed, TensorFlow is equipped with a broad range of optimisers. These are different variants of the stochastic gradient descent algorithm, including some new ones  recently proposed by researchers at Google in order to cope with datasets with very large number of covariates. 

Another great feature of TensorFlow is that it doesn’t require to load the full dataset in memory. At each iteration of the stochastic gradient descent, a batch of data is loaded into memory, used to upgrade the gradient and then discarded. In some cases, it can be useful to go through the data more than once, until the SGD algorithm has converged. The number of times that each batch is considered can specified by the user (and is referred to as the number of “epochs” in machine learning jargon.

On the other hand, we found that Estimators lack some basic functionalities that a statistician would expect. For instance, while metrics such as the precision and recall of the Estimators are automatically computed in the evaluation step, retrieving the coefficients (or weights, as they are called in the machine learning community) of each covariate is not straightforward. For categorical covariates, for example, there exists a method that allows the user to get a list of coefficients, but this does not include any indication about the category to which each coefficient corresponds. This can be an issue when the model includes covariates with a large number of categories (in the airplane data example in the ‘Logistic regression with TensorFlow’ section, these are the carrier name, origin and destination airports, and flight number). Similarly, normalising the data before the training step can be tricky, especially if the design matrix contains a mix of numeric and categorical covariates. 

Moreover, when running on CPUs, regression models than can be fitted in just a few minutes with other machine learning libraries (such as Apache Spark) can take much longer with TensorFlow. 
In addition to that, even though some of the SGD algorithms available in TensorFlow have been specifically developed for datasets with up to a few billion covariates, the time required to run linear and logistic regression with TensorFlow on one or more CPUs quickly explodes for increasing values of p. 
Even with a large number of CPUs on the same machine, it is easy to observe that TensorFlow makes use only a few of them to train Estimators. 

Using distributed TensorFlow can help, but does not entirely solves the problem. Indeed, when splitting the computations between multiple workers, each worker is responsible for loading a different batch of data and asynchronously updating the model parameters, that are stored on a separate parameter server. Therefore, when using n workers, the time needed to train a LinearRegressor or LinearClassifier is roughly divided by n. Unfortunately the procedure to start a large number of workers is quite cumbersome, and starting more than 10 workers is not very convenient (for details, see the ‘How to parallelise TensorFlow code’ section). 

Finally, it is worthwhile to mention that TensorFlow is in continuous evolution. This means that the features that are not available now, may be implemented very soon. At the same time, most classes and methods are rapidly  

To conclude, before starting to use TensorFlow, we recommend weighing the pros and cons of using TensorFlow and consider what are the main concerns: speed? ease of use? portability? Finally, many other factors should be taken into consideration: programming abilities, computing resources available, desired metrics and output of the analysis, etc. Depending on those, it may or may not be useful to use TensorFlow.
