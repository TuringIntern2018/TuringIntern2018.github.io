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

To conclude, before starting to use TensorFlow, we recommend weighing the pros and cons of using TensorFlow and consider what are the main concerns: speed? ease of use? portability? Finally, many other factors should be taken into consideration: programming abilities, computing resources available, desired metrics and output of the analysis, etc. Depending on those, it may or may not be useful to use TensorFlow.

Finally, it is worthwhile to mention that TensorFlow is in continuous evolution. This means that the features that are not available now, may be implemented very soon.
