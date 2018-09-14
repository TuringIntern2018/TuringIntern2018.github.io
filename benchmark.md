# Benchmarking Regression algorithms with Apache Spark

On this page we give an overview of how we conducted benchmarks on Linear Regression in Spark, on generated, synthetic, normally distributed data of a range of sizes under different settings on the Cray-Urika GX. Broadly speaking, our benchmarks were conducted for two types of data sizes, 'Large n, small p', where n, the number of rows, is much larger than the number of columns p, as well as the opposite case of Large p, small n. These two scenarios are the most commonly encountered in practical applications. 'Large n, small p' is typically encountered when a small amount of data is collected frequently over a large number of instances, for example transactional data. 'Large p and small n', or high dimensional datasets arise in applications such as genomic data.

We run the benchmarks via a bash script which calls a python file under different input parameters through argparse. The Python file contains the PySpark script that runs the actual linear regression algorithm to be benchmarked. The variable settings are:

'coresmax', number of maximum cores. Cray-Urika GX has 324 cores in total. You can set this number to be greater than 324, but it's just effectively capped at 324.

'executorcores', number of cores used by each executor or node. Spark has 36 CPU cores per node. Note this must be less than 'coresmax' otherwise Spark will not have enough cores to initiate and return an error.

'executormemory', amount of memory assigned to each executor. In general (min(coresmax, 324)/executorcores')*executormemory<=1800 should be true if you want Spark to run on coresmax cores, otherwise Spark will scale down on the number of cores used due to the lack of memory. In general for other clusters, replace 324 by the (maximum) number of CPU cores and 1800 by the total memory of your cluster, respectively.

'n', number of rows in the data

'p', number of columns in the data

'maxiter', maximum number of iterations of the linear regression

'lamda', parameter controlling how much penalisation you want

'alpha', parameter between 0 and 1 controlling type of parametrisation (elastic net). 1 denotes LASSO/L1 penalisation, 0 denotes Ridge/L2 penalisation. Any number between 0 and 1 is a weighted linear combination of both.

The bash code looks like this:

```bash

for i in {4..9}; do for j in {1..2}; do for k in 1 5 10 22; do for l in 1 10 100; do for z in 0 0.1; do for x in 0 0.5 1; do for c in 1 4 16 36; do for v in 1 16 256 512 36000; do 
    if [ $v -ge $c ]; then 
    spark-submit LRLargeNsmallPBenchmark.py --n $((10**$i)) --p $((10**$j)) --executormemory $((10*$k))g --maxiter $((10*$l)) --lamda $z --alpha $x --executorcores $c --coresmax $v; 
    else
    	echo coresmax $v smaller than executorcores $c
    fi
done;
done;
done;
done;
done;
done;
done;
done


```


## Large n, small p



## Large p, small n