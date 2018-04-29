## Avinash Kappa
##### This markdown file contains Assignment-2b of Session 2  
##### Assignment-2a has been pushed to the GitHub repo - [click here](https://github.com/theimgclist/DeepLearningInternship/blob/master/Session%202/Session2_Assignment_Python%26Numpy.ipynb)

**Step 0**: Read input and output
In this assignment, I will be using simple numerical samples to illustrate backpropagation.
The input X and the output Y for the exercise are :

| X | Y |
| --------| ----- |
|1 0 1 0  | 1|
| 1 0 1 1 | 1 |
| 0 1 0 1 | 0|

**Step 1**: Initialize weights and biases with random values 
 
We are considering a neural network with one hidden layer. 
wh, bh, wout and bout are the parameters that get initialized to some random values. Using backpropagation their values are optimized.  
Let's generate the random values for the weights and bias:    

```Python
wh = np.random.randn(4,3)
wout = np.random.randn(3,1)
bh = np.random.randn(3)
bout = np.random.randn(1)
```


| X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E |
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.87 -0.88 -0.24 | -0.37 -0.30 0.30 |  |  | 0.42 | 0.14 | | 1 | 
| 1 0 1 1 | -0.27 -0.47 1.32 |                  |  |  | 0.31 |      |  | 1 | 
| 0 1 0 1 | -2.31 0.36 -0.79|                   | |  | -0.92 |       |  | 0 | 
|         | -1.74 -0.35 -0.49


**Step 2**: Calculate hidden layer input:


|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.87 -0.88 -0.24 | -0.37 -0.30 0.30 | -1.81 -0.82 -0.72  |  | 0.42 | 0.14 | | 1 | 
| 1 0 1 1 | -0.27 -0.47 1.32 |                  | -3.56 -1.17 -1.22 |  | 0.31 |      |  | 1 | 
| 0 1 0 1 | -2.31 0.36 -0.79|                   | -2.39 -1.13 1.13 |  | -0.92 |       |  | 0 | 
|         | -1.74 -0.35 -0.49

**Step 3**: Perform non-linear transformation on hidden linear input


|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.87 -0.88 -0.24 | -0.37 -0.30 0.30 | -1.81 -0.82 -0.72| 0.13 0.30 0.32| 0.42 | 0.14 | | 1 | 
| 1 0 1 1 | -0.27 -0.47 1.32 |                  | -3.56 -1.17 -1.22| 0.02 0.23 0.22| 0.31 |      |  | 1 | 
| 0 1 0 1 | -2.31 0.36 -0.79|                   | -2.39 -1.13  1.13| 0.08 0.24 0.75| -0.92 |       |  | 0 | 
|         | -1.74 -0.35 -0.49

**Step 4**: Perform linear and non-linear transformation of hidden layer activation at output layer


|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.87 -0.88 -0.24 | -0.37 -0.30 0.30 | -1.81 -0.82 -0.72| 0.13 0.30 0.32| 0.42 | 0.14 | 0.49 | 1 | 
| 1 0 1 1 | -0.27 -0.47 1.32 |                  | -3.56 -1.17 -1.22| 0.02 0.23 0.22| 0.31 |      | 0.50 | 1 | 
| 0 1 0 1 | -2.31 0.36 -0.79|                   | -2.39 -1.13  1.13| 0.08 0.24 0.75| -0.92 |     | 0.38 | 0 | 
|         | -1.74 -0.35 -0.49


**Step 5**: Calculate gradient of Error(E) at output layer



|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.87 -0.88 -0.24 | -0.37 -0.30 0.30 | -1.81 -0.82 -0.72| 0.13 0.30 0.32| 0.42 | 0.14 | 0.49 | 1 | 0.50
| 1 0 1 1 | -0.27 -0.47 1.32 |                  | -3.56 -1.17 -1.22| 0.02 0.23 0.22| 0.31 |      | 0.50 | 1 | 0.49
| 0 1 0 1 | -2.31 0.36 -0.79|                   | -2.39 -1.13  1.13| 0.08 0.24 0.75| -0.92 |     | 0.38 | 0 | -0.38
|         | -1.74 -0.35 -0.49

**Step 6**: Compute slope at output and hidden layer


| slope_hidden_layer |  slope_output |
| ------- | ------- |
0.248 0.244 0.243 | 0.235
0.249 0.246 0.246 | 0.234
0.249 0.246 0.217 | 0.240


**Step 7**: Compute delta at output layer


| delta_output |
| ----- |
0.117 |
0.116 |
-0.09 |

**Step 8**: Calculate Error at hidden layer


| error at hidden layer |
| ----- |
0.05   0.03   -0.10 |
0.04   0.03   -0.10 |
-0.03  -0.02   0.08 |


**Step 9**: Compute delta at hidden layer


| delta hidden layer |
| ----- |
0.012  0.008  -0.026 |
0.012  0.008  -0.026 |
-0.009 -0.007  0.018|


**Step 10**: Update weight at both output and hidden layer


|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.89 -0.86 -0.29 | -0.37 -0.30 0.30 | -1.81 -0.82 -0.72| 0.13 0.30 0.32| 0.43 | 0.14 | 0.49 | 1 | 0.50
| 1 0 1 1 | -0.28 -0.48 1.34 |                  | -3.56 -1.17 -1.22| 0.02 0.23 0.22| 0.35 |      | 0.50 | 1 | 0.49
| 0 1 0 1 | -2.29 0.38 -0.84|                   | -2.39 -1.13  1.13| 0.08 0.24 0.75| -0.93 |     | 0.38 | 0 | -0.38
|         | -1.74 -0.34 -0.50



**Step 11**: Update biases at both output and hidden layer



|    X  | wh | bh | hidden_layer_iput | hidden_layer_activations | wout | bout | output | Y | E
| ------------- | ----- | --- | ---- | ------ | ----- | ----- | ----- | ----- | -----
|1 0 1 0  | 0.89 -0.86 -0.29 | -0.35 -0.29 0.27 | -1.81 -0.82 -0.72| 0.13 0.30 0.32| 0.43 | 0.28 | 0.49 | 1 | 0.50
| 1 0 1 1 | -0.28 -0.48 1.34 |                  | -3.56 -1.17 -1.22| 0.02 0.23 0.22| 0.35 |      | 0.50 | 1 | 0.49
| 0 1 0 1 | -2.29 0.38 -0.84|                   | -2.39 -1.13  1.13| 0.08 0.24 0.75| -0.93 |     | 0.38 | 0 | -0.38
|         | -1.74 -0.34 -0.50


We have now done one cycle of feed forward and backpropagation. Let's compare the network's weights and biases :


| wh | wh_updated|
| ------------- | -------- |
| 0.87 -0.88 -0.24 | 0.89 -0.86 -0.29 
| -0.27 -0.47 1.32 | -0.28 -0.48 1.34
| -2.31 0.36 -0.79 | -2.29 0.38 -0.84
| -1.74 -0.35 -0.49|  -1.74 -0.34 -0.50 |



| bh | bh_updated|
| ------------- | -------- |
|-0.37 -0.30 0.30  | -0.35 -0.29 0.27 |



| wout | wout_updated|
| ------------- | -------- |
|0.42 | 0.49
|0.91 | 0.50
|-0.32| 0.38 |


| bout | bout_updated|
| ------------- | -------- |
|0.14  | 0.28 |

**Note**: We haven't considered learning rate in this example. In other words, the learning rate is 1.