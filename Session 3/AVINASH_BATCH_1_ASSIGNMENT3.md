## Avinash Kappa
##### This markdown file contains Assignment-3 of Session 3  

**What are activation functions and why do we need them?**
Activation functions are important components of neural networks. Both in Perceptron models or Neural Network models, they take in some data, add non linearity to the data before giving the output to the next layer. The terms linear and non-linear can be intuitively understood by considering them as straight lines and non-straight curves. Without the activation functions, the perceptron models that take weighted sum of inputs in each layer or the convolution models that use kernels by element wise multiplication, are just a stack of layers that perform linear regression and can detect or model data that is linearly detectable or separable.

Now that we know why we need to use activation functions, let's see which activation function is giving the best results. There are many activation functions that are used.Let's see how some of the most used activation functions work.

**Sigmoid Function**

Sigmoid is one of the most used activation functions. Given a value to it, it squashes it and outputs value between 0 and 1. So Sigmoid's output can be treated as probability values. As seen in the curve below, from a certain negative value, the curve gets flat and close to 0. Similarly for some positive values, the curve gets flat and constant. The gradients at these regions will be 0 and this leads to vanishing gradient.
![](https://cdn-images-1.medium.com/max/1600/1*XxxiA0jJvPrHEJHD4z893g.png)

**Rectified Linear Unit** 

As shown in the plot, ReLu is a simple activation function. It takes some numerical input **z** and outputs **0** if z is negative, **z** otherwise. Though ReLu is not entirely linear, we can see that it has a linear form for values that are positive. <br>
ReLu is easy to compute. It's curve is horizontal for values below 0 and incrementing linearly for values above 0. What this means is that, during the forward pass, the neurons that are fed with negative data will be set to 0. This might seem like a problem but considering the fact that the initial weights are random,could be negative and should be vastly optimized to minimize the cost, it is ok to have some neurons that are set to 0. This induces some kind of sparsity into the network.
<br>
But the same reasons that make ReLu simple and easy to implement also lead to some drawbacks. Consider the horizontal line again for values below 0. During the backpropagation phase, the gradient for this region will be 0 which means the neurons with this gradient will never be updated. There are other variants of ReLu that try to fix this problem.

**Leaky ReLu**

As we identified, the problem is with the horizontal region in the ReLu curve. So instead of giving 0 for all negative values, it's better to have some output for the region with the horizontal curve. This can be done using Leaky ReLu:

**R(z) = max(0.1z,z)**

With Leaky ReLu, for the negative values, the slope will not be zero. We give it a slight positive slope of 0.1 so that all the neurons with negative values wont be set to 0. Though this partially fixes the dying gradient problem in ReLu, it can still face the same problems in some cases.


**ELU**

Exponential Linear Unit function is of the form:

![](http://saikatbasak.in/public/img/elu.jpg)

In ELU, instead of giving the negative region a slight slope of 0.1, we have α(exp(x) - 1) which gives mean activations of zero that helps in faster convergance.

**SELU**

Scaled Exponential Linear Units is of the form:
![](https://www.hardikp.com/assets/selu.png)  

From the paper [here](https://arxiv.org/abs/1706.02515), SELUs induce self-normalizing properties which makes training deep networks with many layers and their learning robust. Though SELUs can give better outcomes than RELUs, it takes some specific configuration that should be used to achieve those results.

The field of Deep Learning and Neural Network is evolving too quickly. Though Sigmoid was widely used as activation function earlier, now the Deep Learning community has switched to Relu as its favorite activation function.  
<br>
<br>
<hr>
<br>

**What is Normalization?**

 Data varies widely depending on the problem. For example, in a task like estimating the salary of a potential employee, the data might be the employee's name, age, current salary, expected salary, experience etc. Each of these could be chosen as data features which means they play a part in data modeling and prediction. As we can observe, each feature's value is in a different scale. Age might vary between 20 to 80. Salary might vary between 200000 to 2500000 and experience between 0 to 15. Considering that the models interpret the values and act accordingly, this doesn't seem good for the model. This is where normalization comes in. Using normalization, we can bring all the features to a uniform scale so that it's easier for the model to optimize.

**Batch Normalization**
This is one variant of normalization. If you have any experience running a deep network, you must have used the parameter **batch_size**. Instead of taking each sample or datapoint at a time to extract features, learn and then optimize, we do it for a batch of samples. This way the model will train and optimize a batch of samples instead of one at a time. The same can be applied for normalization. Instead of applying normalization only at the input layer, we do it at every other layer too in the network. This is done by taking the input to a neuron across a batch of samples, their mean and variance are caluclated and then the normalization is applied. This way, the input to the non linear activation functions will be standardized and normalized. Normalization for the hidden/intermediate layers is done before applying the activation function. This reduces training time and allows higher learning rates so that the model can converge sooner.

**Weight Normalization**
This is similar to batch normalization. The main difference being, while batch normalization considers the mean and variance of a batch of layers for a batch of samples, weight normalization takes the magnitude of weights of each neuron in a layer. Since it acts on each layer independently, the results are more deterministic than batch normalization.

**Layer Normalization**
Though batch norm has led to improved results, specially in feed forward convolutional networks, they fail in cases where batch size is 1 or when the data samples are fed one after the other in an online mode. Unlike batch normalization, layer normalization is applied for every sample, one at a time. This works better than the former when batch size is 1. 

<br>
<hr>

**What is Regularization?**

The utlimate goal of every learning model is to get its predictions/classifications correct on unseen data. Given some existing data, the model is trained and is allowed to learn from it. The model's performance depends on how good its accuracy is on new/unseen data. This leads to a widely talked about term in Machine Learning - **overfitting**. Overfitting is what happens when the model does well on training data but does poorly on unseen data. Making sure that the model generalizes well for unseen data during testing is as important as minimizing the error during training of the model. Regularization helps in making the model generalize for unseen data. 

**Dropout**

Dropout is one of the ways of applying regularization. Neural networks are universal approximators. So given enough data and parameters, the networks can learn any function for some known data. Though this might sound efficient, it comes with its own problems. If the model is given all the data and parameters it needs to learn a function, chances are that it does poorly on unseen data. This leads to overfitting and very poor accuracy results. And that is why it is not a bad idea to intentionally introduce some kind of unlearning into the networks. Dropout does exactly that.

<p align="center"><img src="https://mlblr.com/images/dropout.gif"/></p>


Dropout when applied on a layer, disables/deactivates some of the layer's neurons. The extent to which the neurons are deactivated depends on the **p** value assigned to the dropout. For example, when p = 0.5, half of the neurons from that layer are shut off. These neurons are randomly chosen which means each time a different set of neurons are selected depending on the amount of dropout. By deactivating a set of neurons from a layer, we are not only lessening the learning by some amount but also forcing the rest of the neurons to learn better. Both of these contribute to better testing accuracy. 


