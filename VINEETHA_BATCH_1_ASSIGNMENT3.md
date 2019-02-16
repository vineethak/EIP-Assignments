# Session 3 Assignment

## Dropout:
The basic concept of dropout  is to randomly remove hidden units from the neural network. This is done by introducing a mask parameter $m^k$ (where k is the layer number) which is multiplied with the output of hidden layer. $m^k$ either takes value 0 or 1 based on the dropout probability. The probability $p$ of neuron being dropped is in general 0.5 (can vary also but 0.5 is popular choice) hence  its contribution to the downstream neurons is nullified for the whole single pass in forward propagation. Consequently, no weight update is done for the drooped neuron. In the similar fashion, at each layer each hidden unit is drooped with a probability $p$ or kept with a probability $(1-p)$. Therefore, due to dropout at each pass of the training we have a reduced network as compared to the actual network.

## Why drop the hidden units?
In traditional convolution neural networks each feature map is weighted sum of all the feature maps of the previous layers, during back propagation also the network tries to minimize the distance between these weighted averages and the ground truth. By doing so we are creating co-dependency among neurons and the features extracted from it, will make sense only when combined with other neurons output. As the depth of network increases this co-dependency further increases making the model unstable. This is referred to in context of neuron as complex co-adaptation. Dropout is an approach of regularization in neural networks which helps reducing interdependent learning amongst the neurons. By introducing dropout a hidden unit will generate a feature which is general and not just part of the feature which only when combined with the feature extract of other hidden unit makes sense. Because of this it is one of the most common regularization method. Any regularization prevents over-fitting by adding a penalty to the loss function. 

## Network

### Training Phase: 
For each hidden layer, for each training sample, for each iteration, ignore (zero out) a random fraction, p, of nodes (and corresponding activations).

**Forward propagation**
•	Use random binary mask $m^k$
•	Layer pre-activation –  $a_k(x) = b^k + W_k*h_{(k-1)}*x$$
•	Hidden layer activation - $h_k(x) = g(a_k(x))*m^k$
•	o/p layer activation -$h_{(L+1)(}x) = o(a_{(L+1)}(x)) = f(x)$

**Back propagation**
 - We need to replace h(x) with $h(x)*m_k$

### Testing Phase 
- Use all activations but reduce them by a factor p (to account for the missing activations during training).
-	If you have used a probability of 0.5, you need to replace each of these masks  for all the neurons with 0.5.

### Note:
-	Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.
- With H hidden units, each of which can be dropped, we have $2^H$ possible models. In testing phase, the entire network is considered and each activation is reduced by a factor p.

### References: 
https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5
http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf 

## Batch Normalization
Normalization/standardization techniques are employed in to bring all the data in one common scale. A normaliztion in general refers to bringing all the data on a scale of 0-5. While standardization on other hand means making the mean of data to zero, i.e. for each data point we perform $(x-m)/s$ where m is mean and s is standard deviation. 

### Why normalization is required?
- Features might be on different scale (eg : age, salary of a person)
- To prevent skewed data. The larger data points in the non – normalized data can cause imbalanced gradients which therefore cause exploding gradient problem
- By normalizing the data we can increase training speed and get rid of exploding gradient problem

### Looks like normalization is pretty much handling all the skewedness in the data. Why perform batch normalization?
In neural network in spite of having normalized data we still have another problem. This is because when the network learns we update the weights with each epoch based on SGD, if the learned weights of one filter is very very much larger than others. This large weight will cause the output from corresponding neuron to be very large, this will be cascaded throughout the network causing instability. To prevent this, we apply batch normalization to the output of hidden activation layers.

### How is batch normalization done?
- For each output of hidden activation unit, we perform we perform $ x' = (x-m)/s $ i.e. make mean zero.
- Multiply this normalized o/p with arbitrary parameter g, i.e. $z = x'*g$
- Add another parameter b to this product => $z = (z + b)$
- This sets new standard deviation and mean for the data
- m,s,g,b all are trainable.

This method makes sure weights in the hidden layers do not go to arbitrary high value.	Batch normalization also takes care of covariance shift, that is if we have trained our model let’s say $h(X) = y$ , if the distribution of X changes , our model will not be able to predict $h(X')=y$. This is because for each iteration od a neural network, the distribution of hidden layer activation changes. By doing batch norm we are telling no matter how the weights and biases are changing, the mean and  variation of a particular hidden layer activation remains the same.

### References
- [Batch normalization in Neural Networks – Towards Data Science](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
- [Batch Normalization — What the hey? – Gab41](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b#.h7n8n32ww)

## Dense Net

In a DenseNet (short for Dense Convolutional Neural Network)  each layer receives i/p from all its preceding layers. For current layer, the feature-maps of all previous layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. This is different from resNet since in resNet we do feature summation while in denseNet we perform feature concatenation. Feature concatenation exploits the potential of the network through feature reuse. By Concatenating feature-maps learned by different layers the variation in the input of subsequent layers  is increased  thereby improving the efficiency. 

### Architecture of denseNet

DenseNet consists of series of blocks called **dense blocks**. This is done to facilitate downsampling which is one of the most essential part of convolutional networks to change the size of feature-maps. The image below consists of 5 dense blocks. With in a dense block all the  feature maps have same size, since feature concatenation is only possible if feature maps are of same size. The layers between dense blocks are called **transition layers**,  which perform 1x1 convolution, batch normalization and 2x2 average pooling. 


![Five-layers-of-a-DenseNet-block-with-a-growth-rate-of-4-feature-maps-per-layer-source.jpg](\:storage\0.i4c23tegb5b.jpg).

**Growth rate $k$** of a denseNet is defined as the increase in number of feature maps per layer, in other words it's the number of filters used per hidden layer. For $l^{th}$ hidden layer, if each hidden unit produces k featuremaps, the output of ℓ th layer would be k(current layer output) + k0( k0 is the number of channels in the input layer) +k ×(ℓ−1) (input feature-maps from the previous layer). Since denseNet are very efficient in preserving and re-using low level features it's hidden layers are very narrow (k =12 or 12 filters per layer was used in the original paper, this is very low as compared to CNN or ResNet. As a result denseNet have very less parameters, easy to train and network is much faster.

### Advantages of using a denseNet:
-	Error signal can be propagated to pervious layers directly
-	No. of parameters in dense net is lower than convolutional neural network
-	More diversified features
-	Classifier uses features from all the complexity levels, this tend to give more smoother decision boundary.

### Disadvantage
- Requires enormously huge amount of memory. Since all the feature maps are saved.

References:
-https://arxiv.org/abs/1608.06993
-http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf





