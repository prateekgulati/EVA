
# Proceeding an Image Classification Problem
This project discusses on approaching an image classification problem.
For this problem, MNIST dataset is being used.

#### **Goal: Achieve 99.40% accuracy in less than 15k total parameters**

The first step is the architecture of the model.
We start with defining an architecture through a vanilla model. 

## Target 1: Vanilla network

The first thing to look at while laying out the architecture is the *receptive field*. Towards the end of the model, the receptive field should be equal to the size of the input image.
The number of layers are also decided by keeping the same concept in mind.
We start with a convolution block that contains multiple *3x3 convolution* and in each succeeding convolution, the number of kernels/filers are increased by a factor of 2
This convolution block is followed by a transition block with a *1x1 convolution* to reduce the number of channels and a *maxPool* layer to double the receptive field.

> Note: Transition block is avoided towards the end of the network,
> because that can result in loss of important information.

Each convolution layer has an activation function (here ReLU) which boosts the features that needs to be taken to the next layer and removes the less important ones.
Towards the end of the network, we flatten the output and use a softmax activation which provides a probability like distribution.
 
*In this particular problem, the transition block is added a little earlier in the network as most of the images have the information within the center. The background doesn't have any information. So it's safe to use maxPool a little early.*

> No of parameters: 1,595,316  
> Best Accuracy: 99.93% (train)


## Target 2: Reduce Parameters
The goal is to achieve 99.40% accuracy in less than 15k parameters. So in this section, the major focus is on reducing the number of kernels/filers in each convolution so that total parameters are less than 15k with slight compromise with accuracy (which can later be improved by further optimization).
Also, the first two changes here deal with reducing the number of filters in 3x3 and 1x1 convolution.
Even though drastically reducing the parameters in all the layers, the total parameters still cross 15k. So another change is done within the architecture by adding a partial transition block exactly in middle of first transition block and the output layer. This partial transition involves the use of reduction of parameters using 1x1 convolution, but no maxPool. 
This makes sure that only the important information is carried further in the network and also avoids loss of information that could have happened due to maxPool.

> No of parameters: 12,490  
> Best Accuracy: 99.71% (train)

## Target 3: Optimization
Since the reduction in number of kernels is drastic, we need some optimizations to make sure our model doesn't have to compromise with accuracy. 
The first one is Batch normalization. Whenever new kernels are being formed, we add a batch normalization layer to make sure that the weights in each kernel are normalized and all the important features are carried forward to the succeeding layers.
Along with that, a custom Learning Rate that works best for the network is also selected (based on experiments).
The accuracy vs epoch is not strictly increasing. Accuracy keeps on oscillating, hence one cannot rely on the last epoch to give the best result. So a validation checkpoint is added that calculates accuracy on validation data at the end of each epoch.

> No of parameters: 12,874  
> Best Accuracy: 99.28% (val), 99.74% (train)

## Target 4: OverFitting and further optimization
On comparing the training accuracy with validation accuracy, you realize that the gap between them is expanding. So to reduce that, you add a dropout after every 3x3 convolution. Dropout basically skips some connections randomly and forces the network to learn from other parameters. This helps in reduction of overfitting.
Also, once the accuracy saturates, the learning rate needs to decreased so that the loss can further be reduced and eventually leading to increase in accuracy.
A larger Batch is also selected which helps in faster training of the model. Note: the batch size should not be extremely high. This can lead to the model getting stuck in local minima.

> No of parameters: 12,874  
> Best Accuracy: 99.45% (val), 99.45% (train), 32nd 
> epoch
