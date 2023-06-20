# Neuro Evolution Search applied to CNNs for Image classification

### Objective
The objective is to utilize a Genetic Algorithm to identify the optimal architecture for a 
Convolutional Neural Network (CNN) specifically tailored for image classification. This repository presents a 
specific application of the proposed methodology for image classification, focusing on the well-known 
“Digits” toy dataset provided by sklearn.

### Methodology
A two-fold process is followed: finding the optimal architecture through a genetic 
algorithm and subsequently training the identified architecture effectively.

For the first step the fitness of a CNN is defined as follows:

$$Fitness = ce + \lambda \frac{N_{p}}{N_{ma}}$$

For the second step the CNN were trained by minimizeing the Negative Log Likelihood Loss:

$$NLL = -\frac{1}{N} \sum_ {n=1} ^ {N} (log(p(y_{n}|x_{n})$$


The CNN structure follows the sequence of: 

    𝐶𝑜𝑛𝑣2𝑑 → 𝑓(. ) → 𝑃𝑜𝑜𝑙𝑖𝑛𝑔 → 𝐹𝑙𝑎𝑡𝑡𝑒𝑛 → 𝐿𝑖𝑛𝑒𝑎𝑟1 → 𝑓(. ) → 𝐿𝑖𝑛𝑒𝑎𝑟2 → 𝑆𝑜𝑓𝑡𝑚𝑎𝑥
    
Different options are available for each building block, resulting in a total of 1200 possible configurations. For 
the Conv2d layer, we explore three options for the number of filters (8, 16, and 32) and two kernel sizes ((3,3) 
and (5,5)) with corresponding stride and padding values (1 and 2). The f(.) activation function offers a choice 
among ReLU, Sigmoid, Tanh, Softplus, and ELU, each providing distinct nonlinear transformations. In the 
Pooling layer, we have two options for the pooling operation (2x2 or Identity) and two pooling methods 
(average and maximum pooling). The Linear 1 layer allows customization of the number of neurons, ranging 
from 10 to 100 in increments of 10. Finally, the network concludes with a second Linear layer and a Softmax 
activation function for multi-class classification.

### Results

The following figure displays the fitness of the best individual of each generation:

| Generations:  | Best fitness: |
| ------------- | ------------- |
| 0 | 0.09 |
|5 | 0.06 |
|10 | 0.05 |
|15 | 0.05 |
|20 | 0.05 |
|25 | 0.05 |
|30 | 0.05 |
|35 | 0.05 |

Following is the plot of training and testing of the best model overall:

![image](https://github.com/LuigTi/CNN_NAS/assets/91637040/82d12252-181d-400f-82fc-9ec68bf2602a)


