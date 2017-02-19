#Arterys Challenge
##Model Overview:
My model is adapted from LeNet5. 

*Note: Due to some issues with my virtualenv I did all my testing on a Theano backend. However, it should still work with TensorFlow backend as well.*


##Part 1: Analysis


**Q: What is your test set error rate?**

A: Test error rate: 1.62%

**Q: What is the test set error rate for each class? Are some classes more challenging than others to distinguish from each other? Why?**

A: [ 99.75  99.76  99.7   99.61  99.8   99.7   99.8   99.52  99.61  99.49]
With numbers like these, it is hard to say if any class was challenging at all. It is possible that the a more complicated task would give us a better opportunity to analyze. 

**Q: Based only on information gathered in the first epoch of training, do you think that the model would benefit from more training time? Why?**

A: I would run it for at least a couple more epochs to see if I can obtain any improvement in validation accuracy . Its impossible to judge from one epoch alone. Generally, more training will surely be useful. However, for a dataset like MNIST, we can say it is "good enough" with about 99% accuracy unless we are trying to break some record. 

**Q: Besides training for a longer time, what would you do to improve accuracy?
Increasing width of each layer, adding more convolution layers and fully connected layers, try different optimizers (adam gave me better accuracy than adagrad) etc.**

##Part 2: Analysis on noisy data


**Q: What are the implications of the dependence of accuracy on noise if you were to deploy a production classifier? How much noise do you think a production classifier could tolerate?**

A: 

**Q: Do you think that Gaussian noise is an appropriate model for real-world noise if the characters were acquired by standard digital photography? If so, in what situations? How would you compensate for it?**

**Q: Is the accuracy of certain classes affected more by image noise than others? Why?**


##Part 3: Analysis on noisy labels


**Q: How important are accurate training labels to classifier accuracy?**

**Q: How would you compensate for label noise? Assume you have a large budget available but you want to use it as efficiently as possible.**


**Q: How would you quantify the amount of label noise if you had a noisy data set?**

**Q: If your real-world data had both image noise and label noise, which would you be more concerned about? Which is easier to compensate for?**
