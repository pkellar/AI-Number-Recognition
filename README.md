# AI-Number-Recognition
Using MNIST data and creating a Machine Learning Model to identify hand written numbers

Patrick Kellar
AI Project
<br>
* Problem statement and proposed solution:
  * The goal was to use the MNIST data, a collection of handwritten, single digit numbers, and implement a classifier. It is proposed that this can be done utilizing:
    * NumPy - python library
    * Matplotlib - plotting library
    * Keras - library that provides a python interface for artificial neural networks
  * With these libraries, I can make a recurrent neural network that can be used to make predictions of numbers. Specifically, I will be making a sequential model utilizing keras. This model will have an input layer, four hidden layers, and an output layer.
* What model did I implement, what parameters and hyper parameters did I use. What are the expected shortcomings of this model?
  * I implemented a recurrent neural network with an input layer, hidden layers, and an output layer.
  * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/627ec338-f12f-431d-9781-b68eb13648d6)
  * The input layer is taking in an input of a 28x28 matrix since that is the size of the images being used.
  * For the hidden layers, I utilized the rectified linear unit activation function (relu). This function returns the value provided if greater than 0, or if the input is less than 0, it will return 0. This is referred to as a piecewise linear function and can be represented as max{0,z}.
    * When it directly returns the values that are greater than zero, this function is behaving like a linear activation function. This means there is no transformation applied which makes the model easier to train.
    * When it returns 0 for a negative value, it is behaving like a nonlinear function which allows the model to learn more complex relationships.
  * For the output later, I utilized the softmax function. This is used because it rescales the output to a probability distribution. This means that the output values are non negative and will add to one. Then, whichever class has the largest of these values, will be the prediction of the model.
    * The input values are the log-odds of the resulting probability. This input is possible because of the sparse categorical cross entropy loss function which will be discussed later.
    * Also, I am able to know which value is the largest by utilizing the argmax() function with returns whichever label scored the highest from the output.
  * This is the parameter count for each of these layers.
    * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/61241d8c-7a23-4764-97b7-26b37fcb6c3e)
  * I then compiled the model with this:
    * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/19b390cf-124e-48ad-8524-4440ecf0b2e8)
  * The model is using the optimizer Adam. This optimizer utilizes ideas from stochastic gradient descent, but it instead maintains a per-parameter learning rate rather than a single learning rate for all the weights. It finds the moving average of the gradient and the squared gradient. This is controlled by parameters beta1 and beta2 that monitor the decay rates of the moving averages.
    * For my model, I left these parameters at their default values.
      *  Learning_rate = 0.001
      * beta1 = 0.9
      * beta2 = 0.999
      * Keras information about Adam: https://keras.io/api/optimizers/adam/#:~:text=LearningRateSchedu le%20%2C%20or%20a%20callable%20that,Defaults%20to%200. 001.
    * In addition, this optimizer was shown to keep training costs low when applied to the MNIST data as seen in this 2015 paper: https://arxiv.org/pdf/1412.6980.pdf
      * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/9d4e7b76-5fdb-4b1a-97e5-ee038b464e04)
    * The model is also utilizing the loss function of Sparse Categorical Cross Entropy
      * I chose this loss function because there are more than two class labels in this data and the labels are not in the one-hot representation. This loss function allows me to have my true labels in the form [1], [2], [3] rather than the form [1,0,0], [0,1,0], [0,0,1].
      * The categorical cross entropy is calculated with this equation:
        *![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/85f7ec42-2f81-4e9e-bf59-6dabc689fb39) with ŷ being the predicted output and y being the true output
      * This will learn to give a high probability to the correct digit and a low probability to the others.
* How did I train, test and validate my model? I.e. What did I do to find the optimal solution for my generalized error?
  * The MNIST data set already was split up into testing and training sets. However, I also needed a validation set, so I moved 6k training instances (of the total 60k) into a validation set.
    * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/50fe51bb-a6de-4693-bdb0-476a981c4984)
    * Since the training set is already randomized, I do not have to worry about all of the validation set being primarily one digit.
  * I then fit my data to my model by passing in the training and the validation sets.
    * The number of epochs is how many times this will iterate over the data. I set it to 60 here.
    * The batch size is how many training images it will use in each epoch. Here, I set it to 50,000.
      * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/151bfa0e-e6a8-4129-855a-897be299e6cc)
  * After the model is fit, I then evaluate it on the test set. This then provides me with the loss and accuracy of the test set of 7.80% and 97.58% respectively.
    * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/379e628a-7c8d-4fb0-bf88-7a76b90877cf)
* Loss and Accuracy Results
  * ![image](https://github.com/pkellar/AI-Number-Recognition/assets/106937373/cfd7468f-930d-4705-adfa-a8f75dbe16d6)
* From the prediction, what digits did I model struggle with?
  * I was able to get predictions on the test image by passing my test images into the model and then comparing the predicted labels to the actual labels with this.
  * After having my model predict the digits of the test images, these were the results:
    * The model struggled most with predicting the 8 digit with 41 instances of it being misclassified.
    * It also struggled with the 5 digit with 36 instances being misclassified.
    * It proved very good at being able to classify the 0 and 1 digits. I believe this is because they have more distinguishing and simple shapes compared to the other digits.
