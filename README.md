# titanic

This project shares my experience and learning while competiting in Kaggle's titanic machine learning competitions.

## Goals

Utilize several different machine learning methods to gain experience as well as compare their strengths and weaknesses.

Also score the highest accuracy possible. It is a competition after all.

## Data

The data provided includes:

| Name | Value Type |  Value Range | Description |
| ---  | --- | --- | ----- |
| PassengerID | # | [0-1309] | The ID of the passenger, this exists to provide a unique ID for each passenger and is not part of the actual data. |
| Survived | # | [0,1] | This is the label for the dataset. 1 means they survived, 0 means that they unfortunately did not. |
| Pclass | # | [1,2,3] | This represents the class of the ticker. 1 being 1st class, 2 being 2nd class, and 3 being 3rd class. |
| Name | String | LastName, Title FirstName (Any Middle Names or suffixes) | Simply the name of the person. |
| Sex | String | [male, female] | The sex of the person. |
| Age | # | [0-80] | The age of the person. |
| SibSp | # | [0-8] | The number of siblings aboard. |
| Parch | # | [0-6] | The number of parents and children aboard. |
| Ticket | # | [000000-999999] | The person's ticket number. |
| Fare | # | [0-512] | The cost of the person's ticket. |
| Cabin | char+num as list | [A-G][00-99] | The letter represents which portion of the boat the person remained in and that portions room number. A person can have multiple cabins. [Deckplans](https://www.encyclopedia-titanica.org/titanic-deckplans/) |
| Embarked | char | [C,Q,S] | Where the person boarded the boat. C = Cherbourg, Q = Queenstown, S = Southampton |

## Model

Some portions of the model are simple, while others require a bit of feature engineering to be able to model mathematically.

Firstly, we try to quantify as many parameters as we desire. For example, age is already quanitified, while cabin number and embarkement location are not. Here we also choose which parameters to use. For example, passengerID is meaningless in the case of survivial, and name will need heavy feature engineering to make useful, so we discard them for now.

For Cabin, we quantify this by using the equation below. This is because we can have orders of 10 for each floor. Since we know that members of the bottom floor are less likely to survive when the ship starts to flood. This should allow the algorithm to assign weights to this.

` roomnum + pow(10, ((ord(floor) - 65 + 1) + 1)) `

For Embarked we use a one hot input technique. So instead of having 1 input representing where they boarded, we have three true false inputs on if they embarked at a certain location where only one of them are true at a time. For example if they boarded at Southampton then we have inputs 0, 0, 1.

After we have quanitfied each input parameter, we normalize the inputs so that every input is from range [0-1], this will allow us to learn weights easier. This is because age only ranges from  0 though 80 while the fare can range from 0 through 512. So if we assign weights to these as they are, then fare is already weighted higher because it has a higher range. For example, for weight .75, the age can only be weighted at maximum 60 while fare can be weighted to 384. The learner must then compensate for this difference.

## The Results
I used three different methods to predict labels.

| Name | Accuracy |
| ---- | ---- |
| Neural Network | 0.78648 |
| SVM with Slack | 0.77511 |
| kNN (k-Nearest Neighbor) | 0.79904 |

## Neural Networks

I helped build a Feed Forward Neural Network with some friends. Its base code can be found at: [Neural Network](https://github.com/aviguptatx/SecretHitlerAI/blob/master/NeuralNet.py)

Some positives about neural networks is that they can model complex models better than simpler methods such as SVMs or kNNs. There are several libraries available such as PyTorch or TensorFlow where you can setup a neural network that will run much more efficiently, however we decided to make our own for sake of learning. We used a neural network structure of a 10 node input layer, 5 node hidden layer, and a 1 node output layer. Neural Networks are good for solving a variety of problems but it proved to be suboptimal for our case.

## SVM with Slack

[Dry SVM](https://github.com/atxarib99/SVMWithSlack)

SVMs are a relatively common methodology for binary classification. Here I used the cvxopt solvers to find the support vectors. I divided the data into a training set, validation set to learn hyperparameters, and a test set to evaluate performance. I attempted to find the best c between the set [.001, .01, .1, 1, 10, 1e+2, 1e+3, 1e+4, 1e+5] using the validation set. An issue I encountered is that the SVM had the same accuracy for different values of c. Intuitively, it makes sense that we should maximize the margins from the support vectors, so between the c values that yielded the best accuracy we chose the highest value of those.

With the number of parameters in our dataset and due to its simplicity, an SVM is both easy to setup and can quickly learn at a decent rate.

## kNN (k-Nearest Neighbor)

[Dry kNN](https://github.com/atxarib99/kNN)

This method is relatively very simple. The idea with kNNs is to find training examples that are the most similar to the example we are trying to predict on, and use its labels to predict our own. An example that can be easily graphed can be found at the Dry kNN link above, which shows a problem with 2 dimensions that can be graphed. The main logic for making a prediction is very simple. All that needs to be done is calculate a distance between every training example, choose the closest k, and average their labels. An advantage with kNN is that it is hard to overfit the problem to the training set. As long as the training set reflects the true data, the predictions should be accuracte.

kNN works well with the titanic set since it is relatively simple. It can easily find training examples close to the test example and make a decent prediction. For more complicated problems, all the training examples will be very far from each other so when a test case tries to make a prediction its "closest" k examples are actually very far and not similar to itself at all, thus making an inaccurate guess.

## Time and space required to learn

Learning time is important in some cases, here our project is not time sensitive and we can spend more time learning weights for a better accuracy, but let's discuss time to learn anyways so we can compare the methods to each other.

For the neural networks runtime, I have already mentioned how it is not the most efficient neural network solver out there, so it's runtime should be taken with quite a few grains of salt. Running 100 epochs on the above mentioned network structure takes roughly 6 seconds for 100 epochs. The learning stops changing at rougly 75 epochs so optimally this would take 4 seconds. It should be kept in mind that it is only learning weights here.

The SVM solution, learning both weights as well as hyperparameters takes roughly 3 seconds. This is almost twice as fast as learning just the weights for our neural network at 100 epochs, although it has slightly less accuracy. In this case, the neural network is more prefereable since our problem is not time sensitive.

Lastly the kNN algorithm takes 6 seconds to run. Here it is not learning weights just setting up the prediction environment, but it is learning the k hyperparameter within this time. It also yields the best accuracy. However, it should be noted that this is the least memory efficient method since it needs to keep all the training examples in memory to compare to. To make a prediction it has a time of O(nm), where n is number of training examples, and m is number of parameters.


## Author

Arib Dhuka