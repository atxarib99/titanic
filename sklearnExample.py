'''
    This file serves the purpose of setting up an environment for sklearn to attempt the dataset. I thought it would be a good idea to compare my models with one of the industry standard models
'''

#import model
from data import TitanicParser as model
#import perceptron learner from sklearn
from sklearn.linear_model import Perceptron

#get train and test data from model
X,Y, testX, testY = model.loadData(includeEmbark=False)

#get perceptron object from sklearn
trainer = Perceptron()

#fit to the training data
trainer.fit(X,Y)

#score train data
trainAcc = trainer.score(X,Y)

#score test data
acc = trainer.score(testX, testY)

#print accuracys
print("Train Accuracy", trainAcc)
print("Test Accuracy", acc)

#guess data to predict on
passenger, guessX = model.loadActualTestData(includeEmbark=False)

#output file
output = open("sklearnout.csv", "w+")
#print header
output.write("PassengerId,Survived\n")

#get labels
labels = trainer.predict(guessX)

#for each label, output to file
for i in range(len(labels)):
    label = labels[i]
    output.write(str(passenger[i]))
    output.write(",")
    output.write(str(label))
    output.write("\n")

#close output file
output.close()