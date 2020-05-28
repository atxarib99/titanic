'''
    This class sets up the titanic problem to be solved using SVM, Support Vector Machines.
    Accuracy: 0.77511
'''

#import the SVM class
import SVM.SVM as svm
#import the model
import data.TitanicParser as model

#build the SVM trainer, use 50% of data to train, 30% to validate, and 20% to test
trainer = svm.SVM(partitioner=[.5,.3,.2])

#load training data from the model
X,Y = model.loadTrainData(includeEmbark=False)

#remap labels to -1,1
for i in range(len(Y)):
    Y[i] = (Y[i] - .5) * 2

#load the data into the trainer
trainer.loadTrainingData(X,Y)

#start training
trainer.train()

#load the data we want to classify
passenger, testX = model.loadActualTestData(includeEmbark=False)

#open output file
output = open("svmout.csv", "w+")
#print header
output.write("PassengerId,Survived\n")

#predict the labels for our test set
guesses = trainer.predict(testX, negativeValue=0)

#write the guesses out to file
for i in range(len(guesses)):
    output.write(str(passenger[i]))
    output.write(",")
    output.write(str(guesses[i]))
    output.write("\n")

#close output
output.close()