'''
    Handles loading data from titanic
'''
import csv

'''
    try to parse a string into a float
    if we can't return the provided error val
'''
def tryParseFloat(item, errorval=-1):
    try:
        val = float(item)
        return val
    except ValueError:
        return errorval

'''
    parses a cabin item into a numeric value
'''
def parseCabin(item, errorval=-1):
    if item == "":
        #TODO: put average here, actually perhaps median would be better
        return errorval
    else:
        try:
            #it is possible for someone to have more cabins
            cabins = item.split(" ")
            #for each cabin
            for cabin in cabins:
                #get the foor character [A-G]
                floor = cabin[0]
                #cast the floor to a number A:1 B:2 C:3
                floornum = ord(floor) - 65 + 1
                #start roomnum as just roomnum without letter
                roomnum = int(cabin[1:])
                #now A should be 100, b should be 1000, c should be 10000
                #so for A we have 10^(1+1)=100
                #and for b we have 10^(2+1)=1000
                roomnum += pow(10, (floornum + 1))
                return roomnum
        except:
            return errorval

'''
    print the normalizing constants
'''
def normalize():
    #will hold the max values
    pclass = 0
    age = 0
    sibs = 0
    heirs = 0
    fare = 0
    cabin = 0

    first = True
    #open csv and reader
    with open("data/train.csv") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        #for each row
        for row in reader:
            if first:
                first = False
                continue
            #choose max of currentMax and currentReading
            pclass = max(tryParseFloat(row[2]), pclass)
            age = max(tryParseFloat(row[5]), age)
            sibs = max(tryParseFloat(row[6]), sibs)
            heirs = max(tryParseFloat(row[7]), heirs)
            fare = max(tryParseFloat(row[9]),fare)
            cabin = max(parseCabin(row[10]), cabin)
    
    #print consts
    print(pclass, age, sibs, heirs, fare, cabin, sep="\n")

'''
    finds the means of each parameter
'''
def findMeans():
    #each parameter and count of occurances
    pclass = 0
    pclasscount = 0

    age = 0
    agecount = 0

    sibs = 0
    sibscount = 0

    heirs = 0
    heirscount = 0

    fare = 0
    farecount = 0

    cabin = 0
    cabinCount = 0

    qcount = 0
    scount = 0
    ccount = 0

    #open csv file and reader
    with open("data/train.csv") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for row in reader:
            #count the parameter if not parsed as -1
            if tryParseFloat(row[2]) != -1:
                pclass += tryParseFloat(row[2])
                pclasscount += 1
            if tryParseFloat(row[5]) != -1:
                age += tryParseFloat(row[5])
                agecount += 1
            if tryParseFloat(row[6]) != -1:
                sibs += tryParseFloat(row[6])
                sibscount += 1
            if tryParseFloat(row[7]) != -1:
                heirs += tryParseFloat(row[7])
                heirscount += 1
            if tryParseFloat(row[9]) != -1:
                fare += tryParseFloat(row[9])
                farecount += 1
            if parseCabin(row[10]) != -1:
                cabin += parseCabin(row[10])
                cabinCount += 1
            if row[11] == "C":
                ccount += 1
            elif row[11] == "Q":
                qcount += 1
            elif row[11] == "S":
                scount += 1

    #print averages
    print("Pclass", pclass / pclasscount)
    print("age", age / agecount)
    print("sibs", sibs / sibscount)
    print("heirs", heirs /heirscount)
    print("fare", fare / farecount)
    print("cabin", cabin / cabinCount)
    print("embark", ccount, qcount, scount)

'''
    gets the size of a set
'''
def getDataSize(filename):
    linecount = 0
    with open(filename) as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for row in reader:
            linecount += 1
    
    return linecount - 1


'''
    loads training data
'''
def loadTrainData(includeEmbark=True):
    #parameters and result
    X = []
    Y = []
    #used for splitting data and skipping header
    count = 0
    #open training data file
    with open("data/train.csv") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for row in reader:
            #skip header
            if count == 0:
                count += 1
                continue
            # add result
            Y.append(int(row[1]))
            #holds parameters
            newList = []
            #parse socio class else add average
            if tryParseFloat(row[2]) != -1:
                newList.append(tryParseFloat(row[2]) / 3)
            else:
                newList.append(2.308641975308642 / 3)
            
            #parse male or female
            if row[4] == "":
                newList.append(.5)
            else:
                newList.append(-1 if row[4] == "male" else 1)
            
            #parse age else add average
            if tryParseFloat(row[5]) != -1:
                newList.append(tryParseFloat(row[5]) / 80)
            else:
                newList.append(29.69911764705882 / 80)

            #parse siblings else add average
            if tryParseFloat(row[6]) != -1:
                newList.append(tryParseFloat(row[6]) / 8)
            else:
                newList.append(0.5230078563411896 / 8)
            
            #parse parents and children else add average
            if tryParseFloat(row[7]) != -1:
                newList.append(tryParseFloat(row[7]) / 6)
            else:
                newList.append(0.38159371492704824 / 6)
            
            #parse ticket price else add average
            if tryParseFloat(row[9]) != -1:
                newList.append(tryParseFloat(row[9]) / 512.3292)
            else:
                newList.append(32.2042079685746 / 512.3292)
            
            if parseCabin(row[10]) != -1:
                newList.append(parseCabin(row[10]) / 100000006)
            else:
                newList.append(2681879.1836734693 / 100000006)
            
            if includeEmbark:
                #parse embarkement
                if row[11] != "":
                    if row[11] == "C":
                        newList.append(1)
                        newList.append(-1)
                        newList.append(-1)
                    elif row[11] == "Q":
                        newList.append(-1)
                        newList.append(1)
                        newList.append(-1)
                    elif row[11] == "S":
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                    else:
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                else:
                    newList.append(-1)
                    newList.append(-1)
                    newList.append(1)

            X.append(newList)
            count += 1
    return X,Y

'''
    loads all data and evenly splits into test, train, and validation if asked
'''
def loadData(validationSet=False, includeEmbark=True):
    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []
    X_valid = []
    Y_valid = []
    size = getDataSize("data/train.csv")
    if validationSet:
        size /= 3
    else:
        size /= 2
    
    count = 0
    with open("data/train.csv") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for row in reader:
            X, Y = None, None
            #skip header
            if count == 0:
                count += 1
                continue
            #if first section
            if count <= size:
                X = X_train
                Y = Y_train
            #elif second section
            elif count <= size * 2:
                X = X_test
                Y = Y_test
            #elif third section
            else:
                X = X_valid
                Y = Y_valid

            #add result
            Y.append(int(row[1]))
            #holds parameters
            newList = []
            #parse socio class else add average
            if tryParseFloat(row[2]) != -1:
                newList.append(tryParseFloat(row[2]) / 3)
            else:
                newList.append(2.308641975308642 / 3)
            
            #parse male or female
            if row[4] == "":
                newList.append(.5)
            else:
                newList.append(-1 if row[4] == "male" else 1)
            
            #parse age else add average
            if tryParseFloat(row[5]) != -1:
                newList.append(tryParseFloat(row[5]) / 80)
            else:
                newList.append(29.69911764705882 / 80)

            #parse siblings else add average
            if tryParseFloat(row[6]) != -1:
                newList.append(tryParseFloat(row[6]) / 8)
            else:
                newList.append(0.5230078563411896 / 8)
            
            #parse parents and children else add average
            if tryParseFloat(row[7]) != -1:
                newList.append(tryParseFloat(row[7]) / 6)
            else:
                newList.append(0.38159371492704824 / 6)
            
            #parse ticket price else add average
            if tryParseFloat(row[9]) != -1:
                newList.append(tryParseFloat(row[9]) / 512.3292)
            else:
                newList.append(32.2042079685746 / 512.3292)
            
            if parseCabin(row[10]) != -1:
                newList.append(parseCabin(row[10]) / 100000006)
            else:
                newList.append(2681879.1836734693 / 100000006)
            
            #parse embarkement
            if includeEmbark:
                if row[11] != "":
                    if row[11] == "C":
                        newList.append(1)
                        newList.append(-1)
                        newList.append(-1)
                    elif row[11] == "Q":
                        newList.append(-1)
                        newList.append(1)
                        newList.append(-1)
                    elif row[11] == "S":
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                    else:
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                else:
                    newList.append(-1)
                    newList.append(-1)
                    newList.append(1)
 
            #parse cabin
            X.append(newList)
            count += 1
    if validationSet:
        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid
    else:
        return X_train, Y_train, X_test, Y_test


'''
    loads test data that doesn't have known values for predictions
'''
def loadActualTestData(includeEmbark=True):
    X = []
    passengerids = []
    count = 0
    with open("data/test.csv") as f:
        reader = csv.reader(f, delimiter=",", quotechar="\"")
        for row in reader:
            if count == 0:
                count += 1
                continue
            
            passengerids.append(row[0])
            #holds parameters
            newList = []
            #parse socio class else add average
            if tryParseFloat(row[1]) != -1:
                newList.append(tryParseFloat(row[1]) / 3)
            else:
                newList.append(2.308641975308642)
            
            #parse male or female
            if row[3] == "":
                newList.append(.5)
            else:
                newList.append(-1 if row[3] == "male" else 1)
            
            #parse age else add average
            if tryParseFloat(row[4]) != -1:
                newList.append(tryParseFloat(row[4]) / 80)
            else:
                newList.append(29.69911764705882)

            #parse siblings else add average
            if tryParseFloat(row[5]) != -1:
                newList.append(tryParseFloat(row[5]) / 8)
            else:
                newList.append(0.5230078563411896)
            
            #parse parents and children else add average
            if tryParseFloat(row[6]) != -1:
                newList.append(tryParseFloat(row[6]) / 6)
            else:
                newList.append(0.38159371492704824)
            
            #parse ticket price else add average
            if tryParseFloat(row[8]) != -1:
                newList.append(tryParseFloat(row[8]) / 512.3292)
            else:
                newList.append(32.2042079685746)
            
            if parseCabin(row[9]) != -1:
                newList.append(parseCabin(row[9]) / 100000006)
            else:
                newList.append(2681879.1836734693 / 100000006)
            
            if includeEmbark:
                #parse embarkement
                if row[10] != "":
                    if row[10] == "C":
                        newList.append(1)
                        newList.append(-1)
                        newList.append(-1)
                    elif row[10] == "Q":
                        newList.append(-1)
                        newList.append(1)
                        newList.append(-1)
                    elif row[10] == "S":
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                    else:
                        newList.append(-1)
                        newList.append(-1)
                        newList.append(1)
                else:
                    newList.append(-1)
                    newList.append(-1)
                    newList.append(1)

            X.append(newList)
            count += 1
    return passengerids, X