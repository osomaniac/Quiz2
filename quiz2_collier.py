import pandas as pd

## read files to csvs
classes = pd.read_csv('animal_classes.csv')
train = pd.read_csv('animals_train.csv')
test = pd.read_csv('animals_test.csv')

## viewing data
#print("CLASSES:\n", classes.head(5))
#print("TRAIN:\n", train.head(5))
#print("TEST:\n", test.head(5))

## making adjustments to the dfs
animal_names = test['animal_name'].to_list()
animal_types = classes['Class_Type'].to_list()
test_new = test.drop('animal_name', axis=1)
y_train = train['class_number']
x_train = train.drop('class_number', axis=1)
x_test = test_new

## train model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X=x_train, y=y_train)

## test model with the test file, leaving out expected becuase we are determining that
predicted = knn.predict(X=x_test)

## open up a csv file to write to
loc = 0
outfile = open('mypredictions.csv', 'w')
header = "animal_name,prediction"
outfile.write(header + '\n')

## for loop to write to the file
for p in predicted:
    line = f"{animal_names[loc]},{animal_types[int(p)-1]}\n"
    # print(line)
    outfile.write(line)
    loc += 1