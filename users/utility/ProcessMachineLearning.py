import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from django.conf import settings
from sklearn import preprocessing
path = settings.MEDIA_ROOT + "\\" + "Blood Pressure.csv"
df=pd.read_csv(path)
#label changing by using pandas
# df["Gender"].replace({'M':1,'F':0},inplace=True)
# df["Stress Level"].replace({'HIGH':2,'LOW':0,'MEDIUM':1},inplace=True)
# df['Hypertension'].replace({'NO':0,'YES':1},inplace=True)
# df['Smoking'].replace({'YES':1,'NO':0},inplace=True)
# df['Daily alcohol'].replace({'NO':0},inplace=True)
label_encoder = preprocessing.LabelEncoder()
df['Gender']= label_encoder.fit_transform(df['Gender'])
df['Stress Level']= label_encoder.fit_transform(df['Stress Level'])
df['Hypertension']= label_encoder.fit_transform(df['Hypertension'])
df['Smoking']= label_encoder.fit_transform(df['Smoking'])
df['Daily alcohol']= label_encoder.fit_transform(df['Daily alcohol'])


#Showing the statistical measures of the data
df.describe()

df.isnull().sum()
import seaborn as sns
sns.set()

df['Gender'].value_counts()

#Making a count plot for gender column
# sns.countplot('Gender', data = df)


#Making a count plot for Daily salt column
# sns.countplot('Daily salt', data = df)

#Making a count plot for Stress Level column
# sns.countplot('Stress Level', data = df)


#Making a count plot for Smoking column
# sns.countplot('Smoking', data = df)

#Making a count plot for smoking_status column
# sns.countplot('BMI', data = df)

#Making a count plot for hypertension column
# sns.countplot('Hypertension', data = df)

#Showing heart disease and no heart disease genderwise
# sns.countplot('Gender', hue ='Hypertension', data = df)

#Seperating the data and labels
X = df.iloc[:,:-1]
y = df['Hypertension']

#Data standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
standard = scaler.transform(X)
X = standard

#Train,Test,Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)

from sklearn import svm
model = svm.SVC(kernel = 'linear')

#Training the SVM Model
model.fit(X_train, y_train)

#Finding the accuracy score on train dataset
from sklearn.metrics import accuracy_score
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, y_train)

train_data_accuracy#Finding the accuracy score on test dataset

#Finding the accuracy score on test dataset
from sklearn.metrics import accuracy_score
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

test_data_accuracy

#MODEL EVALUATION FOR HYPERTENSION PREDICTION

#Predicting System
data = (51,1,24.0,0,5000,250,3.00,2)
#Converting to numpy array
data_array = np.asarray(data)

#Reshaping the array
data_reshape = data_array.reshape(1, -1)

#Standardizing the data
data_standard = scaler.transform(data_reshape)

prediction = model.predict(data_standard)

if(prediction[0] == 0):
    print('No Hypertension')
else:
    print('Hypertension')

#SAVING THE TRAINED MODEL FOR HYPERTENSION PREDICTION
import pickle
filename = 'hypertension_model.sav'
pickle.dump(model,open(filename,'wb'))

#loading the saved model
loaded_model = pickle.load(open('hypertension_model.sav','rb'))



def classification(test_data):
    
    model = svm.SVC(kernel = 'linear')

    #Training the SVM Model
    model.fit(X_train, y_train)
    X_test_prediction = model.predict([test_data])
    
    print(X_test_prediction)
    
    return X_test_prediction
    
    





