from django.shortcuts import render
from django.contrib import messages
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from .forms import UserRegistrationForm


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    return render(request, 'users/UserHome.html', {})
    # if request.method == "POST":
    #     loginid = request.POST.get('loginid')
    #     pswd = request.POST.get('pswd')
    #     print("Login ID = ", loginid, ' Password = ', pswd)
    #     try:
    #         check = UserRegistrationModel.objects.get(
    #             loginid=loginid, password=pswd)

    #         status = check.status
    #         print('Status is = ', status)
    #         if status == "activated":
    #             request.session['id'] = check.id
    #             request.session['loggeduser'] = check.name
    #             request.session['loginid'] = loginid
    #             request.session['email'] = check.email
    #             print("User id At", check.id, status)
    #             return render(request, 'users/UserHome.html', {})
    #         else:
    #             messages.success(request, 'Your Account is not yet activated')
    #             return render(request, 'UserLogin.html')
    #     except Exception as e:
    #         print('Exception is ', str(e))
    #         pass
    #     messages.success(request, 'Invalid Login id and password')
    # return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})



def ML(request):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_style('darkgrid')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    #from sklearn.naive_bayes import GaussianNB
    #from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from django.conf import settings
    import os
    from sklearn.linear_model import LogisticRegression
    path = os.path.join(settings.MEDIA_ROOT + "\\" + "Blood Pressure.csv")
    df = pd.read_csv(path)
    
    print(df.shape)
    df.head()
    df.info()
    df.isna().mean()

    df["Gender"].replace({'M':1,'F':0},inplace=True)
    df["Stress Level"].replace({'HIGH':2,'LOW':0,'MEDIUM':1},inplace=True)
    df['Hypertension'].replace({'NO':0,'YES':1},inplace=True)
    df['Smoking'].replace({'YES':1,'NO':0},inplace=True)
    df['Daily alcohol'].replace({'NO':0},inplace=True)

    import seaborn as sns
    sns.set()

    # df['gender'].value_counts()

    #Making a count plot for gender column
    sns.countplot(df)


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
    sns.countplot(df)




    #Seperating the data and labels
    X = df.iloc[:,:-1]
    y = df['Hypertension']

    #Data standardisation
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    standard = scaler.transform(X)
    X = standard

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    # X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy1 = accuracy_score(y_test, y_pred) * 100
    print('Accuracy:', accuracy1)
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    from sklearn.metrics import precision_score

    precision1 = precision_score(y_test, y_pred) * 100
    print('Precision Score:', precision1)
    from sklearn.metrics import recall_score
    recall1 = recall_score(y_test, y_pred) * 100
    print('recall_score:',recall1)
    from sklearn.metrics import f1_score
    f1score1 = f1_score(y_test, y_pred) * 100
    print('f1score:',f1score1)
    # roc1 = roc_auc_score(y_test, y_pred) * 100

    model2 = SVC()
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy2 = accuracy_score(y_test, y_pred2) * 100
    print('Accuracy2:', accuracy2)
    precision2 = precision_score(y_test, y_pred2) * 100
    print('precision2:',precision2)
    recall2 = recall_score(y_test, y_pred2) * 100
    print('recall2:',recall2)
    f1score2 = f1_score(y_test, y_pred2) * 100
    print('f1score2:',f1score2)
    # roc2 = roc_auc_score(y_test, y_pred2) * 100

    # model3 = GaussianNB()
    # model3.fit(X_train, y_train)
    # y_pred3 = model3.predict(X_test)
    # accuracy3 = accuracy_score(y_test, y_pred3) * 100
    # precision3 = precision_score(y_test, y_pred3) * 100
    # recall3 = recall_score(y_test, y_pred3) * 100
    # f1score3 = f1_score(y_test, y_pred3) * 100
    # roc3 = roc_auc_score(y_test, y_pred3) * 100

    model4 = LogisticRegression()
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    accuracy4 = accuracy_score(y_test, y_pred4) * 100
    print('accuracy:',accuracy4)
    precision4 = precision_score(y_test, y_pred4) * 100
    print('precision:',precision4)
    recall4 = recall_score(y_test, y_pred4) * 100
    print('recall:',recall4)
    f1score4 = f1_score(y_test, y_pred4) * 100
    print('f1score:',f1score4)
    # roc4 = roc_auc_score(y_test, y_pred4) * 100
    
    # model5 = MLPClassifier()
    # model5.fit(X_train, y_train)
    # y_pred5 = model5.predict(X_test)
    # accuracy5 = accuracy_score(y_test, y_pred5) * 100
    # precision5 = precision_score(y_test, y_pred5) * 100
    # recall5 = recall_score(y_test, y_pred5) * 100
    # f1score5 = f1_score(y_test, y_pred5) * 100
    # roc5 = roc_auc_score(y_test, y_pred5) * 100

    accuracy = {'RF': accuracy1, 'SVM': accuracy2, 'LogisticRegression': accuracy4}
    precision = {'RF': precision1, 'SVM': precision2, 'LogisticRegression': precision4}
    recall = {'RF': recall1, 'SVM': recall2, 'LogisticRegression': recall4}
    f1score = {'RF': f1score1, 'SVM': f1score2, 'LogisticRegression': f1score4}
    # roc = {'RF': roc1, 'SVM': roc2, 'LogisticRegression': roc,  'MLP': roc5}
    return render(request, 'users/ML.html',
                  {"accuracy": accuracy, "precision": precision, "recall":recall, "f1score": f1score})


def Viewdata(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = settings.MEDIA_ROOT + "\\" + "Blood Pressure.csv"
    data = pd.read_csv(path)
    # print(data)
    data = data.to_html()
    
    return render(request, "users/Viewdata.html", {"data": data})



def prediction(request):
    if request.method == 'POST':
        age = int(request.POST.get('age'))
        Gender = int(request.POST.get('Gender'))
        BMI = float(request.POST.get('BMI'))
        Smoking = int(request.POST.get('Smoking'))
        Daily_steps	= int(request.POST.get('Daily steps'))
        Daily_alcohol = int(request.POST.get('Daily alcohol'))
        Daily_salt = request.POST.get('Daily_salt')
        Stress_Level = request.POST.get('Stress_Level')
        print('daily salt : ',Daily_salt)
        print('Stress_Level salt : ',Stress_Level)
        # Stress_Level = int(request.POST.get('Stress Level'))
        
        
        test_data = [age,Gender,BMI,Smoking,Daily_steps,Daily_alcohol,Daily_salt,Stress_Level]
        print(test_data)
        from .utility import ProcessMachineLearning
        test_pred = ProcessMachineLearning.classification(test_data)
        print("Test Result is:", test_pred)
        if test_pred[0] == 0:
            rslt = False
        else:
            rslt = True
            
            return render(request, "users/predictions_form.html", {"test_data": test_data, "result": rslt})
    else:

        return render(request, 'users/predictions_form.html', {})