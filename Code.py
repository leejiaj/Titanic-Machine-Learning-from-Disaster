
# coding: utf-8

# In[209]:

#import packages
import pandas as pd
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# In[210]:

#read the train and test datasets
train=pd.read_csv('path/train.csv', header=0)
test=pd.read_csv('path/test.csv', header=0)


# In[211]:

#Shape of the datasets 
print('Shape of train dataset: ',train.shape)
print('Shape of test dataset: ',test.shape)
#Features in the datasets
print('Columns in train dataset: ',train.columns)
print('Columns in test dataset: ',test.columns)


# In[212]:

#Statistical summary of train dataset
train.describe()


# In[213]:

#Statistical summary of test dataset
test.describe()


# In[214]:

#displaying the top 20 instaces of the train dataset
train.head(20)


# In[215]:

#displaying the top 20 instaces of the test dataset
test.head(20)


# In[216]:

#Checking the number of null values in the columns of the train dataset
train.isnull().sum()


# In[217]:

#Checking the number of null values in the columns of the test dataset
test.isnull().sum()


# In[218]:

#Analyzing the relationship between the features and the class


# In[219]:

sns.barplot('Pclass','Survived', data=train)


# In[220]:

sns.barplot('Sex','Survived', data=train)


# In[221]:

ageplot = sns.FacetGrid(train, col='Survived')
ageplot.map(plt.hist, 'Age', bins=10)


# In[222]:

sns.barplot('SibSp','Survived', data=train)


# In[223]:

sns.barplot('Parch','Survived', data=train)


# In[224]:

sns.barplot('Embarked','Survived', data=train)


# In[225]:

#Splitting the feature name into features name, surname and title in the train dataset
train[['Surname', 'Name']] = train['Name'].str.split(',\s+', expand=True)
train['Title'] = train['Name'].str.split(' ').str[0]
#Displaying the top 20 instances from the train dataset
train.head(20)


# In[226]:

#Splitting the feature name into features name, surname and title in the test dataset
test[['Surname', 'Name']] = test['Name'].str.split(',\s+', expand=True)
test['Title'] = test['Name'].str.split(' ').str[0]
#Displaying the top 20 instances of the test dataset
test.head(20)


# In[227]:

#Removing the unimportant features from the train dataset
train=train.drop(['PassengerId','Name','Ticket','Fare','Cabin','Surname'],axis=1)
#Displaying the top 20 instances of the train dataset
train.head(20)


# In[228]:

#Removing the unimportant features from the test dataset
test=test.drop(['PassengerId','Name','Ticket','Fare','Cabin','Surname'],axis=1)
#Displaying the top 20 instances of the test dataset
test.head(20)


# In[229]:

#Displaying the various values possible for feature Title in the Train dataset
pd.value_counts(train.Title)


# In[230]:

#Displaying the various values possible for feature Title in the Test dataset
pd.value_counts(test.Title)


# In[231]:

#Replacing values Mlle and Ms by Miss in the train dataset
train['Title']=train['Title'].replace(['Mlle.','Ms.'],'Miss.')
pd.value_counts(train.Title)


# In[232]:

#Replacing values Mlle and Ms by Miss in the test dataset
test['Title']=test['Title'].replace(['Mlle.','Ms.'],'Miss.')
pd.value_counts(test.Title)


# In[233]:

#Replacing value Mme by Mrs in the train dataset
train['Title']=train['Title'].replace('Mme.','Mrs.')
pd.value_counts(train.Title)


# In[234]:

#Replacing values Col, Major, Capt by Officer in the Train dataset
train['Title']=train['Title'].replace(['Col.','Major.','Capt.'],'Officer.')
pd.value_counts(train.Title)


# In[235]:

#Replacing values Col, Major, Capt by Officer in the Test dataset
test['Title']=test['Title'].replace(['Col.','Major.','Capt.'],'Officer.')
pd.value_counts(test.Title)


# In[236]:

#Replacing values Don, Dona, Sir, Jonkhheer, Lady, the by Royalty in the Train dataset
train['Title']=train['Title'].replace(['Don.','Dona.','Sir.','Jonkheer.','Lady.','the'],'Royalty.')
pd.value_counts(train.Title)


# In[237]:

#Replacing values Don, Dona, Sir, Jonkhheer, Lady, the by Royalty in the Test dataset
test['Title']=test['Title'].replace(['Don.','Dona.','Sir.','Jonkheer.','Lady.','the'],'Royalty.')
pd.value_counts(test.Title)


# In[238]:

#Imputing missing values


# In[239]:

#Imputing missing values for attribute Age in the train dataset
train.Age.fillna(train.Age.median(), inplace=True)
#Displaying the top 20 instances in the train dataset
train.head(20)


# In[240]:

#Imputing missing values for attribute Age in the test dataset
test.Age.fillna(test.Age.median(), inplace=True)
#Displaying the top 20 instances in the test dataset
test.head(20)


# In[241]:

#Checking that the null values in column Age have been imputed in the train dataset
train.isnull().sum()


# In[242]:

#Checking if any column has null values in the test dataset
test.isnull().sum()


# In[243]:

#Displaying values possible for feature Embarked in the train dataset
pd.value_counts(train.Embarked)


# In[244]:

#Imputing missing values for attribute Embarked in the train dataset
train.Embarked.fillna('S', inplace=True)
#Displaying the top 20 instances in the tarin dataset
train.head(20)


# In[245]:

#Checking that the missing values in column Embarked have been imputed in the train dataset
train.isnull().sum()


# In[246]:

#Mapping categorical values to numerical values


# In[247]:

#Mapping categorical values to numerical values for feature title in train dataset
train['Title']=train['Title'].map({'Mr.':0, 'Miss.':1, 'Mrs.':2, 'Master.':3, 'Dr.':4, 'Rev.':5, 'Royalty.':6, 'Officer.':7})
train.head(20)


# In[248]:

#Mapping categorical values to numerical values for feature title in test dataset
test['Title']=test['Title'].map({'Mr.':0, 'Miss.':1, 'Mrs.':2, 'Master.':3, 'Dr.':4, 'Rev.':5, 'Royalty.':6, 'Officer.':7})
test.head(20)


# In[249]:

#Displaying values possible for feature Sex in train dataset
pd.value_counts(train.Sex)


# In[250]:

#Displaying values possible for feature Sex in test dataset
pd.value_counts(test.Sex)


# In[251]:

#Mapping categorical values to numerical values for feature Sex in train dataset
train['Sex']=train['Sex'].map({'male':0, 'female':1})
train.head(20)


# In[252]:

#Mapping categorical values to numerical values for feature Sex in test dataset
test['Sex']=test['Sex'].map({'male':0, 'female':1})
test.head(20)


# In[253]:

#Displaying values possible for feature Embarked in the train dataset
pd.value_counts(train.Embarked)


# In[254]:

#Dispalying values possible for feature Embarked in the test dataset
pd.value_counts(test.Embarked)


# In[255]:

#Mapping categorical values to numerical values for feature Embarked in train dataset
train['Embarked']=train['Embarked'].map({'S':0, 'C':1, 'Q':2})
train.head(20)


# In[256]:

#Mapping categorical values to numerical values for feature Embarked in test dataset
test['Embarked']=test['Embarked'].map({'S':0, 'C':1, 'Q':2})
test.head(20)


# In[257]:

#Splitting features and class
X=train.loc[:, 'Pclass':'Title']
Y=train.loc[:,'Survived']


# In[258]:

#Predicting the class values (survived or not) for the test dataset provided using all the models
x=np.array(X)
y=np.array(Y)
cnames=["Decision Tree","Deep Learning","Bagging","Random Forest","Ada Boosting","Gradient Boosting","SVM","GaussianNB","Logistic Regression","k-NN"]
classifiers=[DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth=6, min_samples_leaf=6),
            MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7,9,7), activation='tanh'),
            BaggingClassifier(n_estimators=10, bootstrap=False, warm_start=False),
            RandomForestClassifier(n_estimators=200, max_depth=5, random_state=150),
            AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=4), algorithm="SAMME",n_estimators=200),
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, random_state=20),
            SVC(kernel='rbf', C=1),
            GaussianNB(),
            LogisticRegression(penalty='l2', solver='liblinear', max_iter=100),
            KNeighborsClassifier(n_neighbors=3)]
predtest = []
for i in range(len(classifiers)):
    predtest.append([])
for i in range(len(classifiers)):
    classifiers[i].fit(x,y)
    predtest[i]=classifiers[i].predict(test)   
for i in range(len(classifiers)):
    print(cnames[i])
    print(predtest[i])


# In[259]:

#Splitting the train dataset into train and test datasets for holdout validation
X_training, X_testing, Y_training, Y_testing = train_test_split(X,Y,test_size=0.2,random_state=1)
print('Shape of X_training: ',X_training.shape)
print('Shape of Y_training: ',Y_training.shape)
print('Shape of X_testing: ',X_testing.shape)
print('Shape of Y_testing: ',Y_testing.shape)


# In[260]:

#holdout validation by splitting the train dataset into train and test datasets


# In[261]:

#Decision Tree
dt=DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth=5, min_samples_leaf=6)
dt.fit(X_training,Y_training)
Y_predictdt=dt.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictdt))


# In[262]:

#Deep Learning
dl=MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7,9,7), activation='tanh')
dl.fit(X_training,Y_training)
Y_predictdl=dl.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictdl))


# In[263]:

#Bagging
bag=BaggingClassifier(n_estimators=10)
bag.fit(X_training,Y_training)
Y_predictbag=bag.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictbag))


# In[264]:

#Random Forest
rf=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=10, min_samples_leaf=4)
rf.fit(X_training,Y_training)
Y_predictrf=rf.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictrf))


# In[265]:

#Ada Boost
ab=AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=4), algorithm="SAMME",n_estimators=200)
ab.fit(X_training,Y_training)
Y_predictab=ab.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictab))


# In[266]:

#Gradient Boosting
gb=GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, random_state=20)
gb.fit(X_training,Y_training)
Y_predictgb=gb.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictgb))


# In[267]:

#SVM
svm=SVC(kernel='rbf', C=1)
svm.fit(X_training,Y_training)
Y_predictsvm=svm.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictsvm))


# In[268]:

#Naive Bayes
gnb=GaussianNB()
gnb.fit(X_training,Y_training)
Y_predictgnb=gnb.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictgnb))


# In[269]:

#Logistic Regression
lr=LogisticRegression(penalty='l2', solver='liblinear', max_iter=100)
lr.fit(X_training,Y_training)
Y_predictlr=lr.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictlr))


# In[270]:

#k-Nearest Neighbors
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_training,Y_training)
Y_predictknn=knn.predict(X_testing)
print('Accuracy: ',accuracy_score(Y_testing, Y_predictknn))


# In[271]:

#cross validation by splitting the train dataset into train and test datasets


# In[272]:

#Decision Tree
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = DecisionTreeClassifier(criterion = "gini", random_state = 50, max_depth=6, min_samples_leaf=6)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Decision Tree')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[273]:

#Deep Learning
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7,9,7), activation='tanh')
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Deep Learning')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[274]:

#Bagging
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = BaggingClassifier(n_estimators=10, bootstrap=False, warm_start=False)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Bagging')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[275]:

#RandomForest
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=150)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[276]:

#AdaBoost
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=4), algorithm="SAMME",n_estimators=200)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - AdaBoost')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[277]:

#GradientBoosting
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, random_state=20)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - GradientBoost')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[278]:

#SVM
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = SVC(kernel='rbf', C=1, probability=True)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - SVM with rbf kernel')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[279]:

#Naive Bayes
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = GaussianNB()
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Gaussian Naive Bayes')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[280]:

#Logistic Regression
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = LogisticRegression(penalty='l2', solver='liblinear', max_iter=100)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[281]:

#kNN
random_state = np.random.RandomState(0)
tprs = []
aucs = []
acc=[]
prec=[]
rec=[]
f1=[]
mean_fpr = np.linspace(0, 1, 100)
i = 0
cv = StratifiedKFold(n_splits=10)
classifier = KNeighborsClassifier(n_neighbors=3)
for train, test in cv.split(x, y):
    probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
    y_pred = classifier.predict(x[test])
    acc.append(accuracy_score(y[test], y_pred))
    prec.append(precision_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    rec.append(recall_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    f1.append(f1_score(y[test], y_pred, average='macro', labels=np.unique(y_pred)))
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
            label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - k Nearest Neighbors')
plt.legend(loc="lower right")
plt.show()
accsum=0
precsum=0
recsum=0
f1sum=0
for j in acc:
    accsum+=j*100
avgacc=accsum/10
for j in prec:
    precsum+=j*100
avgprec=precsum/10
for j in rec:
    recsum+=j*100
avgrec=recsum/10
for j in f1:
    f1sum+=j*100
avgf1=f1sum/10
print('Accuracy: ',avgacc)
print('Precision: ',avgprec)
print('Recall: ',avgrec)
print('F1-Score: ',avgf1)


# In[ ]:



