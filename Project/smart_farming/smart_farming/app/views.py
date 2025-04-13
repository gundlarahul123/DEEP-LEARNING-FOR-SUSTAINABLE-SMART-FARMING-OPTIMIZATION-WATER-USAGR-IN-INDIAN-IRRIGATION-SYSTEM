# Libraries
from django.shortcuts import render,redirect
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from .models import User




################ Home #################
def home(request):
	return render(request,'home1.html')
def login(request):
	return render(request,'loginform.html')
def loginCheck(request):
	if request.method=="POST":
		print('printtttttttttttttttttttttttttttttttt')
		username= request.POST.get('username')
		password= request.POST.get('email')
		try:
			user_object = User.objects.get(firstname=username,password=password)
			print(user_object)
		except:
			#user_object = None
			print('hello')
		if user_object is not None:
			print('hiiiiiiii')
			request.session['useremail'] = user_object.email
			return redirect('home')
			print('hiiiiiiii')
	return render(request,'home.html')	
def logout(request):
	return render(request,'index.html')	
def reg(request):
	return render(request,'register.html')

######## SVM ######
def save(request):
	if request.method == 'POST':
		print('printtttttttttttttttttttttttttttttttt')
		print('checkkkkkkkkkkkkkkkkk')
		username= request.POST.get('username')
		password= request.POST.get('password')
		address= request.POST.get('address')
		email= request.POST.get('email')
		age= request.POST.get('age')
		gender= request.POST.get('gender')
		phone= request.POST.get('phone')
		user=User()
		user.firstname= request.POST.get('username')
		user.password= request.POST.get('password')
		user.address= request.POST.get('address')
		user.email= request.POST.get('email')
		user.age= request.POST.get('age')
		user.gender= request.POST.get('gender')
		user.phone= request.POST.get('phone')
		user.save()		
		return render(request,'loginform.html')
	return render(request,'loginform.html')	

######## SVM ######
def nvb(request):
	return render(request,'pacweb1.html')
def pac(request):
	if request.method == 'POST':
		if request.method == 'POST':
			headline1= request.POST.get('headline1')
			headline2= request.POST.get('headline2')
			headline3= request.POST.get('headline3')
			headline4= request.POST.get('headline4')
			headline5= request.POST.get('headline5')
			headline6= request.POST.get('headline6')
			headline7= request.POST.get('headline7')
			headline8= request.POST.get('headline8')
			headline9= request.POST.get('headline9')
			headline10= request.POST.get('headline10')
			headline11= request.POST.get('headline11')
			headline12= request.POST.get('headline12')
			headline13= request.POST.get('headline13')
			headline14= request.POST.get('headline14')
			headline15= request.POST.get('headline15')
			headline16= request.POST.get('headline16')

			print(headline1)
			
			
			headline1= int(headline1)
			headline2 = int(headline2)
			headline3 = int(headline3)
			headline4 = float(headline4)	
			headline5 = float(headline5)	
			headline6 = float(headline6)	
			headline7 = float(headline7)	
			headline8 = float(headline8)	
			headline9 = int(headline9)	
			headline10 = float(headline10)	
			headline11 = float(headline11)	
			headline12 = float(headline12)	
			headline13 = float(headline13)	
			headline14 = int(headline14)	
			headline15 = float(headline15)	
			headline16 = float(headline16)	

			from django.shortcuts import render
			from django.http import HttpResponse
			import pandas as pd
			import numpy as np
			import matplotlib.pyplot as plt
			from sklearn.model_selection import train_test_split
			from sklearn.feature_extraction.text import TfidfVectorizer
			import itertools
			from sklearn import metrics
			import os
			import seaborn as sns
			from sklearn.model_selection import train_test_split
			from sklearn.metrics import confusion_matrix
			df = pd.read_csv(r'C:\Users\Rahul\OneDrive\Desktop\Project\smart_farming\smart_farming\smart_farming.csv', encoding='ISO-8859-1')

			df1=df.fillna(0)

			dfle = df1.copy()
			dfle
			dfle
			print(dfle)
			X = dfle.iloc[:, 0:16].values


			y = dfle.label


			#atest=[[0,0,0,0,0,5849,0,320,360,1,0]]
			#atest1=[[0,0,0,0,0,12500,3000,320,360,1,1]]
			#train_test separation
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
			atest=[[headline1,headline2,headline3,headline4,headline5,headline6,headline7,headline8,headline9,headline10,headline11,headline12,headline13,headline14,headline15,headline16]]
			#train_test separation
			X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
			linear_clf = RandomForestClassifier(n_estimators = 100, max_depth =4, random_state=42, min_samples_split = 5, oob_score = True, n_jobs = -1, max_features = "auto",criterion = 'entropy', max_leaf_nodes = 30,class_weight='balanced_subsample',min_samples_leaf = 10)
			linear_clf.fit(X_train, y_train)
			pred = linear_clf.predict(X_test)
			print('=====================================================================')
			pred1 = linear_clf.predict(atest)
			#pred2 = linear_clf.predict(atest1)
			print(pred1)
			
			print(linear_clf.score(X_train, y_train))
			d={'predictedvalue':pred1,'accuracy':linear_clf.score(X_train, y_train)}				 
	return render(request,'result.html',d)
def svm(request):	
	df = pd.read_csv(r'C:\Users\Rahul\OneDrive\Desktop\Project\smart_farming\smart_farming\smart_farming.csv', encoding='ISO-8859-1')


	print(df)

	X = df.iloc[:, 0:16].values


	y = df.label


	#A_test=[[90,42,43,20.87974371,82.00274423,6.502985292,202.9355362,29.44606392,2,8.677355267,10.10987524,435.6112257,3.121394502,4,11.7439101,57.60730814]]
	#A_test1=[[54,77,85,17.1418614,17.0662427,7.829211144,83.74606679,14.94566796,1,6.323744635,6.741236944,350.9962152,9.448286238,1,16.62088061,26.40204791]]

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)



	from sklearn.ensemble import RandomForestClassifier
	rf = RandomForestClassifier(n_estimators = 200, random_state = 0)
	rf.fit(X_train, y_train)
	pred = rf.predict(X_test)
	#pred1 = rf.predict(A_test)
	#pred2 = rf.predict(A_test1)
	print(pred)
	print('--------------------------------------------------------------------')
	#print(pred1)
	#print(pred2)
	from sklearn.metrics import accuracy_score
	print(accuracy_score(y_test, pred))
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)		
def dec(request):
	return render(request,'result.html')
def randomf(request):
	df = pd.read_csv(r'C:\Users\Rahul\OneDrive\Desktop\Project\smart_farming\smart_farming\smart_farming.csv', encoding='ISO-8859-1')


	print(df)

	X = df.iloc[:, 0:16].values


	y = df.label


	#A_test=[[90,42,43,20.87974371,82.00274423,6.502985292,202.9355362,29.44606392,2,8.677355267,10.10987524,435.6112257,3.121394502,4,11.7439101,57.60730814]]
	#A_test1=[[54,77,85,17.1418614,17.0662427,7.829211144,83.74606679,14.94566796,1,6.323744635,6.741236944,350.9962152,9.448286238,1,16.62088061,26.40204791]]

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)



	from sklearn.ensemble import RandomForestClassifier
	rf = DecisionTreeClassifier()
	rf.fit(X_train, y_train)
	pred = rf.predict(X_test)
	#pred1 = rf.predict(A_test)
	#pred2 = rf.predict(A_test1)
	print(pred)
	print('--------------------------------------------------------------------')
	#print(pred1)
	#print(pred2)
	from sklearn.metrics import accuracy_score
	print(accuracy_score(y_test, pred))
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def mnb(request):
	return render(request,'acc1.html',d)
def graph(request):
	df = pd.read_csv(r'C:\Users\Rahul\OneDrive\Desktop\Project\smart_farming\smart_farming\smart_farming.csv', encoding='ISO-8859-1')


	print(df)

	X = df.iloc[:, 0:16].values


	y = df.label


	#A_test=[[90,42,43,20.87974371,82.00274423,6.502985292,202.9355362,29.44606392,2,8.677355267,10.10987524,435.6112257,3.121394502,4,11.7439101,57.60730814]]
	#A_test1=[[54,77,85,17.1418614,17.0662427,7.829211144,83.74606679,14.94566796,1,6.323744635,6.741236944,350.9962152,9.448286238,1,16.62088061,26.40204791]]

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)



	from sklearn.ensemble import RandomForestClassifier
	rf = PassiveAggressiveClassifier()
	rf.fit(X_train, y_train)
	pred = rf.predict(X_test)
	#pred1 = rf.predict(A_test)
	#pred2 = rf.predict(A_test1)
	print(pred)
	print('--------------------------------------------------------------------')
	#print(pred1)
	#print(pred2)
	from sklearn.metrics import accuracy_score
	print(accuracy_score(y_test, pred))
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def accuracy(request):
	df=pd.read_csv(r'C:\Users\Rahul\OneDrive\Desktop\Project\smart_farming\smart_farming\smart_farming.csv')

	print(df)

	X = df.iloc[:, 0:16].values


	y = df.label


	#A_test=[[90,42,43,20.87974371,82.00274423,6.502985292,202.9355362,29.44606392,2,8.677355267,10.10987524,435.6112257,3.121394502,4,11.7439101,57.60730814]]
	#A_test1=[[54,77,85,17.1418614,17.0662427,7.829211144,83.74606679,14.94566796,1,6.323744635,6.741236944,350.9962152,9.448286238,1,16.62088061,26.40204791]]

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)



	from sklearn.ensemble import RandomForestClassifier
	rf = LogisticRegression()
	rf.fit(X_train, y_train)
	pred = rf.predict(X_test)
	#pred1 = rf.predict(A_test)
	#pred2 = rf.predict(A_test1)
	print(pred)
	print('--------------------------------------------------------------------')
	#print(pred1)
	#print(pred2)
	from sklearn.metrics import accuracy_score
	print(accuracy_score(y_test, pred))
	score = metrics.accuracy_score(y_test, pred)
	print(metrics.accuracy_score(y_test, pred))
	d={'accuracy':metrics.accuracy_score(y_test, pred)}	
	return render(request,'acc1.html',d)
def accuracy1(request):
	return render(request,'index.html')	
			