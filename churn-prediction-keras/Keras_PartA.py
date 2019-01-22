
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from bokeh.plotting import figure,show
from bokeh.layouts import column

data=pd.read_csv('Churn_Modelling.csv')

bank_data = data.iloc[:, 3:13].values
op_data = data.iloc[:, 13].values

lblx=LabelEncoder()
bank_data[:,1]=lblx.fit_transform(bank_data[:,1])

lblx2=LabelEncoder()
bank_data[:,2]=lblx2.fit_transform(bank_data[:,2])

onehotEncoder=OneHotEncoder(categorical_features=[1])
bank_data=onehotEncoder.fit_transform(bank_data).toarray()
bank_data=bank_data[:,1:]

#Splitting into train test data
train_data,test_data,train_op,test_op=train_test_split(bank_data,op_data,test_size=0.2,random_state=0)

#Standardization
print(test_data.shape)
scaler=StandardScaler()
train_data=scaler.fit_transform(train_data)
test_data=scaler.fit_transform(test_data)

print(train_data.shape)
#Building our model
model=Sequential()
model.add(Dense(units=6,init = 'uniform',activation="relu",input_dim=11))
model.add(Dense(units=6,init = 'uniform',activation="relu"))
model.add(Dense(1,init='uniform',activation="sigmoid"))

model.summary()
#Compiling our model
model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
epochs=100
#Fitting the model
history=model.fit(train_data,train_op, epochs=epochs,validation_split=0.1, batch_size=10,shuffle=True)

#Evaluating our model
y_pred=model.predict(test_data,batch_size=10,verbose=1)
print(y_pred-test_op)
score=model.evaluate(test_data,test_op,verbose=1)
print(score)

#Plotting the analysis using bokeh
p = figure(plot_width=1000, plot_height=600,title="Accuracy of the model for each epoch")
x=np.arange(1,epochs+1)
p.circle(x,history.history['acc'], size=8, color="red", alpha=0.5)
p.line(x,history.history['acc'],line_color="red",line_width=2,legend="Computed Accuracy")
p.line(x,history.history['val_acc'],line_width=2,legend="Validation Accuracy")
p.circle(x,history.history['val_acc'], size=8, alpha=0.5)

p1 = figure(plot_width=1000, plot_height=600,title='Loss calculated for each epoch')
x=np.arange(1,epochs+1)
p1.circle(x,history.history['loss'], size=8, color="red", alpha=0.5)
p1.line(x,history.history['loss'],line_width=2,line_color="red",legend="Computed Loss")
p1.line(x,history.history['val_loss'],line_width=2,legend="Validation loss")
p1.circle(x,history.history['val_loss'], size=8, alpha=0.5)

p.legend.location = "top_left"
p.legend.click_policy="hide"
p.xaxis.axis_label='Iterations'
p1.xaxis.axis_label='Iterations'
p.yaxis.axis_label="Accuracy"
p1.yaxis.axis_label="Loss"
p1.legend.location = "top_left"
p1.legend.click_policy="hide"

dat=data.iloc[:,[5,13]].values
gender=['Male','Female']
count=[]
malecount=0
for i in dat:
    if i[0]=="Male":
        malecount+=i[1]
count.append(malecount)

femalecount=0
for i in dat:
    if i[0]=="Female":
        femalecount+=i[1]
count.append(femalecount)

p2 = figure(x_range=gender, plot_height=350, title="Count of people based on gender who decided to exit",
           toolbar_location=None, tools="")

p2.vbar(x=gender, top=count, width=0.4)
p2.xgrid.grid_line_color = None
p2.y_range.start = 0

ageArray=data[data['Exited']==1].iloc[:,6].values
ageArray
unique, counts = np.unique(ageArray, return_counts=True)
ageDict=dict(zip(unique, counts))
ageList=[]
for c in ageDict:
    ageList.append(str(c))

countList=[]
for d in ageDict.values():
    countList.append(d)
p3 = figure(x_range=ageList, plot_height=350,plot_width=900, title="Age of people who decided to exit",
           toolbar_location=None, tools="")

p3.vbar(x=ageList, top=countList, width=0.4)
p3.xgrid.grid_line_color = None
p3.y_range.start = 0
p3.xaxis.axis_label='Age'
p3.yaxis.axis_label="Count"

show(column(p2,p3,p,p1))
