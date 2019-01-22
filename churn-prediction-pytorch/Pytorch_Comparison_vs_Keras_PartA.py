import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv('Churn_Modelling.csv')
bank_data = data.iloc[:, 3:13].values
op_data = data.iloc[:, 13].values

#Removing categorical values
lblx = LabelEncoder()
bank_data[:, 1] = lblx.fit_transform(bank_data[:, 1])
lblx2 = LabelEncoder()
bank_data[:, 2] = lblx2.fit_transform(bank_data[:, 2])
bank_data

#Splitting into training and testing data
train_data, test_data, train_op1, test_op = train_test_split(bank_data, op_data, test_size=0.2, random_state=0)

#Standardizing the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
x_train = train_data.astype(np.float32)
test_data = scaler.fit_transform(test_data)
x_test = test_data.astype(np.float32)
train_op = train_op1.astype(np.float32)
y_train = train_op.reshape(-1, 1)
test_op = test_op.astype(np.float32)
y_test = test_op.reshape(-1, 1)

#Defining our custom model
class CustomModel(nn.Module):
    def __init__(self, input_dim,hidden_dim, output_dim):
        super(CustomModel, self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.sigmoid=nn.Sigmoid()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out=self.fc1(x)
        out=self.sigmoid(out)
        out = self.linear(out)
        return out

input_dim = 10
output_dim = 1
hidden_dim=100
loss_data=[]
accuracy_data=[]
residual_list=[]

model = CustomModel(input_dim,hidden_dim, output_dim)
#Instantiate Loss
criterion = nn.MSELoss()

#Set Learning rate
learning_rate = 0.01

#Define optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training our model
epochs = 500
for epoch in range(epochs):
    epoch += 1


    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    #Cleaning Gradients
    optimizer.zero_grad()

    outputs = model(inputs)

    # Loss Computation
    loss = criterion(outputs, labels)

    # Getting gradients
    loss.backward()

    # Updating parameters for optimization
    optimizer.step()

    #Print the data to console
    print('epoch count: {}, loss-value: {}'.format(epoch,loss.data[0]))
    loss_data.append(float(loss.data[0]))
    op = outputs.detach().numpy()

    pred = [1 if i > 0.5 else 0 for i in op]
    pred = np.array(pred).reshape(-1, 1)
    correct = 0
    correct = correct + ((pred == y_train).sum())
    accuracy = (correct * 100) / len(y_train)
    print("Accuracy:{}".format(accuracy))
    accuracy_data.append(accuracy)


#Evaluating our model
predicted=model(Variable(torch.from_numpy(x_test)))
y_pred=predicted.detach().numpy()
residual_list.append(y_test - y_pred)
y_pred1=[]
for i in y_pred:
    if(i>0.5):
        y_pred1.append(1)
    else:
        y_pred1.append(0)

y=np.array(y_pred1,dtype=np.float32)
y=y.reshape(-1,1)
correct=0
correct=correct+((y==y_test).sum())
correct
len(y_test)
accuracy=(correct*100)/len(y_test)
print("Accuracy:{}".format(accuracy))


#Plotting our analysis
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='Prajwal93', api_key='i10RjQJQzRUVBPSvoLzK')
import numpy as np

trace1 = go.Scatter(
    x = np.arange(1,epochs),
    y = np.array(loss_data),
    mode = 'markers',
)

trace2 = go.Scatter(
    x = np.arange(1,epochs),
    y = np.array(accuracy_data),
    mode = 'markers',
    name='accuracy values'
)

trace3 = go.Scatter(
    x = np.arange(1,len(y_test)),
    y = np.array(residual_list).flatten(),
    name='residuals',
    mode = 'markers'
)

trace4 = go.Heatmap(z=data.corr().as_matrix(),
                    x=data.columns.values,
                    y=data.columns.values)

trace5=go.Box(y=data[data.Exited==1].Age)

fig = tools.make_subplots(rows=5, cols=1, subplot_titles=('BoxPlot of the Age of Customers who had churned','Loss Computation', 'Accuracy Computation',
                                                          'Residual Plot', 'Correlation HeatMap'))
fig.append_trace(trace5,1,1 )
fig.append_trace(trace1,2,1)
fig.append_trace(trace2,3,1 )
fig.append_trace(trace3,4,1 )
fig.append_trace(trace4,5,1 )

fig['layout']['xaxis1'].update(title='Iterations')
fig['layout']['xaxis2'].update(title='Iterations')
fig['layout']['xaxis3'].update(title='Samples')
fig['layout']['xaxis4'].update(title='Columns')

fig['layout']['yaxis1'].update(title='Loss')
fig['layout']['yaxis2'].update(title='Accuracy')
fig['layout']['yaxis3'].update(title='Residuals')
fig['layout']['yaxis4'].update(title='Column')
fig['layout'].update(title='Churn Prediction')
plot_url = py.plot(fig, filename='Churn-Prediction')

