
# coding: utf-8

# # Linear Regression Example

# In[1]:


import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy
from sklearn.metrics import mean_squared_error


# ## Read file in pandas `DataFrame`

# In[2]:


sf = pandas.read_csv('final_data.csv')


# ## Drop unwanted features

# In[3]:


sf.drop(sf.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True)
sf.head()


# ## Analyze Linear correlation with house prices

# In[4]:


sf.corr()['lastsoldprice']


# ## Pick top correlated values for simplicity

# In[5]:


X = sf[['finishedsqft']]
Y = sf['lastsoldprice']
X.head()


# ## Split into training and test data set
# `X` represent features, `y` represents labels. Take 30% random samples as test samples for verification.

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# ## Train the model

# In[7]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Model: Coefficient: {} Intercept: {}'.format(regressor.coef_, regressor.intercept_))


# ## Test the model

# In[8]:


y_pred = regressor.predict(X_test)


# In[9]:


lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = numpy.sqrt(lin_mse)
print('Liner Regression RMSE: {:,.2f}'.format(lin_rmse))


# ## Plot Regression Results

# In[10]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')
plt.scatter(X_test['finishedsqft'], y_test,  color='black')
plt.xlabel('Finished Sqft.')
plt.xticks(numpy.arange(100, 2000, step=100))
plt.ylabel('Price ($)')

axes = plt.gca()
x_vals = numpy.array(axes.get_xlim())
y_vals = regressor.coef_[0] * x_vals + regressor.intercept_
plt.plot(x_vals, y_vals, '-')

plt.xticks(())
plt.yticks(())

plt.show()


# ## Run the model
# In this simple case, we can just call `regressor.predict(X)` where X is a vector of bathrooms and finished sqft.

# In[11]:


prediction = regressor.predict(numpy.array([[900]]))
print('Predicted price is ${:,.2f}'.format(prediction[0]))

