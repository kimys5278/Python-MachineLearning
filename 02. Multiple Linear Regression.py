#!/usr/bin/env python
# coding: utf-8

# # 02. Multiple Linear Regression

# ## SubTitle: one-hot-encoding

# In[49]:


import pandas as pd


# In[50]:


dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# In[51]:


X


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(drop='first'),[2])],remainder='passthrough')
X=ct.fit_transform(X)
X

#1 0 : 홈
#0 1 : 도서관
#0 0 : 카페


# # 데이터 세트 분리

# In[54]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)


# ### 학습(다중 선형 회귀)

# In[58]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)


# ### 예측 값과 실제 값 비교(테스트세트)

# In[59]:


Y_pred = reg.predict(X_test)


# In[60]:


Y_pred


# In[61]:


Y_test


# In[62]:


reg.coef_


# In[63]:


reg.intercept_


# # 모델평가

# In[64]:


reg.score(X_train,Y_train)


# In[65]:


reg.score(X_test,Y_test)


# # 다양한 평가 지표(회귀 모델)

# 1. MAE (Mean Absolute Error) : (실제 값과 예측값) 차이의 절대값
# 2. MSE (Mean Squared Error) : (실제 값과 예측값) 차이의 제곱
# 3. RMSE (Root Mean Squared Error) : (실제 값과 예측값) 차이의 제곱에 루트
# 4. R2 : 결정 계수
# 
# >R2 는 1에 가까울수록 좋고, 나머지는 0에 가까울수록 더 좋다.

# In[68]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_test,Y_pred) #실제 값, 예측 값 #MAE


# In[71]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,Y_pred) #실제 값, 예측 값 #MAE


# In[72]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,Y_pred,squared = False) #실제 값, 예측 값 #RMAE 
#squared = False -> 제곱하지 말라는 코드 즉, 루트,


# In[73]:


from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)#R2


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




