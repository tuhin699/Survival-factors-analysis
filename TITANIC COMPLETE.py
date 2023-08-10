#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install scikit-learn


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[7]:


titanicdata=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\TITANIC.csv")


# In[8]:


titanicdata.head(15)


# In[10]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
target = 'Survived'
titanicdata = titanicdata[features + [target]]


# In[12]:


titanicdata['Age'].fillna(titanicdata['Age'].median(), inplace=True)


# In[14]:


label_encoder = LabelEncoder()
titanicdata['Sex'] = label_encoder.fit_transform(titanicdata['Sex'])


# In[16]:


X = titanicdata.drop(target, axis=1)
y = titanicdata[target]


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


clf = RandomForestClassifier(random_state=42)


# In[19]:


clf.fit(X_train, y_train)


# In[20]:


predictions = clf.predict(X_test)


# In[21]:


accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")


# In[22]:


#pip install matplotlib seaborn


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[24]:


titanicdata=pd.read_csv("C:\\Users\\LENOVO\\Downloads\\TITANIC.csv")


# In[26]:


plt.figure(figsize=(10, 6))
sns.histplot(data=titanicdata, x='Age', hue='Survived', multiple='stack', bins=20)
plt.title('Distribution of Survivors by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['Did not survive', 'Survived'])
plt.show()


# In[27]:


feature_importances = clf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[ ]:




