#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warning
import warnings
warnings.filterwarnings("ignore")


# In[16]:


df=pd.read_csv('survey lung cancer.csv')
df


# In[17]:


df.shape


# In[18]:


df.duplicated()


# In[19]:


df.duplicated().sum()


# In[20]:


df=df.drop_duplicates()


# In[21]:


df.isnull().sum()


# In[22]:


df.info()


# In[23]:


df.describe()


# In[24]:


column_of_interest = 'AGE'  
mode = df[column_of_interest].mode()
print(f"Mode of {column_of_interest}: {mode}")
column_of_interest = 'SMOKING'  
mode = df[column_of_interest].mode()
print(f"Mode of {column_of_interest}: {mode}")
column_of_interest = 'ANXIETY'  
mode = df[column_of_interest].mode()
print(f"Mode of {column_of_interest}: {mode}")


# In[25]:


column_of_interest = 'AGE'  
mean = df[column_of_interest].mean()
print(f"Mean of {column_of_interest}: {mean}")
column_of_interest = 'SMOKING'  
mean = df[column_of_interest].mean()
print(f"Mean of {column_of_interest}: {mean}")
column_of_interest = 'ANXIETY'  
mean = df[column_of_interest].mean()
print(f"Mean of {column_of_interest}: {mean}")


# In[26]:


column_of_interest = 'AGE'  
median = df[column_of_interest].median()
print(f"Median of {column_of_interest}: {median}")
column_of_interest = 'SMOKING'  
median = df[column_of_interest].median()
print(f"Median of {column_of_interest}: {median}")
column_of_interest = 'ANXIETY'  
median = df[column_of_interest].median()
print(f"Median of {column_of_interest}: {median}")


# In[ ]:







# In[28]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])


# In[ ]:





# In[29]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

# Assuming 'target' is your dependent variable
model = sm.OLS.from_formula('LUNG_CANCER ~ AGE + SMOKING  + YELLOW_FINGERS', data=df).fit()

# Perform the ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Extract the F-statistic and associated p-value
f_statistic = anova_table['F'][0]
p_value = anova_table['PR(>F)'][0]

print("F-Statistic:", f_statistic)
print("P-Value:", p_value)


# In[30]:


import scipy.stats as stats
group_a_column = df['AGE']
group_b_column = df['YELLOW_FINGERS']

# Perform a two-sample t-test assuming unequal variances (Welch's t-test)
t_statistic, p_value = stats.ttest_ind(group_a_column, group_b_column, equal_var=False)

# Display the results
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Determine whether the difference is statistically significant at a significance level (e.g., alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")


# In[31]:


contingency_table = pd.crosstab(df['AGE'], df['YELLOW_FINGERS'])
print("Contingency Table:")
print(contingency_table)


# In[32]:


import scipy.stats as stats

# Perform the chi-square test of independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Display the results
print("Chi-Square Statistic:", chi2_stat)
print("P-Value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies Table:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
# Determine whether there is a significant association between the variables
alpha = 0.05
if p_value < alpha:
    print("There is a significant association between the variables.")
else:
    print("There is no significant association between the variables.")


# In[33]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
X = df[['AGE']]  # Independent variable
Y = df['SMOKING']     # Dependent variable
# Create a linear regression model
model = LinearRegression()
# Fit the model to the data
model.fit(X, Y)
# Get the coefficients (slope and intercept)
slope = model.coef_[0]
intercept = model.intercept_

# Predict Y values based on the model
predicted_Y = model.predict(X)

# Print the coefficients
print("Slope (Coefficient):", slope)
print("Intercept:", intercept)
# Plot the original data points and the regression line
plt.scatter(X, Y, label='Original Data')
plt.plot(X, predicted_Y, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()


# In[34]:


def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))


# In[35]:


plot('GENDER')


# In[36]:


plot('AGE')


# In[37]:


plot('ANXIETY')


# In[38]:


plot('YELLOW_FINGERS')


# In[39]:


df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new


# In[40]:


cn=df_new.corr()
cn


# In[41]:


cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()


# In[42]:


kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size= 0.25, random_state=0)


# In[44]:


from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)


# In[45]:


y_lr_pred= lr_model.predict(X_test)
y_lr_pred


# In[46]:


from sklearn.metrics import classification_report, accuracy_score, f1_score
lr_cr=classification_report(y_test, y_lr_pred)
print(lr_cr)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
dt_model= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_model.fit(X_train, y_train)


# In[48]:


y_dt_pred= dt_model.predict(X_test)
y_dt_pred


# In[49]:


dt_cr=classification_report(y_test, y_dt_pred)
print(dt_cr)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[51]:


y_rf_pred= rf_model.predict(X_test)
y_rf_pred


# In[52]:


rf_cr=classification_report(y_test, y_rf_pred)
print(rf_cr)


# In[53]:


from sklearn.neighbors import KNeighborsClassifier  
knn_model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn_model.fit(X_train, y_train)


# In[54]:


y_knn_pred= knn_model.predict(X_test)
y_knn_pred


# In[55]:


knn_cr=classification_report(y_test, y_knn_pred)
print(knn_cr)


# In[56]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Logistic regerssion model
lr_model_scores = cross_val_score(lr_model,X, Y, cv=kf)

# Decision tree model
dt_model_scores = cross_val_score(dt_model,X, Y, cv=kf)

# KNN model
knn_model_scores = cross_val_score(knn_model,X, Y, cv=kf)


# Random forest model
rf_model_scores = cross_val_score(rf_model,X, Y, cv=kf)




print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Decision tree models' average accuracy:", np.mean(dt_model_scores))
print("KNN models' average accuracy:", np.mean(knn_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))


# In[87]:


# Import necessary library
import pandas as pd

# Load your dataset (assuming you have a CSV file)
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('survey lung cancer.csv')

# Assuming the column 'gender' contains 'Male' and 'Female' strings
# Replace 'Male' with 1 and 'Female' with 0 in the 'gender' column
data['GENDER'] = data['GENDER'].replace({'M': 1, 'F': 0})

# Now, 'Male' is replaced with 1 and 'Female' is replaced with 0 in the 'gender' column



# In[83]:


import pandas as pd

# Load your dataset (assuming you have a CSV file)
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('survey lung cancer.csv')

# Assuming the column 'gender' contains 'Male' and 'Female' strings
# Map 'Male' to 1 and 'Female' to 0
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'Yes': 1.0, 'No': 0.0})

# Now, 'Male' is represented as 1 and 'Female' is represented as 0 in the 'gender' column


# In[88]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Assuming you have a dataset loaded into a DataFrame called 'data'
# 'X' represents the features, and 'y' represents the target variable
# Split the dataset into training and testing sets
X = data.drop(columns=['CHEST PAIN']) # Replace 'target_column' with the actual name
y = data['CHEST PAIN'] # Replace 'target_column' with the actual name of your target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an SVM classifier
model = SVC(kernel='linear', random_state=42)
# You can change the 'kernel' parameter to 'rbf' or other options for non-linear SV
# Fit the model to the training data
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)


# In[66]:


get_ipython().system('pip install tensorflow')


# In[68]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Assuming you have a dataset loaded into a DataFrame called 'data'
# 'X' represents the features, and 'y' represents the target variable
# Split the dataset into training and testing sets
X = data.drop(columns=['LUNG_CANCER']) # Replace 'target_column' with the actual name
y = data['LUNG_CANCER'] # Replace 'target_column' with the actual name of your target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the input features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
                                                    # Create a simple feedforward neural network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
# Make predictions on the test data
y_pred = (model.predict(X_test) > 0.5).astype(int)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)


# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Select the features you want to use for clustering
X = data[['SMOKING', 'LUNG_CANCER']].values # Adjust the feature selectio
# Choose the number of clusters (k) for k-means
k = 3 # Adjust the number of clusters as needed
# Create and fit the k-means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
# Get cluster labels for each data point
labels = kmeans.labels_
# Add the cluster labels to the DataFrame
data['Cluster'] = labels
# Visualize the clusters (for 2D data)
if X.shape[1] == 2:
 plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
 plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200
 plt.xlabel('Feature 1')
 plt.ylabel('Feature 2')
 plt.title('K-Means Clustering')
 plt.legend()
 plt.show()
# You can further analyze the clusters based on the assigned cluster labels


# In[ ]:




