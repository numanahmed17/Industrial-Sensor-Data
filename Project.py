#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# The file task_data.csv contains an example data set that has been artificially
# generated. The set consists of 400 samples where for each sample there are 10
# different sensor readings available. The samples have been divided into two
# classes where the class label is either 1 or -1. The class labels define to what
# particular class a particular sample belongs.
# 
# Your task is to rank the sensors according to their importance/predictive power
# with respect to the class labels of the samples. Your solution should be a
# Python script or a Jupyter notebook file that generates a ranking of the sensors
# from the provided CSV file. The ranking should be in decreasing order where the
# first sensor is the most important one.

# In[1]:


#importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble.forest import RandomForestClassifier
#reading the CSV File
dataset = pd.read_csv(r'C:\Users\Numan Ahmed\Documents\German Project\data.csv')


# In[2]:


df = pd.DataFrame(dataset)


# In[3]:


df


# In[4]:


#selcting X and y for predicting the importance of features based on predictive power
drop_col = ['sample index','class_label']
X = df.drop(drop_col, axis = 1)
y = df.class_label


# In[5]:


X


# In[6]:


y


# In[7]:


sns.heatmap(X.assign(target = y).corr().round(2), annot = True)


# ##  *Univariate Selection*
# 
# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# 
# The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
# 
# The example below uses the chi-squared (chi²) statistical test for non-negative features to select 10 of the best features from the Mobile Price Range Prediction Dataset.

# In[8]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[9]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# ## *Feature Importance*
# 
# You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.

# In[10]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nsmallest(10).plot(kind='barh')
plt.show()


# ## XGBoost
# Using theBuilt-in XGBoost Feature Importance Plot

# In[11]:


from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)


# In[12]:


plot_importance(model)
pyplot.show()


# ## Model Based Ranking(Random Forest)
# 
# We can fit a classfier to each feature and rank the predictive power. This method selects the most powerful features individually but ignores the predictive power when features are combined.
# 
# Random Forest Classifier is used in this case because it is robust, nonlinear, and doesn't require scaling.

# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True, random_state=42)

scores = []
num_features = len(X.columns)
for i in range(num_features):
    col = X.columns[i]
    score = np.mean(cross_val_score(clf, X[col].values.reshape(-1,1), y, cv=10))
    scores.append((float(score)*100, col))

print(sorted(scores, reverse = True))


# In[14]:


data = pd.DataFrame(scores)


# In[15]:


scores


# In[16]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X,y)


# In[17]:


def print_best_worst (scores):
    scores = sorted(scores, reverse = True)
    
    print("The 5 Best features selected by this method are :")
    for i in range(5):
        print(scores[i][1])
    
    print ("The 5 Worst features selected by this method are :")
    for i in range(5):
        print(scores[len(scores)-1-i][1])


# In[18]:


scores = []
for i in range(num_features):
    scores.append((clf.feature_importances_[i],X.columns[i]))
        
print_best_worst(scores)


# ## Permutation Importance

# In[19]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(clf, random_state=1).fit(X, y)
eli5.show_weights(perm, feature_names = X.columns.tolist())


# In[20]:


def imp_fi(column_names, importances):
    fi = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df


# In[21]:


def var_imp_plot(imp_fi, title):
    imp_fi.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_fi, orient = 'h', color = 'red')        .set_title(title, fontsize = 20)


# In[22]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
rf.fit(X, y)


# In[23]:


from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(rf, X, y):
    return r2_score(y, rf.predict(X))

perm_imp_rfpimp = permutation_importances(rf, X, y, r2)


# In[24]:


from treeinterpreter import treeinterpreter as ti, utils

selected_rows = [0, 200]
selected_df = X.iloc[selected_rows,:].values
prediction, bias, contributions = ti.predict(rf, selected_df)

for i in range(len(selected_rows)):
    print("Row", selected_rows[i])
    print("Prediction:", prediction[i][0], 'Actual Value:', y[selected_rows[i]])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 X.columns), 
                             key=lambda x: -abs(x[0])):
        print(feature, round(c, 2))
    print("-"*20) 


# ## LIME
# Local Interpretable Model-agnostic Explanations is a technique explaining the predictions of any classifier/regressor in an interpretable and faithful manner. To do so, an explanation is obtained by locally approximating the selected model with an interpretable one (such as linear models with regularisation or decision trees). The interpretable models are trained on small perturbations (adding noise) of the original observation (row in case of tabular data), thus they only provide a good local approximation.

# In[25]:


import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X.values,
                                                   mode = 'regression',
                                                   feature_names = X.columns,
                                                   categorical_features = [3], 
                                                   categorical_names = ['CHAS'], 
                                                   discretize_continuous = True)
                                                   
np.random.seed(42)
exp = explainer.explain_instance(X.values[399], rf.predict, num_features = 10)
exp.show_in_notebook(show_all=False) #only the features used in the explanation are displayed

exp = explainer.explain_instance(X.values[399], rf.predict, num_features = 10)
exp.show_in_notebook(show_all=False)


# ## Conclusion
#           From the above we can conclude that sensor 6 is the most important feature of all by using algorithms like Random Forest,XGBoost,Lime,Feature Importance using ExtratreesClassifier.
#           sensor 8 is the second most important feature althought it is first in some we can say sensor 6 is the most important feature beacause it tops in most reliable algorithms and majority of the algos state sensor 6 is the most important feature

# ## References
# *https://www.kaggle.com/dkim1992/feature-selection-ranking
# *https://www.kaggle.com/paultimothymooney/feature-selection-with-permutation-importance
# *https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
