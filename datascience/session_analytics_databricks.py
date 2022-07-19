#!/usr/bin/env python
# coding: utf-8

# In[400]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[272]:


def load_data():   #load the data 
    df_all = pd.read_csv('sessionsDetails_2022-6-27-1656313280070.csv')
    # Take a subset
    return df_all.loc[:, ["Session Identifier","Searches","Clicks","Cases Logged",'Support Visit','Session Start Time','Session last Activity Time',"Activity Type","Activity Detail","Internal/External","Activity Additional Detail","Search Result Count","Activity Time"]]
data = load_data()


# In[273]:


data.head()


# In[274]:


data.info()


# In[275]:


data.isnull().sum()


# In[276]:


data.describe()


# In[376]:


data.nunique()


# In[377]:


data.columns


# # TOP TEN WEBSITES VISITED

# In[436]:


data['Activity Additional Detail'].value_counts()[:10].plot(kind='barh')


# # Websites Visited grouped by sessions

# In[504]:


data_si = data.groupby(['Session Identifier'],sort=True).agg({'Activity Additional Detail': [list,'count']})
data_si


# # 
# ##Data Analysis for each feature
# 
# 

# In[278]:


data['Searches'].value_counts()  #determines number of searches


# In[279]:


data['Clicks'].value_counts() 


# In[280]:


data['Cases Logged'].value_counts()


# In[281]:


data['Activity Type'].value_counts()


# In[282]:


data['Internal/External'].value_counts().plot(kind='barh')


# In[374]:


fig2 = px.histogram(data,x ='Support Visit',color='Support Visit')
fig2.show()


# In[373]:


fig1 = px.histogram(data,x ='Activity Type',color='Activity Type')
fig1.show()


# In[420]:


#function that to get a list of the sites by mapping the key to the webpage
def get_site_name(site_key, site_dictionary):
    site_list = []
    for i in range(len(site_key)):
        for key, value in site_dictionary.items():
            if value == site_key[i]:
                site_list.append(key)
    return site_list


# In[ ]:





# # DIFFERENCE in START AND END TIME 

# In[305]:


data['Session Start Time'] = pd.to_datetime(data['Session Start Time'], errors='coerce')
data['Session last Activity Time'] = pd.to_datetime(data['Session last Activity Time'], errors='coerce')


# In[306]:


data.info()


# In[309]:


data['difftime'] = data['Session last Activity Time'] - data['Session Start Time'] 


# In[312]:


data['difftime_seconds'] = (data['Session last Activity Time'] - data['Session Start Time']).dt.total_seconds()


# # GROUPING AND AGGREGATING
# https://stackoverflow.com/questions/57893148/how-to-aggregate-dataframe-by-user-with-session-scope-complexity

# In[346]:


data6 = data.groupby(['Session Identifier']).agg({'difftime_seconds':'max','Searches': 'max','Clicks':'max', 'Cases Logged':'max','Support Visit':lambda x: ' '.join(set(x)),'Activity Type':lambda x: ' '.join(set(x)),'Activity Detail':lambda x: ' '.join(set(x))})


# In[347]:


data6.head()


# # CASES LOGGED WITH/WITHOUT CLICKS
# 
# 

# In[349]:


data6[(data6['Cases Logged'] == 1) & (data6['Clicks'] == 0)]


# # SESSION WITH LESS THAN 10 minutes

# In[371]:


data6.loc[data6['difftime_seconds'] < 600]


# # PREPROCESSING OF THE DATA¶

# In[381]:


# Group by Session_ID
grouped_df = data.groupby('Session Identifier').agg({'Activity Detail':list})
 
# Join all queries in a cases
grouped_df['Activity Detail'] = grouped_df['Activity Detail'].apply(lambda x: ' '.join(x))
 
# Create a list of grouped queries
# for each title. 
queries = list(grouped_df['Activity Detail'])
queries


# In[382]:


vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(queries)
tf_idf_norm = normalize(X)
tf_idf_array = tf_idf_norm.toarray()
tfidf = pd.DataFrame(tf_idf_array, columns=vec.get_feature_names())
 


# In[384]:


# Run elbows in groups of 10
# Limit to a max of the total num cases
ks = [i * 10 if i*10 < len(grouped_df) else len(grouped_df) for i in range(1,15)]
sse = []
 
for k in ks:
    model =  KMeans(
        n_clusters=k, 
        init='k-means++',
        max_iter=100,
        n_init=1)
 
    model.fit(X)
 
    sse.append(model.inertia_)
     
# Plot ks vs SSE
plt.plot(ks, sse, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('Sum of Squares Errors (SSE)')
plt.title('Elbow method for optimal K')
plt.axvline(x=120,linestyle='--',c='grey')
plt.axvline(x=133,linestyle='--',c='grey')
plt.xticks(ks)
plt.show()


# # silhoutte_score
# #https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6

# In[403]:


range_n_clusters = [30, 40, 50, 60,70,80,90,100,120,133,135,140]
for n_clusters in range_n_clusters:
    #Initializing the clusterer with n_clusters value and a random   generator
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    #The silhouette_score gives the average value for all the   samples.
    #Calculating number of clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,"The average   silhoutte_score is :", silhouette_avg)
    #Using Silhouette Plot
    #visualizer = SilhouetteVisualizer(clusterer,colors =  'yellowbrick')
    #Fit the data to the visualizer
    #visualizer.fit(X)       
    #Render the figure
    #visualizer.show()


# # KMeans Clustering

# In[387]:


# define number of categories (clusters)
k = 50
 
# Instantiate the model
model = KMeans(
    n_clusters=k, 
    init='k-means++', 
    max_iter=100, 
    n_init=1,
random_state=42)
 
# Train the model on X
model.fit(X)


# In[405]:


# assign predicted clusters
labels = model.labels_
#print(len(labels)) 
# create a dataframe that contains 
# clusters matched to pages and their queries 
mapping = list(zip(labels, grouped_df.index, queries))
clusters = pd.DataFrame(mapping, columns=['cluster','Session Identifier','Activity Detail'])
clusters.head(10)


# In[410]:


# Dimensionality reduction
# Reduce features to 2D
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X.toarray())
 
# Reduce centroids to 2D
reduced_centroids = pca.transform(model.cluster_centers_)
 
print('Tf-IDF feature size: ', X.toarray().shape)
print('PCA feature size: ', reduced_features.shape)
print('Tf-IDF cluster centroids size: ', model.cluster_centers_.shape)
print('PCA centroids size: ', reduced_centroids.shape)


# In[411]:


pca_df_scale = pd.DataFrame(reduced_features, columns=['pc1','pc2'])
print(pca.explained_variance_ratio_)
pca_df_scale


# # FEATURE EXTRACTION

# In[412]:


def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vec.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs
dfs = get_top_features_cluster(tf_idf_array, predictions, 20)


# In[414]:


plt.figure(figsize=(8,6))
sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[1][:5])
#https://www.kaggle.com/code/dfoly1/k-means-clustering-from-scratch/notebook#Kmeans-Class


# Seaborn library to visualize our grouped texts in a very simple way.

# In[419]:


# Plot the individual groupings
# of cases_title

sns.scatterplot(
    x=reduced_features[:,0],
    y=reduced_features[:,1],
    hue=clusters['cluster'],
    palette='Set2')
 
# # plot the cluster centroids
# #for i in range(5):
# plt.scatter(
#     reduced_centroids[:, 0],
#     reduced_centroids[:,1],
#     marker='x',
#     s=30,
#     c='k'
#     )
 
# plot the graph
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)
#ax1.rcParams["figure.figsize"] = (20, 15)
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
plt.title('Potential Category groupings (PCA)')
plt.show()


# In[ ]:


#https://www.kaggle.com/code/leomauro/text-clustering-grouping-texts

