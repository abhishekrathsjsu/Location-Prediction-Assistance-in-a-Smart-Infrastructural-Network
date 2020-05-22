#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from scipy import interp
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

# plotly visualization library
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools

# classifier libraries
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression


# In[ ]:


#Wrangling Data
class DataWrangle:
    def __init__(self):# Extracting data for the model
        self.input_path = "dataset/MovementAAL_RSS_"
        self.target_path = "dataset/MovementAAL_target.csv"
        self.group_path = "groups/MovementAAL_DatasetGroup.csv"

    def load_data(self):# Loading 315 individual files and making small dataframes

        self.df_lst = []

        for i in range(1, 315):
            self.file_path = self.input_path + str(i) + ".csv"
            self.file_df = pd.read_csv(self.file_path, header=0).reset_index(drop=True)
            self.file_df["_id"] = i
            self.file_df = self.file_df.reset_index(drop=True)
            self.df_lst.append(self.file_df)

        # Constructing the final input dataframe by concatenating the small dataframes
        self.input_df = pd.concat(self.df_lst, sort=True)
        self.target_df = pd.read_csv(self.target_path)[" class_label"]
        self.group_df = pd.read_csv(self.group_path)[" dataset_ID"]

    def add_target_class_and_time(self):

        # Adding target classes to the input dataframe
        self.group_lst = []
        # Adding time to the dataset at 8Hz sampling frequency
        for idx, (id_num, target, group) in enumerate(
            zip(self.input_df["_id"].unique(), self.target_df, self.group_df)
        ):
            self.gr = self.input_df[self.input_df["_id"] == id_num]
            self.gr = self.gr.reset_index(drop=True)
            self.gr["target"] = [target] * self.gr.shape[0]
            self.gr["group"] = [group] * self.gr.shape[0]
            self.gr["time"] = np.arange(0, self.gr.shape[0] / 8, 1 / 8)
            self.group_lst.append(self.gr)

        # Constructing the final dataframe by concatenating all the group dataframes
        self.df = pd.concat(self.group_lst)
        self.df = self.df.reset_index(drop=True)

        # Adding target label and group label - Movement and Non-Movement
        self.df["target_label"] = self.df["target"].apply(
            lambda x: "Movement" if x == 1 else "Non-Movement"
        )
        self.df["group_label"] = np.select(
            condlist=[self.df["group"] == 1,self.df["group"] == 2,self.df["group"] == 3,], 
            #Alloting datapoints to three envionments 
            choicelist=["environment_1", "environment_2", "environment_3"],
        )
    #Renaming and rearranging columns for better processing and understanding
    def rename_and_rearrange_columns(self):
        self.df = self.df.rename(
            columns={
                "#RSS_anchor1": "RSS_anchor1"," RSS_anchor2": "RSS_anchor2"," RSS_anchor3": "RSS_anchor3",
                " RSS_anchor4": "RSS_anchor4",
            }
        )

        self.df = self.df[
            [
                "_id","time","RSS_anchor1","RSS_anchor2","RSS_anchor3","RSS_anchor4","target","target_label",
                "group","group_label",
            ]
        ]

        # saving the preprocessed file
        self.df.to_csv("indoor_movement.csv", index=False)
        return self.df


if __name__ == "__main__":
    wrangle = DataWrangle()
    wrangle.load_data()
    wrangle.add_target_class_and_time()
    df = wrangle.rename_and_rearrange_columns()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


# Modified Dataset
df.head()


# In[ ]:


#Dimension Decomposition/ Feature Regularization
class DimensionDecomp:
    def __init__(self, df):
        self.input_cols = df[
            ["RSS_anchor1", "RSS_anchor2", "RSS_anchor3", "RSS_anchor4"]
        ]
        self.target_col = df[["target_label"]]
        self.target_group = df[["group_label"]]

    def pca_comp(self):
        self.pca = PCA(n_components=10, random_state=42)
        self.pca_res = self.pca.fit_transform(self.input_cols)
        self.pca_res = pd.DataFrame(self.pca_res, columns=["pc1", "pc2"])
        self.pca_res = pd.concat([self.pca_res, self.target_col, self.target_group], axis=1, sort=False)
        self.pca_res = self.pca_res.sample(n=1000, random_state=42)
        self.pca_res = self.pca_res.sort_values(by=["group_label"])

        return self.pca_res


# In[ ]:


new_df = df[df.columns.difference(["group_label","target_label"])]
new_df.head()


# In[ ]:


class BaseClassifier:
    def __init__(self, df):
        # selecting the input columns
        self.X = df[["RSS_anchor1", "RSS_anchor2", "RSS_anchor3", "RSS_anchor4"]]
        self.y = df["target"]

    def train_test_split(self):
        # splitting the datset into train and test subsets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
#    def clf_fit(self):
#        self.clf = LogisticRegression(random_state=0)
#        self.clf.fit(X_train, y_train)
#        self.score = self.clf.score(X_test, y_test)
#        print(
#            "validation score of baseline classifier: {}{}".format(
#                round(self.score * 100, 3), "%"
#            )
#       )
 #   def clf_fit(self):
 #       self.clf = NearestNeighbors(n_neighbors=10)
 #       self.clf.fit(X_train, y_train)
 #       self.score = self.clf.score(X_test, y_test)
 #       print(
 #           "validation score of baseline classifier: {}{}".format(
 #               round(self.score * 100, 3), "%"
 #           )
 #      )
  #  def clf_fit(self):
  #      self.clf = RandomForestClassifier(max_depth=2, random_state=0)
  #      self.clf.fit(X_train, y_train)
  #      self.score = self.clf.score(X_test, y_test)
  #      print(
  #          "validation score of baseline classifier: {}{}".format(
  #              round(self.score * 100, 3), "%"
  #          )
  #     )
   # def clf_fit(self):
   #     self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
   #     self.clf.fit(X_train, y_train)
   #     self.score = self.clf.score(X_test, y_test)
   #     print(
   #         "validation score of baseline classifier: {}{}".format(
   #             round(self.score * 100, 3), "%"
   #         )
   #    )
   # def clf_fit(self):
   #     self.clf = HistGradientBoostingClassifier()
   #     self.clf.fit(X_train, y_train)
   #     self.score = self.clf.score(X_test, y_test)
   #     print(
   #         "validation score of baseline classifier: {}{}".format(
   #             round(self.score * 100, 3), "%"
   #         )
   #    )
   # def clf_fit(self):
   #     self.clf = AdaBoostClassifier()
   #     self.clf.fit(X_train, y_train)
   #     self.score = self.clf.score(X_test, y_test)
   #     print(
   #         "validation score of baseline classifier: {}{}".format(
   #             round(self.score * 100, 3), "%"
   #         )
   #     )
   # def clf_fit(self):
   #     self.clf = DecisionTreeClassifier()
   #     self.clf.fit(X_train, y_train)
   #     self.score = self.clf.score(X_test, y_test)
   #     print(
   #         "validation score of baseline classifier: {}{}".format(
   #             round(self.score * 100, 3), "%"
   #         )
   #     )
    #def clf_fit(self):
    #    self.clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0).fit(X_train, y_train)
    #    self.clf.fit(X_train, y_train)
    #    self.score = self.clf.score(X_test, y_test)
    #    print(
    #        "validation score of baseline classifier: {}{}".format(
    #            round(self.score * 100, 3), "%"
    #        )
    #    )
    #Extra Tree Classifier
    def clf_fit(self):
        self.clf = ExtraTreesClassifier(n_estimators=100,n_jobs=-1,criterion='gini',random_state=42)
        self.clf.fit(X_train, y_train)
        self.score = self.clf.score(X_test, y_test)
        y_pred = self.clf.predict(X_test)
        print("validation score of baseline classifier: {}{}".format(round(self.score * 100, 3), "%"))#Validation Score
        print("Predictions ",y_pred) #Prediction using Extra Tree Classifier
        print("MSE ", mean_squared_error(y_pred,y_test)) #Calculate Mean Squared Error
        print(classification_report(y_test, y_pred)) #Classification report

if __name__ == "__main__":
    base_classifier = BaseClassifier(df)
    X_train, X_test, y_train, y_test = base_classifier.train_test_split()
    base_classifier.clf_fit()

