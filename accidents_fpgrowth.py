import numpy as np
import pandas as pd

#Visualisation Libraries
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from pandas.plotting import scatter_matrix

#Training and Preprocessing Libraries
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("accidents_2012_to_2014.csv")
attributes = [ "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area", 'Special_Conditions_at_Site', 'Carriageway_Hazards']
data = data[attributes]

c=0
for index, row in data.iterrows():
    if str(row["Speed_limit"])=="70" and str(row["Special_Conditions_at_Site"])=="Other object in carriageway":
    # if str(row["Speed_limit"])=="70" and str(row["Weather_Conditions"])=="Raining without high winds":
        c+=1

"""
# 'Carriageway_Hazards=None', '1st_Road_Class=-1', 'Special_Conditions_at_Site=None', 
itemsets = []
with open("fatal_frequent_itemsets.txt", "r") as lines:
    for line in lines:
        hold = line.strip().split(",")
        if len(hold)==2: continue
        itemsets.append(hold)
        continue
        a1=hold[0].split("=")[0]
        v1=hold[0].split("=")[1].replace('_',' ')
        a2=hold[1].split("=")[0]
        v2=hold[1].split("=")[1].replace('_',' ')
        # x = data[(str(data[a1])==v1) & (str(data[a2])==v2)]
        # new_data = data[[a1,a2]]
        c=0
        for index, row in data.iterrows():
            if str(row[a1])==v1 and str(row[a2])==v2:
                c+=1
            # if index==10000: break
        
        try:
            print hold[0]+","+hold[1]+","+str(1.0*int(hold[2])/c)
        except:
            continue


data = pd.read_csv("accidents_2012_to_2014.csv")
fatal_data = data[data.Accident_Severity==1]
attributes = [ "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area", 'Special_Conditions_at_Site', 'Carriageway_Hazards']
fatal_data = fatal_data[attributes]
item_list = {}
with open("fatal_itemsets.csv", "w") as of:
    for index, row in fatal_data.iterrows():
        res_string = ""
        for attribute in attributes[:-1]:
            item = attribute+"="+str(row[attribute]).replace(' ', '_')
            if item in item_set: continue

            res_string+=item+","
            # of.write(item+",")
        item = attribute+"="+str(row[attributes[-1]]).replace(' ', '_')
        if item in item_set: 
            of.write(res_string[:-1]+"\n")
            continue
        of.write(res_string+item+"\n")
"""
# item_list = sorted(item_list.items(),key=lambda x:x[1], reverse=True)[:20]

"""

def find_all_values(data, attribute, severity):
    res = {}
    # data = data[data.Accident_Severity==severity]
    fatal_data = data[data.Accident_Severity==1]
    res_all = list(data[attribute])
    
    res_fatal = list(fatal_data[attribute])
    # res = list(data.values.tolist())
    result = {}
    res_set = set(res_all)
    for category in res_set:
        total,fatal = res_all.count(category), res_fatal.count(category)
        result[attribute+"="+str(category)] = [round(100.0*fatal/(total*1.1417),2), fatal, total]
    
    res = sorted(result.items(), key=lambda x:x[1][0], reverse=True)
    # print res[0]
    return res


    
data = pd.read_csv("accidents_2012_to_2014.csv")
# attribute = 'Accident_Severity'
# severity=1 

attributes = [ "Accident_Severity", "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Year", "Urban_or_Rural_Area", 'Special_Conditions_at_Site', 'Carriageway_Hazards', '1st_Road_Class', '2nd_Road_Class',]


result = []
for attribute in attributes:
    result = result+find_all_values(data, attribute, 1)
result = sorted(result, key=lambda x:x[1][0], reverse=True)

for r in result:
    if r[0]=='nan': continue
    print("{0},{1},{2},{3}".format(r[0], r[1][0],r[1][1],r[1][2]))
print("")

"""
"""
for attribute in attributes:
    res = find_all_values(data, attribute, 1)
    print(attribute+",Fatality Rate,Fatal,Total")
    for r in res:
        if r[0]=='nan': continue
        print("{0},{1},{2},{3}".format(r[0], r[1][0],r[1][1],r[1][2]))
    print("")
"""    
    
    
