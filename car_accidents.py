# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

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









"""
df2 = pd.DataFrame({ 'D' : np.array([3] * 4+[2]*5+1*[6],dtype='int32')})

time_of_day  = data['Day_of_Week'].value_counts().sort_index()
plot = time_of_day.plot('bar', figsize=(20,8),color='blue')
plot.set_xlabel("Day of Week")
plot.set_ylabel("No of Accidents")
"""

def find_all_values(data, attribute, severity):
    res = {}
    data = data[data.Accident_Severity==severity]
    data = data[[attribute]]
    res = list(data.values)
    result = {}
    res_set = set(res)
    for category in res_set:
        result[category] = res.count(category)
    return sorted([result.items()], key=lambda x:x[0])
    
    
class_names = ['Fatal', 'Nonfatal']#, 'Slight']
data = pd.read_csv("accidents_2012_to_2014.csv")
attributes = [ "Accident_Index", "Accident_Severity", "Speed_limit", "Urban_or_Rural_Area", "Time", "Road_Type", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions", 'Special_Conditions_at_Site', 'Carriageway_Hazards']

data = data[attributes]

# find_all_values(data, 1, 'Accident_Severity')
# data = data.sample(1000)

def normalise_label(x):
    if x>2: return 0
    return 1

data = data[data.Accident_Severity!=2]
data['Accident_Severity'] = data['Accident_Severity'].apply(normalise_label)



def max_val(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]





def preprocessing(data):
    #Drop useless columns and nan values
    # data.drop(['Police_Force', 'Junction_Detail', 'Junction_Control', 'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location', 'Local_Authority_(District)', 'Local_Authority_(Highway)'], axis=1, inplace=True)
    data.dropna(inplace=True)
    
    #Drop rows with 'Unknown' values
    data = data[data.Weather_Conditions!='Unknown']
    data = data[data.Road_Type!='Unknown']
    
    #Encode "String" Labels into "Int" Labels for easy training
    le = LabelEncoder()
    data["Pedestrian_Crossing-Physical_Facilities"]= le.fit_transform(data["Pedestrian_Crossing-Physical_Facilities"])
    data["Light_Conditions"]= le.fit_transform(data["Light_Conditions"])
    data["Weather_Conditions"] = le.fit_transform(data["Weather_Conditions"])
    data["Road_Surface_Conditions"] = le.fit_transform(data["Road_Surface_Conditions"])
    data["Pedestrian_Crossing-Human_Control"] = le.fit_transform(data["Pedestrian_Crossing-Human_Control"])
    data["Road_Type"] = le.fit_transform(data["Road_Type"])
    data["Carriageway_Hazards"] = le.fit_transform(data["Carriageway_Hazards"])
    data["Special_Conditions_at_Site"] = le.fit_transform(data["Special_Conditions_at_Site"])    
    
    
    
    #Converting Time into Int for easy training
    data["Time"]= data["Time"].astype(str)
    data['Time']=data['Time'].str.slice(0,2, 1)
    data["Time"]= data["Time"].astype(int)
    
    #Creating 3 additional columns, one each for each class we need to classify into
    # onehot = pd.get_dummies(data.Accident_Severity,prefix=['Severity'])
    # data["Fatal"] = onehot["['Severity']_1"]
    # data["Nonfatal"] = onehot["['Severity']_2"]
    # data["Slight"] = onehot["['Severity']_3"]
    
    #Finally splitting the data into train and test
    train,test = train_test_split(data,test_size=.25,stratify=data["Accident_Severity"])
    
    return (train,test)
"""
def undersample(data):
    # UNDERSAMPLING (only in training)
    SCALE = 10
    num_fatal = len(data[data['Accident_Severity'] == 1])
    
    non_fatal_indices = data[data.Accident_Severity == 0].index
    random_indices = np.random.choice(non_fatal_indices,num_fatal*SCALE, replace=False)
    fatal_indices = data[data.Accident_Severity == 1].index
    under_sample_indices = np.concatenate([fatal_indices,random_indices])
    data = data.loc[under_sample_indices]
    return data
"""

train,test = preprocessing(data)
# train = undersample(train)


attributes.remove("Accident_Index")
attributes.remove("Accident_Severity")

train_features = train[attributes]
test_features =test[attributes]
# "Number_of_Casualties", "Number_of_Vehicles"
# train_features = train[["Number_of_Vehicles","Number_of_Casualties", "Day_of_Week", "Time", "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Year", "Urban_or_Rural_Area"]]
# test_features =test[["Number_of_Vehicles","Number_of_Casualties", "Day_of_Week", "Time", "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Year", "Urban_or_Rural_Area"]]


"""
from sklearn.ensemble import ExtraTreesClassifier

# feature extraction
for class_name in class_names:
    train_target = train[class_name]
    model = ExtraTreesClassifier()
    model.fit(train_features, train_target)
    print(model.feature_importances_)
"""




# rus = RandomUnderSampler(random_state=0, ratio={0: 50000, 1: 5303})
# rus.fit(train_features, train_target)
# train_features, train_target = rus.sample(train_features, train_target)




scores=[]
acc_score=[]
submission = pd.DataFrame.from_dict({'Accident_Index': test['Accident_Index']})




classifier1 = XGBClassifier(max_depth=4, learning_rate=0.2, n_estimators=600, silent=True,
                    subsample = 0.8,
                    gamma=0.5,
                    min_child_weight=10,
                    objective='binary:logistic',
                    colsample_bytree = 0.6,
                    max_delta_step = 1,
                    nthreads=1,
                    n_jobs=1)


# classifier = BernoulliNB() #54.7
# classifier = MultinomialNB() #61.0
# classifier = RandomForestClassifier(n_estimators=200)
# classifier = DecisionTreeClassifier()
# classifier = LogisticRegression() # zero fscore

# classifier = SVC(kernel="linear", probability=True)
classifier = EasyEnsembleClassifier(n_estimators=12, base_estimator=classifier)



class_name = 'Accident_Severity'
train_target = train[class_name]


def ml(classifier):
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='f1'))
    scores.append(cv_score)
    print('CV f1 score is {}'.format(cv_score))
    
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    acc = roc_auc_score(test[class_name],submission[class_name])
    acc_score.append(acc)
    print('Area under ROC is {}'.format(acc))
    
    
    
    def get_PRF(submission, test, threshold):
        submission = [1 if entry>threshold else 0 for entry in submission]
        tp = len([x for x, y in zip(submission, test) if x==y and x==1])
        fp = len([x for x, y in zip(submission, test) if x==1 and y==0])
        tn = len([x for x, y in zip(submission, test) if x==0 and y==0])
        fn = len([x for x, y in zip(submission, test) if x==0 and y==1])
        tp+=0.01
        fp+=0.01
        tn+=0.01
        fn+=0.01
        pre = 1.*tp/(tp+fp)
        rec = 1.*tp/(tp+fn)
        fscore = 2*pre*rec/(pre+rec)
        acc = 1.*(tp+tn)/(tp+tn+fp+fn)
        return [round(x, 5) for x in [pre,rec,fscore,acc]]
        
    print get_PRF(list(submission[class_name]), list(test[class_name]),0.5)


for classifier in [LogisticRegression()]:#SVC(kernel="linear", probability=True), DecisionTreeClassifier(), MultinomialNB(),classifier1]:
    classifier2 = EasyEnsembleClassifier(n_estimators=12, base_estimator=classifier)
    # ml(classifier)
    ml(classifier2)
    print ""
    
"""
import sklearn

import csv


attributes = [[] for x in range(33)]

# Accident_Index,Location_Easting_OSGR,Location_Northing_OSGR,Longitude,Latitude,Police_Force,Accident_Severity,Number_of_Vehicles,Number_of_Casualties,Date,Day_of_Week,Time,Local_Authority_(District),Local_Authority_(Highway),1st_Road_Class,1st_Road_Number,Road_Type,Speed_limit,Junction_Detail,Junction_Control,2nd_Road_Class,2nd_Road_Number,Pedestrian_Crossing-Human_Control,Pedestrian_Crossing

raw_data = []
with open('accidents_2012_to_2014.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            raw_data.append(row)
            for index, element in enumerate(row):
                try:
                    attributes[index].append(element)
                except:
                    attributes[index] = [element]
            line_count += 1
        if line_count>10000: break
    print('Processed {line_count} lines.')
    


def get_count_map(l):
    res = {}
    for ll in set(l):
        res[ll] = l.count(ll)
    return sorted(res.items(), key = lambda x:x[1], reverse = True)

raw_sample = raw_data[:1000]

attribute_counts = [get_count_map(attribute) for attribute in attributes]


def convert_categorical_feature_vector_to_input_matrix(column):
    
    return


attribute_sets = [list(set([x for x,y in attr])) for attr in attribute_counts]

"""


"""
0 Accident_Index (Unique ID)
 Location_Easting_OSGR (Local British coordinates x-value)
 Location_Northing_OSGR (Local British coordinates y-value)
 Longitude
 Latitude
 Police_Force
 Accident_Severity (1 = Fatal, 2 = Serious, 3 = Slight)
 Number_of_Vehicles
 Number_of_Casualties
9 Date (In dd/mm/yyyy format)
 Day_of_Week (Numeric: 1 for Sunday, 2 for Monday, and so on.)
 Time (Time the accident was reported, in UTC+0)
 Local_Authority_(District)
13 Local_Authority_(Highway)
 1st_Road_Class (This field is only used for junctions)
 1st_Road_Number (This field is only used for junctions)
16 Road_Type (Some options are Roundabout, One Way, Dual Carriageway, Single
Carriageway, Slip Road, Unknown)
 Speed_limit
 Junction_Detail (Some options are Crossroads, Roundabouts, Private Roads, Not a
Junction.)
 Junction_Control (A person, a type of sign, automated, etc.)
 2nd_Road_Class (This field is only used for junctions)
 2nd_Road_Number (This field is only used for junctions)
 Pedestrian_Crossing-Human_Control (Was there a human controller and what
type?)
 Pedestrian_Crossing-Physical_Facilities (Was it a zebra crossing, or bridge, or
another type?)
24 Light_Conditions (Day, night, street lights or not.)
 Weather_Conditions (Wind, rain, snow, fog)
 Road_Surface_Conditions (Wet, snow, ice, flood)
 Special_Conditions_at_Site (Was anything broken or defective, e.g. an obscured
sign?)
 Carriageway_Hazards (Was something in the way, e.g. a pedestrian, another
accident, something in the road?)
 Urban_or_Rural_Area
 Did_Police_Officer_Attend_Scene_of_Accident
 LSOA_of_Accident_Location
32 Year
"""
