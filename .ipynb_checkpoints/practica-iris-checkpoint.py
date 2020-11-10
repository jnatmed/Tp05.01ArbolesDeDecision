import pandas as pd 

# Lee el archivo
data = pd.read_csv("data\zoo.csv") 
# Preview the first 5 lines of the loaded data 
data.head()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for column_name in data.columns:
  if (data[column_name].dtype == object) & (column_name!='type'):
    data[column_name] = le.fit_transform(data[column_name])

data.head()