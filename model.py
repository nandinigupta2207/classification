import joblib
import pandas as pd
import pickle
df=pd.read_csv("C:\\Users\\Nandini Gupta\\Downloads\\ObesityDataSet_raw_and_data_sinthetic.csv")
print(df.head())
df_prep = df.copy()
# create dummy variables
df_prep = pd.get_dummies(df_prep,columns=["Gender","family_history_with_overweight","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"])
df_prep.head()
# split dataset in features and target variable
# Features
X = df_prep.drop(columns=["NObeyesdad"])
# Target variable
y = df_prep['NObeyesdad']
# import sklearn packages for data treatments
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
mm = MinMaxScaler()
X_train_mm_scaled = mm.fit_transform(X_train)
X_test_mm_scaled = mm.transform(X_test)
model=DecisionTreeClassifier()
clf_mm_scaled = model.fit(X_train_mm_scaled, y_train)
clf_scaled = model.fit(X_train_mm_scaled,y_train)
y_pred_mm_scaled = clf_scaled.predict(X_test_mm_scaled)

pickle.dump(model, open('rf_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(mm, open('scaler.pkl', 'wb'))



