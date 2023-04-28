import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------

df=pd.read_csv(r"https://raw.githubusercontent.com/nandinigupta2207/classification/main/ObesityDataSet_raw_and_data_sinthetic.csv")
df_prep = df.copy()

# create dummy variables
df_prep = pd.get_dummies(df_prep,columns=["Gender","family_history_with_overweight","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"])

# split dataset in features and target variable
# Features
X = df_prep.drop("NObeyesdad", axis = 1)

# Target variable
y = df_prep['NObeyesdad']

# import sklearn packages for data treatments
# Import train_test_split function

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

mm = MinMaxScaler()
X_train_mm_scaled = mm.fit_transform(X_train)
X_test_mm_scaled = mm.transform(X_test)
model=DecisionTreeClassifier()
clf_mm_scaled = model.fit(X_train_mm_scaled, y_train)
clf_scaled = model.fit(X_train_mm_scaled,y_train)
y_pred_mm_scaled = clf_scaled.predict(X_test_mm_scaled)
# -----------------------------------------------------------
st.set_page_config(page_title="Obesity Prediction", page_icon="🍕", layout="wide", initial_sidebar_state="expanded")
#st.title("Uncovering Hidden Relationships: Obesity, Lifestyle Expressions")
st.markdown("<h1 style='text-align: center; color: orange;'>Uncovering Hidden Relationships: Obesity, Lifestyle Expressions</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: orange;'>FIND YOUR WAY TO HEALTH</h1>", unsafe_allow_html=True)
st.header("LIFESTYLE CHOICES")
col1, col2 = st.columns(2)
inp=[]

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("food.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

with col1:
    # st.text("Sepal characteristics")
    gen = st.selectbox("Select your gender", options=["Male", "Female"])
    age = st.slider("Age", 100, 10)
    inp.append(age)
    height = st.slider("Select your height", 1.0, 2.0, step=0.01, format="%0.2f")
    inp.append(height)
    weight = st.slider("Select  your weight", 0.0, 300.0)
    inp.append(weight)
    fm = st.selectbox("Family history of obesity", options=["Yes", "No"])
    favc = st.selectbox("Frequent consumption of high caloric food ", options=["Yes", "No"])
    fcvc = st.slider("Frequency of consumption of vegetables", 1.0, 4.0, step=0.1, format="%0.2f")
    inp.append(fcvc)
    ncp = st.slider("Number of main meals", 1.0, 5.0, step=0.1)
    inp.append(ncp)

with col2:
    # st.text("Pepal characteristics")
    caec = st.selectbox("Consumption of food between meals", options=["Sometimes", "Frequently", "Always", "No"])
    smoke = st.selectbox("do you smoke", options=["Yes", "No"])
    ch20 = st.slider("Consumption of water daily(L)", 1.0, 4.0, step=0.1)
    inp.append(ch20)
    scc = st.selectbox("Calories consumption monitoring", options=["No", "Yes"])
    faf = st.slider("Physical activity frequency per day", 1.0, 3.0, step=0.5)
    inp.append(faf)
    tue = st.slider("Time using technology devices", 0.0, 12.0)
    inp.append(tue)
    calc = st.selectbox("Consumption of alcohol", options=['No', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox("Mode of transportation",
                          options=['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

if st.button("Predict type of obesity"):
    if gen == 'Female':
        inp.append(1)
        inp.append(0)
    elif gen == 'Male':
        inp.append(0)
        inp.append(1)
    #fm = int(input("Family history with obesity: yes(1), no(0)"))
    if fm == 'No':
        inp.append(1)
        inp.append(0)
    elif fm == 'Yes':
        inp.append(0)
        inp.append(1)
    #favc = int(input("Frequent consumption of high caloric food: yes(1), no(0)"))
    if favc == 'No':
        inp.append(1)
        inp.append(0)
    elif favc == 'Yes':
        inp.append(0)
        inp.append(1)
    #caec = int(input("Consumption of food between meals : Always(1),Frequently(2),Sometimes(3),No(4)"))
    if caec == 'Always':
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif caec == 'Frequently':
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif caec == 'Sometimes':
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    elif caec== 'No':
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    #smoke = int(input("Do you smoke: yes(1), no(0)"))
    if smoke == 'No':
        inp.append(1)
        inp.append(0)
    elif smoke == 'Yes':
        inp.append(0)
        inp.append(1)
    #scc = int(input("Do you monitor your calorie consumption: yes(1), no(0)"))
    if scc == 'No':
        inp.append(1)
        inp.append(0)
    elif scc == 'Yes':
        inp.append(0)
        inp.append(1)
    #calc = int(input("Consumption of alcohol: Always(1),Frequently(2),Sometimes(3),No(4)"))
    if calc == 'Always':
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif calc == 'Frequently':
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif calc == 'Sometimes':
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    elif calc == 'No':
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    #mtrans = int(input("What mode of transportation do you use: Automobile(1), Bike(2), Motorbike(3), Public Transport(4), Walking(5)"))
    if mtrans == 'Automobile':
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif mtrans == 'Bike':
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
        inp.append(0)
    elif mtrans == 'Motorbike':
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
        inp.append(0)
    elif mtrans == 'Public Transport':
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
        inp.append(0)
    elif mtrans =='Walking':
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(0)
        inp.append(1)
    
    st.write(input)

    
    input_arr = np.array(inp)
    #input_arr_scaled = mm.transform(input_arr)
    input_arr = input_arr.reshape(1, -1)

    # make prediction
    result = model.predict([inp])[0]
    #st.success(f'The obesity type is {result}')
    #st.write(f'The obesity type is {result}')
  
    if result=='Normal_Weight':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI between 18.5 and 24.9 ,Generally considered to have a lower risk of health problems compared to other BMI categories. However, even within the "normal weight" BMI range, individuals who have a high percentage of body fat or poor lifestyle habits may still be at risk for certain health issues.')
        st.write(f'Maintain a healthy lifestyle that includes a balanced diet, regular physical activity, adequate sleep, stress management, and avoiding smoking and excessive alcohol consumption. It is important to continue to monitor your weight and health to ensure that you maintain a healthy BMI.')
    if result=='Overweight_Level_I':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI between 25.0 and 29.9 ,Increased risk for health problems such as high blood pressure, high cholesterol, heart disease, stroke, type 2 diabetes, sleep apnea, and certain cancers.')
        st.write(f'Aim to achieve a modest weight loss of 5-10% of your body weight through a combination of a healthy diet and increased physical activity. Adopting healthy habits such as regular exercise, reducing calorie intake, avoiding processed and high-fat foods, and increasing intake of fruits, vegetables, and whole grains can help reduce the risk of health problems associated with excess weight.')
    if result=='Overweight_Level_II':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI between 30.0 and 34.9, Increased risk for the same health problems as those in Overweight Level I, but at a higher risk due to the increased amount of excess body fat.')
        st.write(f'Similar to Overweight Level I, aim to achieve a modest weight loss of 5-10% of your body weight through a combination of healthy eating and regular physical activity. Consulting with a healthcare professional or registered dietitian can help you develop a personalized weight loss plan and provide guidance and support along the way.')
    if result=='Obesity_Type_I':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI between 35.0 and 39.9 ,Increased risk for the same health problems as those in Overweight Level II, as well as other health problems such as gallbladder disease, osteoarthritis, and infertility.')
        st.write(f'A weight loss of 5-10% of your body weight can provide significant health benefits. However, for individuals with severe obesity, a greater amount of weight loss may be needed to improve health. Consultation with a healthcare professional or registered dietitian can help develop a personalized weight loss plan that includes a combination of a healthy diet, physical activity, and potentially other interventions such as medication or bariatric surgery.')
    if result=='Obesity_Type_II':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI between 40.0 and 44.9, Increased risk for the same health problems as those in Obesity Type I, but at a higher risk due to the increased amount of excess body fat.')
        st.write(f'A weight loss of 5-10% of your body weight can provide significant health benefits. However, for individuals with severe obesity, a greater amount of weight loss may be needed to improve health. Consultation with a healthcare professional or registered dietitian can help develop a personalized weight loss plan that includes a combination of a healthy diet, physical activity, and potentially other interventions such as medication or bariatric surgery.')
    if result=='Obesity_Type_III':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI 45 or above, Increased risk for the same health problems as those in Obesity Type II, but at a higher risk due to the significantly higher amount of excess body fat.')
        st.write(f'A weight loss of 5-10% of your body weight can provide significant health benefits. However, for individuals with severe obesity, a greater amount of weight loss may be needed to improve health. Consultation with a healthcare professional or registered dietitian can help develop a personalized weight loss plan that includes a combination of a healthy diet, physical activity, and potentially other interventions such as medication or bariatric surgery.')
    if result=='Insufficient_Weight':
        st.success(f'The obesity type is {result}')
        st.write(f'The obesity type is {result}')
        st.write(f'BMI below 18.5,  Increased risk for malnutrition, weakened immune system, anemia, osteoporosis, and other health problems related to being underweight.')
        st.write(f'It is important to consult with a healthcare professional or registered dietitian to determine the underlying cause of insufficient weight and develop a personalized plan to achieve a healthy weight. This may include increasing calorie and nutrient intake through a balanced diet, incorporating strength training to build muscle mass, and addressing any underlying medical conditions.')
