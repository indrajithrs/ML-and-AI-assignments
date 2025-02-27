
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load the Titanic dataset
titanic_deploy_data = pd.read_csv("titanic_deploydata.csv")

# Define target and features
target_column = ['Survived']
feature_columns = ['PassengerId', 'Pclass', 'Sex','Age','Fare','Embarked','SibSp','Parch']  # Specify features explicitly

# Separate features (X) and target (Y)
X = titanic_deploy_data[feature_columns]
Y = titanic_deploy_data[target_column]

# Train the model
model = LogisticRegression()
model.fit(X, Y)

# Streamlit app for predictions
st.title('Titanic Survival Prediction')

st.sidebar.header('Enter Passenger Details')

def user_input_features():
    PassengerId = st.sidebar.number_input("Enter the Passenger ID", min_value=1, step=1)
    Pclass = st.sidebar.selectbox("Enter the Passenger Class", ("1", "2", "3"))
    Sex = st.sidebar.selectbox('Sex (Male=1, Female=0)', ('1', '0'))
    Age = st.sidebar.number_input("Enter the Passenger Age", min_value=0.0, step=1.0)
    Fare = st.sidebar.number_input("Enter the Fare", min_value=0.0, step=1.0)
    Embarked = st.sidebar.selectbox("Enter the Embarked Place (Cherbourg=0,Queenstown=1,Southampton=2)",("0", "1", "2")),
    SibSp = st.sidebar.selectbox("Enter the number of siblings/spouses aboard",('0','1','2','3','4','5','8')),
    Parch = st.sidebar.selectbox("Enter the number of parents/children aboard",('0','1','2','3','4','5','6','9'))

    # Construct the feature DataFrame
    data = {
        'PassengerId': PassengerId,
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'Fare': Fare,
        'Embarked': Embarked,
        'SibSp': SibSp,
        'Parch': Parch
    }
    return pd.DataFrame(data, index=[0])

# Get user inputs
df = user_input_features()

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Display results
st.subheader('Survival Analysis')
st.write('Prediction: **Survived !** 🟢' if prediction[0] == 1 else 'Prediction: **Sorry! Not Survived** 🔴')
st.write(f"Probability of Survival: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Probability of Not Surviving: {prediction_proba[0][0]*100:.2f}%")