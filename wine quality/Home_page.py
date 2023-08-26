#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot  as plt
import seaborn as sns
#app heading
st.write("""
# Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

# st.subheader('User Input parameters')
# st.write(df)

#reading csv file
data=pd.read_csv("winequality-red.csv")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])
#random forest model
rfc= RandomForestClassifier()
rfc.fit(X, Y)
# st.subheader('Wine quality labels and their corresponding index number')



prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
st.subheader('Prediction')
emoji_unicode = "\U0001F44D"  # Unicode for thumbs-up emoji

if prediction==3 or  prediction==4:
    st.markdown(f"<h2 style='font-size: 65px; color:green;'>&#x1F44E; Bad Quality</h2>", unsafe_allow_html=True)
elif prediction==5 or prediction==6 or prediction==7:
    st.markdown(f"<h2 style='font-size: 65px; color:green;'>&#x1F641; Moderate Quality</h2>", unsafe_allow_html=True)

elif prediction==8:
    st.markdown(f"<h2 style='font-size: 65px; color:green;'>'{emoji_unicode} Good Quality'</h2>", unsafe_allow_html=True)
else:
    st.subheader("Nothing ")

st.subheader('Prediction Probability')
# st.write(prediction_proba)

# Set Seaborn style to dark background
sns.set(style="dark")

instance_index = 0
prediction_probabilities = prediction_proba[instance_index]
categories = ['Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7', 'Category 8']
labels = [f'{category}' for category, prob in zip(categories, prediction_probabilities)]
colors = ['#EB995E', '#F0D1AF', '#EAEDAB', '#67B5B4', '#5067BF', '#4160EF']

fig1, ax1 = plt.subplots()
ax1.pie(prediction_probabilities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
ax1.axis('equal')
st.pyplot(fig1)




hide_st_style="""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)