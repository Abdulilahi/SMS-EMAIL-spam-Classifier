import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tfind=pickle.load(open("/home/illahi/Desktop/shouldaddtogit/SMS_Spam_Classifier/vectorizer.pkl","rb"))
model=pickle.load(open("/home/illahi/Desktop/shouldaddtogit/SMS_Spam_Classifier/model.pkl","rb"))

def Transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if (i not in stopwords.words('english')) and (i not in string.punctuation):
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)



st.title("EMAIL/SMS SPAM CLASSIFIER")
input_sms=st.text_area("Enter the message: ")
if st.button("predict"):
    # step 1     preprocess
    transformed_sms=Transform_text(input_sms)
    # step 2     Vectorize
    vector_input=tfind.transform([transformed_sms])        #transform([transformed_sms])
    # step 3     Predict
    result=model.predict(vector_input)[0]
    # step 4     Display
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")