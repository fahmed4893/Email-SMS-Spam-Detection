
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import spacy
nlp= spacy.load("en_core_web_lg")
svm_model_load= pickle.load(open("spam-svm-model.sav","rb"))
def clean_text(x):
     
     x=x.lower()
     data=nlp(x)
     l=[]
     for tokens in data:

        if tokens.is_punct or tokens.is_stop:
            continue
    
        l.append(tokens.lemma_)
     return l
def spam_pred(input_text):
    df2=pd.DataFrame([input_text],columns=['text'])
    df2['clean_text']=df2['text'].apply(lambda z: clean_text(z))
    df2['wordvec']=df2['clean_text'].apply(lambda z: nlp(str(z)).vector)
    val=df2['wordvec'].values
    val_2d=np.stack(val)

    prediction=svm_model_load.predict(val_2d)
    print(prediction)
    if prediction[0]== 1:
        return "Spam"
        
    else:
        return "Not Spam"
def main():
    st.title("Email/SMS Spam Detection")
    text=st.text_input("Enter text")

    diag= ''

    if st.button('Spam Prediciton'):
        diag= spam_pred(text)

    st.success(diag)

if __name__=='__main__':
    main()    
