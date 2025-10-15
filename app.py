import streamlit as st
import pickle
import string
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



tfidf=pickle.load(open('Vectorizer_.pkl','rb'))
clf=pickle.load(open('Classifier.pkl','rb'))

# st.title('SMS Spam Classifier')

st.markdown(
        """
        <div style='text-align:center; margin-top:150px;'>
            <h1 style='font-size:3em; color:white;'>SMS Spam Classifier</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

input_msg=st.text_area("Enter your message below:", height=150, placeholder="Type or paste a message...")


if st.button('Classify this Message as Spam or Not Spam'):
    # Preprocessing
    def transform_text(text):
        # converting the data into lower case
        text = text.lower()

        # Tokenization
        text = nltk.word_tokenize(text)
        res = []
        # Removing StopWords from the data if present
        for i in text:
            if i.isalnum():
                res.append(i)
        text = res[:]
        res.clear()

        # removing all the stop words and punctuations
        from nltk.corpus import stopwords
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                res.append(i)
        text = res[:]
        res.clear()

        # stemming
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()

        for i in text:
            res.append(ps.stem(i))

        return " ".join(res)

    transformed_text=transform_text(input_msg)

    # Vectorization
    vector=tfidf.transform([transformed_text]).toarray()

    # Prediction

    prediction=clf.predict(vector)[0]

    if prediction==1:
        st.header('Spam Message Detected!')
    else:
        st.header('Message is Not Spam!')

