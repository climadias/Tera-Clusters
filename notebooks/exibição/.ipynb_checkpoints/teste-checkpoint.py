import streamlit as st
st.title("Hello World")
st.write("Pick an option")
keys = ["Normal","Uniform"]
dist_key = st.selectbox("Which Distribution do you want?",keys)

st.write("You have chosen {}".format(dist_key))
