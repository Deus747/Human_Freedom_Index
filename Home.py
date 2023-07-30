import streamlit as st 
from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Human Freedom Index"
)
add_logo(
    "Logo.png"
)
st.title("About")
# if st.sidebar.button("Go To"):
#     switch_page("test")
st.write("Welcome to our website dedicated to visualizing data on the Human Freedom Index and Terrorism Attacks! At our core, we are passionate about analyzing vast amounts of data to uncover meaningful insights that can drive positive change in the world. Our mission is to present complex information in a clear and visually appealing manner, allowing individuals, researchers, and policymakers to explore and understand the relationship between human freedom and terrorism."
        "The Human Freedom Index is a comprehensive measurement of personal, civil, and economic freedoms in countries around the globe. We believe that understanding and promoting freedom is essential for fostering peace, prosperity, and overall well-being in societies. On the other hand, terrorism remains a significant global challenge, with its profound impact on security, stability, and human rights. By juxtaposing these two critical datasets, we aim to shed light on potential connections and patterns that may exist.")


col1, col2 = st.columns([5,1])

with col2 :
    next_page = st.button("Next Page")
    if next_page : 
        switch_page("data visualization")
    