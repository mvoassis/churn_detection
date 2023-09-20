import pandas as pd
import predict
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(layout='wide')

col01, col02 = st.columns(2)
with col01:
    st.title('CHURN PREDICTION')
    st.text('by: Marcos Oliveira (mvoassis@gmail.com)')

with col02:
    image = Image.open('src/app/img/logo.png')
    st.image(image)



dict_convert = {'Female': 0,
                'Male': 1,
                'Yes': 1,
                'No': 0,
                0: 'Not Churn',
                1: 'Churn'}


## Plots
def plot_gauge(prob) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': ("#FF7131" if prob < 50 else "#4c1854")},
            'steps': [
                {'range': [0, 25], 'color': "#f2f2f2"},
                {'range': [25, 50], 'color': "#f2f2f2"},
                {'range': [50, 75], 'color': "#f2f2f2"},
                {'range': [75, 100], 'color': "#f2f2f2"},
            ],
        }
    ))
    return fig


## Visualização no streamlit

aba1, aba2 = st.tabs(['Prediction', 'EDA Dataset'])

with aba1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        customer_gender = st.selectbox('Gender', ('Female', 'Male'))
        customer_SeniorCitizen = st.selectbox('Age >= 65?', ('Yes', 'No'))
        customer_Partner = st.selectbox('Have a Partner?', ('Yes', 'No'))
        customer_Dependents = st.selectbox('Have dependents?', ('Yes', 'No'))
        customer_tenure = st.number_input('Contract months', min_value=0, max_value=72)

    with col2:
        phone_PhoneService = st.selectbox('Have a telephone service subscription?', ('Yes', 'No'))
        phone_MultipleLines = st.selectbox('Have Multiple phone lines?', ('No', 'Yes'))
        internet_InternetService = st.selectbox('Have an Internet service subscription?', ('No', 'DSL', 'Fiber optic'))
        internet_OnlineSecurity = st.selectbox('Have an additional online security subscription?', ('No', 'Yes'))
        internet_OnlineBackup = st.selectbox('Have an additional online backup subscription?', ('No', 'Yes'))

    with col3:
        internet_DeviceProtection = st.selectbox('Have an additional Device protection subscription?', ('No', 'Yes'))
        internet_TechSupport = st.selectbox('Have an additional Tech support subscription?', ('No', 'Yes'))
        internet_StreamingTV = st.selectbox('Have an additional cable TV subscription?', ('No', 'Yes'))
        internet_StreamingMovies = st.selectbox('Have an additional Movie streaming subscription?', ('No', 'Yes'))

    with col4:
        account_Contract = st.selectbox('Contract type', ('Month-to-month', 'One year', 'Two year'))
        account_PaperlessBilling = st.selectbox('Paperless billing?', ('No', 'Yes'))
        account_PaymentMethod = st.selectbox('Payment method',
                                             ('Mailed check', 'Electronic check',
                                              'Credit card (automatic)', 'Bank transfer (automatic)'))
        account_Charges_Monthly = st.number_input('total of all customer services per month', 0, 120)

    if st.button('Predict'):
        classifier = predict.Classifier()

        data = [dict_convert[customer_gender], dict_convert[customer_SeniorCitizen],
                dict_convert[customer_Partner], dict_convert[customer_Dependents], customer_tenure,
                dict_convert[phone_PhoneService], dict_convert[phone_MultipleLines], internet_InternetService,
                dict_convert[internet_OnlineSecurity], dict_convert[internet_OnlineBackup],
                dict_convert[internet_DeviceProtection], dict_convert[internet_TechSupport],
                dict_convert[internet_StreamingTV], dict_convert[internet_StreamingMovies], account_Contract,
                dict_convert[account_PaperlessBilling], account_PaymentMethod,
                account_Charges_Monthly, customer_tenure * account_Charges_Monthly]

        pred, pred_proba = classifier.predict(data)

        col21, col22 = st.columns(2)
        with col21:
            st.header('RESULT:', divider='grey')
            if pred[0] == 0:
                # st.header(f'-> :blue[{dict_convert[pred[0]]}]') #
                # st.header(f'<span style="color:#4c1854">{dict_convert[pred[0]]}</span>')
                st.markdown(f'<h1><font color="#FF7131">{dict_convert[pred[0]]}</font></h1>',
                            unsafe_allow_html=True)
                st.markdown("When the churn classification result is negative, it signifies that a customer is less "
                            "likely to churn or terminate their association with a product or service provider. This "
                            "outcome is a positive signal for businesses, suggesting that the customer is satisfied "
                            "and engaged, with a reduced likelihood of discontinuing their subscription or contract. ")
            else:
                # st.header(f'-> :red[{dict_convert[pred[0]]}]')
                st.markdown(f'<h1><font color="#FF7131">{dict_convert[pred[0]]}</font></h1>',
                            unsafe_allow_html=True)
                st.markdown("When the churn classification result is positive, it indicates that a customer is "
                            "likely to churn or discontinue their relationship with a product or service provider. "
                            "A positive churn classification serves as a crucial alert for businesses, prompting "
                            "them to take proactive measures to retain the customer, such as targeted marketing "
                            "campaigns, personalized offers, or improved customer support, in order to mitigate "
                            "the risk of losing valuable clientele.")

        with col22:
            fig = plot_gauge(pred_proba[0][1]*100)
            st.plotly_chart(fig, use_container_width=True)

with aba2:
    data = pd.read_csv('https://raw.githubusercontent.com/mvoassis/churn_detection/main/data/data_clean_v5.csv')
    data2 = data.copy()

    coluna = st.selectbox('Select a Column', list(data2.columns.drop('Churn')))    
    fig = px.bar(data, x=coluna, y='Churn',
                 title=f'Barplot: Distribution of Churn per {coluna}')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Distribution of the numeric features:')

    col31, col32, col33 = st.columns(3)
    with col31:
        fig = px.box(data, x='Churn', y='account_Charges_Monthly',
                     title='Boxplot: Distribution of Churn per Monthly charges')
        st.plotly_chart(fig, use_container_width=True)
    with col32:
        fig = px.box(data, x='Churn', y='account_Charges_Total',
                     title='Boxplot: Distribution of Churn per Total charges')
        st.plotly_chart(fig, use_container_width=True)
    with col33:
        fig = px.box(data, x='Churn', y='customer_tenure',
                     title='Boxplot: Distribution of Churn per Tenure')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Distribution of the categorical features:')
    col41, col42, col43 = st.columns(3)
    with col41:
        fig = px.bar(data, x='Churn', y='account_PaymentMethod',
                     title='Barplot: Distribution of Churn per Payment Method')
        st.plotly_chart(fig, use_container_width=True)
    with col42:
        fig = px.bar(data, x='Churn', y='internet_InternetService',
                     title='Barplot: Distribution of Churn per Internet Service')
        st.plotly_chart(fig, use_container_width=True)
    with col43:
        fig = px.bar(data, x='Churn', y='account_Contract',
                     title='Barplot: Distribution of Churn per Contract type')
        st.plotly_chart(fig, use_container_width=True)

