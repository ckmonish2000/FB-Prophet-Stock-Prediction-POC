import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import date


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


list_of_companies = ('DEEPAKNTR.NS','KPITTECH.NS','CDSL.NS','IDEA.NS')

st.title("Stock Prediciton")

selected_stock=st.selectbox('Pick A Stock',list_of_companies)

n_years = st.slider("years of prediciton",1,4)

period = n_years*365

@st.cache
def download_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load = st.text("loading data")
data = download_data(selected_stock)
data_load.text("loaded data....")

# data frame
st.title("Raw data")
st.write(data)

def plot_raw_data():
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="stock Open price "))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="stock Close price "))
    fig.layout.update(title_text="Time series",xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

# st.title("Training Data")
# st.write(df_train)

model =Prophet()
model.fit(df_train)

future= model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.title("Predicted data")
st.write(forecast)

st.write("Forecast")
fig1=plot_plotly(model,forecast)
st.plotly_chart(fig1)

st.write("seasonal Forecast ")
fig2= model.plot_components(forecast)
st.write(fig2)
