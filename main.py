import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def generate_house_data(n_sample=100):
    np.random.seed(50)
    size = np.random.normal(1400,50,n_sample)
    price = size*50 + np.random.normal(0,50,n_sample)
    return pd.DataFrame({"size":size, "price":price})

def train_model():
    df = generate_house_data(100)
    X = df[['size']]   # corrected
    y = df['price']
    model = LinearRegression()
    model.fit(X,y)
    return model

def main():
    st.title("House Price Prediction")

    model = train_model()

    size = st.number_input("House Size", min_value=500, max_value=5000, value=1000)

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])
        st.success(f"Estimated Price : ₹ {predicted_price[0]:,.2f}")

        df = generate_house_data()

        fig = px.scatter(df, x="size", y="price", title="Size vs House Price")
        fig.add_scatter(x=[size], y=[predicted_price[0]], mode='markers', marker=dict(size=15,color='red'), name="Prediction")

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
