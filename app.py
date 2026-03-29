import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.title("Fraud Detection Dashboard")

file = st.file_uploader("Upload dataset (CSV / Excel / ZIP)")

df = None

if file is not None:
    if file.name.endswith(".zip"):
        with zipfile.ZipFile(file) as z:
            name = z.namelist()[0]
            df = pd.read_csv(z.open(name), encoding="latin1", engine="python")
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file, encoding="latin1", engine="python")
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head())

    if "Class" in df.columns:

        st.subheader("Fraud vs Legit Count")
        st.bar_chart(df["Class"].value_counts())

        st.subheader("Transaction Labels")
        st.write(df["Class"].map({0: "Legit Transaction", 1: "Fraud Transaction"}))

        st.subheader("Correlation Matrix (4x4)")
        numeric_df = df.select_dtypes(include=["number"]).iloc[:, :4]
        st.dataframe(numeric_df.corr())

        if "Amount" in df.columns:

            st.subheader("Amount Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Amount"], bins=30, ax=ax)
            st.pyplot(fig)

            st.subheader("Amount Comparison (Fraud vs Legit)")
            fig, ax = plt.subplots()
            sns.boxplot(x=df["Class"], y=df["Amount"], ax=ax)
            st.pyplot(fig)

            X = df[["Amount"]]
            y = df["Class"]

            model = RandomForestClassifier()
            model.fit(X, y)

            amount = st.number_input("Enter Amount", value=100.0)

            if st.button("Check Transaction"):
                pred = model.predict([[amount]])[0]

                if pred == 1:
                    st.error("Fraud Transaction")
                else:
                    st.success("Legit Transaction")
                  
