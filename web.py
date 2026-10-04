import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Spam Detector")

model = pickle.load(open("model.pkl", "rb"))

with st.container(border=True):
    st.title("SMS / Email Spam Detection System")
    st.write("Detects spam messages using machine learning and Streamlit.")
    st.caption("Classifies messages as Spam or Not Spam and shows prediction confidence.")

tab1, tab2, tab3 = st.tabs(["Single Message", "Batch Analysis", "Model Insights"])

examples = {
    "Write my own": "",
    "Spam example": "Congratulations! You won a free prize, click now to claim",
    "Normal example": "Hey, are we meeting for lunch tomorrow?",
}

with tab1:
    st.header("Analyze a message")
    choice = st.selectbox("Example", list(examples.keys()))
    msg = st.text_area("Message text", value=examples[choice],
                       placeholder="Example: You won a free prize, click now.", height=150)
    if st.button("Analyze message", type="primary", use_container_width=True):
        if msg.strip() == "":
            st.warning("Write a message first.")
        else:
            p = model.predict_proba([msg])[0]
            conf = round(max(p) * 100, 1)
            if p[1] > 0.5:
                st.error("Spam (" + str(conf) + "% confidence)")
            else:
                st.success("Not Spam (" + str(conf) + "% confidence)")
            st.progress(float(p[1]), text="Spam probability: " + str(round(p[1] * 100, 1)) + "%")

with tab2:
    st.header("Analyze many messages")
    st.write("Upload a CSV file. Messages should be in the first column.")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        texts = df.iloc[:, 0].astype(str)
        probs = model.predict_proba(texts)
        df["Result"] = ["Spam" if x[1] > 0.5 else "Not Spam" for x in probs]
        df["Spam %"] = [round(x[1] * 100, 1) for x in probs]
        c1, c2 = st.columns(2)
        c1.metric("Spam", int((df["Result"] == "Spam").sum()))
        c2.metric("Not Spam", int((df["Result"] == "Not Spam").sum()))
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results", df.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.header("Model insights")
    try:
        vec = model.steps[0][1]
        clf = model.steps[-1][1]
        words = vec.get_feature_names_out()
        coef = clf.coef_[0]
        top = coef.argsort()[-10:][::-1]
        st.subheader("Top words that signal spam")
        st.bar_chart(pd.DataFrame({"Weight": coef[top]}, index=words[top]))
    except Exception:
        st.info("Word insights are not available for this model.")

st.divider()
st.caption("Developed by Ram Kumar Yadav")