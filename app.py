import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Marks Prediction")
st.write(
    "Predict student marks using study hours and attendance."
)

# Sample dataset from the project
data = pd.DataFrame({
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 72, 78, 82, 88, 90],
    "Marks": [35, 40, 45, 52, 58, 65, 70, 78, 85, 90]
})

# Train the model
X = data[["Study_Hours", "Attendance"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Student Details")

study_hours = st.slider(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

attendance = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

if st.button("Predict Marks"):
    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance]
    })

    prediction = model.predict(student)[0]

    st.success(f"Predicted Marks: {prediction:.2f}")

    st.caption(
        "This is an educational demo using only 10 sample records. "
        "Predictions may not reflect real student performance."
    )

st.divider()
st.subheader("Sample Dataset")
st.dataframe(data, use_container_width=True)
