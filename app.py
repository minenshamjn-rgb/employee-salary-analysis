import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Salary Analysis",
    page_icon="💼",
    layout="wide"
)


# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #f8f9ff 0%,
        #eef3ff 50%,
        #f8f5ff 100%
    );
}

/* Main title */
.main-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #5B5FEF, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #60657a;
    margin-bottom: 30px;
}

/* Section title */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #292d45;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #e8e8f5;
    box-shadow: 0px 5px 20px rgba(80, 80, 150, 0.08);
}

[data-testid="stMetricLabel"] {
    font-size: 16px;
    color: #656a80;
}

[data-testid="stMetricValue"] {
    font-size: 32px;
    font-weight: 700;
    color: #5B5FEF;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #5B5FEF, #8B5CF6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: 600;
    font-size: 16px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17182d 0%, #27294b 100%);
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
}

/* Success box */
[data-testid="stAlert"] {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("Salary_Data.csv")

# Remove rows with missing values for modelling
model_df = df.dropna().copy()


# ---------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------

features = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience"
]

X = model_df[features]
y = model_df["Salary"]

categorical_features = [
    "Gender",
    "Education Level",
    "Job Title"
]

numeric_features = [
    "Age",
    "Years of Experience"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown("# 💼 SalaryScope")
st.sidebar.write("Employee Salary Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Analysis",
        "💰 Salary Predictor"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("### Project")
st.sidebar.caption("Factors Affecting Employee Salary")

st.sidebar.write("### Model")
st.sidebar.caption("Multiple Linear Regression")


# ---------------------------------------------------
# OVERVIEW PAGE
# ---------------------------------------------------

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">Factors Affecting Employee Salary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore how experience, education, age, job title and gender '
        'are associated with employee salary.'
        '</div>',
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Employees",
        f"{len(df):,}"
    )

    col2.metric(
        "💵 Average Salary",
        f"{df['Salary'].mean():,.0f}"
    )

    col3.metric(
        "🏆 Highest Salary",
        f"{df['Salary'].max():,.0f}"
    )

    col4.metric(
        "📉 Lowest Salary",
        f"{df['Salary'].min():,.0f}"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # About project
    st.markdown(
        '<div class="section-title">📌 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        This project investigates the major factors associated with employee
        salary. The main variables studied are Age, Gender, Education Level,
        Job Title and Years of Experience.

        The project also uses Multiple Linear Regression to estimate salary
        based on employee characteristics.
        """
    )

    # Dataset
    st.markdown(
        '<div class="section-title">📁 Employee Salary Dataset</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=380
    )

    st.markdown(
        '<div class="section-title">📋 Dataset Information</div>',
        unsafe_allow_html=True
    )

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Rows",
        df.shape[0]
    )

    info2.metric(
        "Variables",
        df.shape[1]
    )

    info3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


# ---------------------------------------------------
# ANALYSIS PAGE
# ---------------------------------------------------

elif page == "📊 Analysis":

    st.markdown(
        '<div class="main-title">Salary Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Visual exploration of the factors related to employee salary.'
        '</div>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------
    # SALARY DISTRIBUTION
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">💵 Salary Distribution</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["Salary"].dropna(),
        bins=25,
        edgecolor="white"
    )

    ax.set_xlabel("Salary")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Distribution of Employee Salaries")

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        "This chart shows how employee salaries are distributed "
        "across the dataset."
    )


    # ------------------------------------------------
    # EXPERIENCE VS SALARY
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Experience vs Salary</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["Years of Experience"],
        df["Salary"],
        alpha=0.4
    )

    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Relationship Between Experience and Salary")

    plt.tight_layout()

    st.pyplot(fig)

    experience_corr = df[
        "Years of Experience"
    ].corr(df["Salary"])

    st.write(
        f"Correlation between experience and salary: "
        f"**{experience_corr:.3f}**"
    )


    # ------------------------------------------------
    # AGE VS SALARY
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">🎂 Age vs Salary</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["Age"],
        df["Salary"],
        alpha=0.4
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Salary")
    ax.set_title("Relationship Between Age and Salary")

    plt.tight_layout()

    st.pyplot(fig)

    age_corr = df["Age"].corr(df["Salary"])

    st.write(
        f"Correlation between age and salary: "
        f"**{age_corr:.3f}**"
    )


    # ------------------------------------------------
    # EDUCATION
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">🎓 Average Salary by Education</div>',
        unsafe_allow_html=True
    )

    education_salary = (
        df.groupby("Education Level")["Salary"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    education_salary.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Education Level")
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Education Level")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------------------------
    # GENDER
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">👥 Average Salary by Gender</div>',
        unsafe_allow_html=True
    )

    gender_salary = (
        df.groupby("Gender")["Salary"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    gender_salary.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Gender")
    ax.set_ylabel("Average Salary")
    ax.set_title("Average Salary by Gender")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------------------------
    # TOP JOB TITLES
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '💼 Top 10 Job Titles by Average Salary'
        '</div>',
        unsafe_allow_html=True
    )

    job_salary = (
        df.groupby("Job Title")["Salary"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    job_salary.plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Average Salary")
    ax.set_ylabel("Job Title")
    ax.set_title("Top 10 Highest Paying Job Titles")

    plt.tight_layout()

    st.pyplot(fig)


    # ------------------------------------------------
    # FACTOR RELATIONSHIP TABLE
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔎 Factors Most Related to Salary'
        '</div>',
        unsafe_allow_html=True
    )

    factor_summary = pd.DataFrame({
        "Factor": [
            "Years of Experience",
            "Age"
        ],
        "Correlation with Salary": [
            experience_corr,
            age_corr
        ]
    })

    factor_summary = factor_summary.sort_values(
        "Correlation with Salary",
        ascending=False
    )

    st.dataframe(
        factor_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Correlation is shown for numerical variables. "
        "Categorical variables such as education, gender and job title "
        "are explored using grouped salary comparisons."
    )


    # ------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------

    st.markdown(
        '<div class="section-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    model_col1, model_col2 = st.columns(2)

    model_col1.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    model_col2.metric(
        "Mean Absolute Error",
        f"{mae:,.0f}"
    )

    st.info(
        """
        The R² score shows how much of the variation in salary
        can be explained by the model.

        Mean Absolute Error shows the average difference between
        predicted salary and actual salary.
        """
    )


# ---------------------------------------------------
# SALARY PREDICTOR PAGE
# ---------------------------------------------------

elif page == "💰 Salary Predictor":

    st.markdown(
        '<div class="main-title">Employee Salary Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter employee details below and let the regression model '
        'estimate the expected salary.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🧑‍💼 Employee Details</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        education = st.selectbox(
            "Education Level",
            sorted(
                model_df[
                    "Education Level"
                ].unique()
            )
        )

        experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0
        )

    with right:

        gender = st.selectbox(
            "Gender",
            sorted(
                model_df[
                    "Gender"
                ].unique()
            )
        )

        job_title = st.selectbox(
            "Job Title",
            sorted(
                model_df[
                    "Job Title"
                ].unique()
            )
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "✨ Predict Salary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [job_title],
            "Years of Experience": [experience]
        })

        prediction = model.predict(
            input_data
        )[0]

        st.markdown("<br>", unsafe_allow_html=True)

        st.success(
            f"### 💰 Estimated Salary: {prediction:,.2f}"
        )

        st.markdown(
            "#### Prediction Summary"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Age",
            age
        )

        result_col2.metric(
            "Experience",
            f"{experience:.0f} years"
        )

        result_col3.metric(
            "Education",
            education
        )

        st.caption(
            "This estimate is generated using a Multiple Linear "
            "Regression model trained on the employee salary dataset."
        )