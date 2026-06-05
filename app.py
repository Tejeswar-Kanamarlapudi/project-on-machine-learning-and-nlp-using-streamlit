"""
app.py

Streamlit web application for your final project.

Flow:
1. UI collects:
   - Name
   - Age (sidebar slider)
   - Highest Degree (sidebar dropdown)
   - Graduation Marks (%)
   - Skills / Resume text (large text area)

2. On "Submit Application & Scan Profile":
   - Compute eligibility score out of 25 based on:
        * Degree
        * Marks
        * Age
   - If score < 15:
        * Show rejection message (no NLP / ML stage is run)
   - If score >= 15:
        * Run NLP classification pipeline on the raw text:
            - Clean text
            - TF-IDF transform
            - Use trained classifier to get probabilities
            - Extract top 3 job categories with confidence percentages
        * Display them in a dashboard-style output.
"""

import os
import re
import string

import joblib
import numpy as np
import streamlit as st


# ---------- Utility: text cleaning (must match training preprocessing) ----------

def clean_text(text: str) -> str:
    """
    Same basic preprocessing used during training:

    - Lowercase
    - Remove HTML tags
    - Remove URLs
    - Remove digits
    - Remove punctuation
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Utility: scoring engine (Phase A) ----------

def compute_score(degree: str, marks: float, age: int) -> int:
    """
    Implements the tiered scoring rules from your project plan.[file:1]

    Score components (max 25):

    - Degree:
        * B.Tech or M.Tech : +10 points
        * BSc or BCA       : +5 points
        * Other            : +0 points (can be extended)
    - Marks:
        * > 80%            : +10 points
        * 70% - 80%        : +8 points
        * 60% - 70%        : +5 points
        * < 60%            : +0 points
    - Age:
        * 21 - 28 (inclusive) : +5 points
    """
    total_points = 0

    # Degree evaluation
    degree_normalized = degree.strip().lower()
    if degree_normalized in ["b.tech", "btech", "m.tech", "mtech"]:
        total_points += 10
    elif degree_normalized in ["bsc", "b.sc", "bca"]:
        total_points += 5

    # Marks evaluation
    if marks > 80:
        total_points += 10
    elif 70 <= marks <= 80:
        total_points += 8
    elif 60 <= marks < 70:
        total_points += 5
    # else: +0

    # Age evaluation
    if 21 <= age <= 28:
        total_points += 5

    return total_points


# ---------- Load model artifacts once at startup ----------

@st.cache_resource
def load_artifacts():
    """
    Load the trained TF-IDF vectorizer, classifier, and label encoder from disk.

    This assumes train_model.py was run and the .pkl files
    were saved into a 'models' directory at the project root.
    """
    tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")
    clf_path = os.path.join("models", "classifier.pkl")
    le_path = os.path.join("models", "label_encoder.pkl")

    if not (os.path.exists(tfidf_path) and os.path.exists(clf_path) and os.path.exists(le_path)):
        raise FileNotFoundError(
            "Model artifacts not found. Run train_model.py first to generate tfidf_vectorizer.pkl, "
            "classifier.pkl, and label_encoder.pkl inside the 'models' folder."
        )

    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(clf_path)
    label_encoder = joblib.load(le_path)
    return tfidf, clf, label_encoder


tfidf_vectorizer, classifier, label_encoder = load_artifacts()


# ---------- NLP pipeline for classification (Phase C) ----------

def predict_top_n_categories(raw_text: str, top_n: int = 3):
    """
    Takes raw resume/skills text and returns top N (category, probability) pairs.

    - Cleans the text using clean_text()
    - Transforms with TF-IDF vectorizer
    - Uses classifier.predict_proba() to get probabilities
    - Maps indices to class names via label_encoder
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return []

    X_vec = tfidf_vectorizer.transform([cleaned])

    # Ensure classifier supports predict_proba (e.g., LogisticRegression).
    if not hasattr(classifier, "predict_proba"):
        # If not, we can fallback to decision_function or just return top class.
        preds = classifier.predict(X_vec)
        classes = label_encoder.inverse_transform(preds)
        return [(classes[0], 1.0)]

    proba = classifier.predict_proba(X_vec)[0]  # shape: (n_classes,)

    # Get top N indices sorted by probability
    top_indices = np.argsort(proba)[::-1][:top_n]

    results = []
    for idx in top_indices:
        category_name = label_encoder.inverse_transform([idx])[0]
        confidence = proba[idx]
        results.append((category_name, float(confidence)))
    return results


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(
        page_title="AI Resume Screening & Profile Scanner",
        layout="wide",
    )

    st.title("AI Resume Screening & Profile Scanner")

    st.markdown(
        """
        This app simulates an intelligent HR screening workflow:

        1. **Eligibility Check (Tiered Scoring Engine)** – based on Degree, Marks, and Age.  
        2. **AI Profile Matching (NLP Classification)** – if eligible, your skills text is analyzed and matched to likely job tracks.
        """
    )

    # --------- Input Section ---------
    # Use two columns for main form and explanation
    col_form, col_info = st.columns([2, 1])

    with col_form:
        st.subheader("Candidate Information")

        # Name input
        name = st.text_input("Full Name", placeholder="Enter your name")

        # Sidebar inputs for age, degree, marks
        st.sidebar.header("Profile Inputs")

        age = st.sidebar.slider(
            "Age",
            min_value=18,
            max_value=60,
            value=22,
            help="Recommended range for this demo is 21–28."
        )

        degree = st.sidebar.selectbox(
            "Highest Degree",
            options=["B.Tech", "M.Tech", "BSc", "BCA", "Other"],
            index=0,
        )

        marks = st.sidebar.slider(
            "Graduation Percentage (%)",
            min_value=40.0,
            max_value=100.0,
            value=72.5,
            step=0.5,
        )

        # Large text area for raw skills / resume text
        skills_text = st.text_area(
            "Paste Your Skills / Resume Text",
            height=250,
            placeholder=(
                "Example: Proficient in Python, SQL, building dashboards, and running ML models "
                "like regression, random forest, etc..."
            ),
        )

        submit = st.button("Submit Application & Scan Profile")

    with col_info:
        st.subheader("How this works")
        st.markdown(
            """
            **Phase A – Tiered Scoring Engine**[file:1]  
            - Degree: B.Tech/M.Tech scores higher than BSc/BCA.  
            - Marks: Higher percentages give more points.  
            - Age: 21–28 gets a bonus.

            **Phase B – Conditional Branching**[file:1]  
            - If your total score is below the threshold, the ML model is not used.  
            - If you pass, we unlock the NLP pipeline.

            **Phase C – NLP Classification**[file:1]  
            - Your skills text is cleaned, vectorized with TF‑IDF, and fed into the trained classifier.  
            - We show the top 3 matching job categories with confidence.
            """
        )

    st.markdown("---")

    # --------- On submit: run scoring + branching + NLP ---------
    if submit:
        if not name:
            st.warning("Please enter your name before submitting.")
            return

        if not skills_text.strip():
            st.warning("Please paste your Skills / Resume text before submitting.")
            return

        # Phase A: compute score
        score = compute_score(degree=degree, marks=marks, age=age)
        max_score = 25

        st.subheader("Eligibility Result")

        st.write(f"**Candidate:** {name}")
        st.write(f"**Profile Evaluation Score:** {score} / {max_score}")

        if score < 15:
            # BRANCH 1 (FAIL) – do not wake up ML model
            st.error(
                f"Application Status: **Denied**\n\n"
                f"Hello {name}, thank you for your interest. Unfortunately, your profile "
                f"evaluation score ({score} / {max_score}) does not meet the configured "
                f"cut-off threshold for this hiring drive.\n\n"
                "Tip: Keep upskilling, improve your marks/experience, and try again next quarter!"
            )
            # Stop further processing
            return

        # BRANCH 2 (PASS) – unlock NLP pipeline
        st.success(
            f"Application Status: **Verified & Eligible!** 🎉\n\n"
            f"Congratulations {name}! Your profile evaluation score of {score} / {max_score} "
            f"passes the eligibility criteria."
        )

        # Phase C: NLP classification pipeline
        st.subheader("AI Profile Matching Analysis")

        with st.spinner("Analyzing your skills text with the trained NLP model..."):
            top_matches = predict_top_n_categories(skills_text, top_n=3)

        if not top_matches:
            st.warning("Could not extract meaningful text from your input. Please try with more detailed skills.")
            return

        # Display top 3 categories with progress bars
        for category, confidence in top_matches:
            percent = round(confidence * 100, 2)
            st.write(f"**{category}** – {percent}% match confidence")
            st.progress(int(percent))

        st.markdown(
            """
            **Next Steps**  
            - In a real system, the top-matching track could be sent to an interview scheduling module.  
            - You can experiment by changing your skills text and seeing how the predicted categories change.
            """
        )


if __name__ == "__main__":
    main()