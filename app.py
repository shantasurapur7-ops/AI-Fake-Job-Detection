import os
import joblib
import streamlit as st

MODEL_PATH = "models/fake_job_pipeline.joblib"

st.set_page_config(page_title="Fake Job & Recruitment Scam Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ AI-Powered Fake Job & Recruitment Scam Detector")
st.write("Screen a job advertisement or recruitment message for patterns associated with fraudulent postings.")

st.warning(
    "This tool provides an estimate, not proof. A message marked legitimate may still be fraudulent, "
    "and a suspicious message may be genuine. Always verify the employer independently."
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if pipeline is None:
    st.error("The trained model is not available yet. First run: python train_model.py")
    st.stop()

message = st.text_area(
    "Paste the job advertisement, email, or hiring message here:",
    height=220,
    placeholder="Example: We are hiring a junior analyst..."
)

def find_warning_signs(text):
    lower = text.lower()
    checks = {
        "Upfront payment or registration fee request": [
            "registration fee", "pay a fee", "upfront fee", "processing fee",
            "security deposit", "pay to secure", "payment to apply"
        ],
        "Unusually high or guaranteed earnings promise": [
            "guaranteed income", "earn huge", "earn money instantly",
            "unlimited salary", "very high salary", "daily income"
        ],
        "Strong pressure or urgency": [
            "act immediately", "urgent response", "limited slots",
            "apply now or lose", "within 1 hour", "immediate joining"
        ],
        "Hiring without a normal interview": [
            "no interview", "without interview", "job confirmed immediately"
        ],
        "Request for sensitive information": [
            "send aadhaar", "share pan", "bank account details",
            "otp", "password", "debit card"
        ],
    }
    found = []
    for label, phrases in checks.items():
        if any(phrase in lower for phrase in phrases):
            found.append(label)
    return found

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.info("Please enter a job advertisement or recruitment message.")
    elif len(message.strip().split()) < 5:
        st.info("This message is very short. Add the complete job details for a more useful screening estimate.")
    else:
        prediction = str(pipeline.predict([message])[0])
        st.subheader("Model screening result")
        if prediction.lower() in {"fraudulent", "fake", "1"}:
            st.error("The model classifies this message as resembling fraudulent job postings.")
        else:
            st.success("The model classifies this message as resembling legitimate job postings.")
        st.caption("This is the machine-learning model's prediction based on its training examples. It is not a verification.")

        st.subheader("Separate warning-sign checks")
        warnings = find_warning_signs(message)
        if warnings:
            for warning in warnings:
                st.write("⚠️ " + warning)
        else:
            st.write("No warning phrases from the current rule checklist were detected.")
        st.caption("These are transparent phrase checks, separate from the machine-learning prediction.")

        st.subheader("Safety checklist")
        st.markdown(
            "- Find the vacancy on the employer's official careers website.\n"
            "- Contact the company using contact details found independently.\n"
            "- Do not pay money to secure an interview or job.\n"
            "- Never share OTPs, passwords, or unnecessary banking/identity details.\n"
            "- Be cautious of pressure to act immediately."
        )

st.divider()
st.subheader("About this project")
st.write(
    "The prototype uses a TF-IDF text representation and a supervised scikit-learn classifier. "
    "Its usefulness depends on the quality, licensing, and representativeness of the training dataset."
)
