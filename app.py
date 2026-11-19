import os

import streamlit as st
from google import genai
from google.genai import errors


def build_prompt(
    current_headline: str,
    current_about: str,
    skills: str,
    target_role: str,
) -> str:
    return f"""You are a practical LinkedIn profile writing assistant for students.

Use the information below to create:
1. Exactly 3 distinct LinkedIn headline options.
2. One polished About section tailored to the target role.

Keep the writing specific, professional, and easy to scan. Use relevant
keywords naturally. Do not invent experience, qualifications, achievements,
or metrics that are not provided. If the student has limited experience,
emphasize skills, interests, and goals honestly.

Return only the requested content in this format:
## Headline options
1. ...
2. ...
3. ...

## About section
...

Current headline:
{current_headline}

Current About section:
{current_about}

Skills:
{skills}

Target role:
{target_role}
"""


def optimize_profile(
    api_key: str,
    model: str,
    current_headline: str,
    current_about: str,
    skills: str,
    target_role: str,
) -> str:
    prompt = build_prompt(
        current_headline=current_headline,
        current_about=current_about,
        skills=skills,
        target_role=target_role,
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    finally:
        client.close()

    if not response.text or not response.text.strip():
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return response.text.strip()


st.set_page_config(
    page_title="LinkedIn Profile Optimizer",
    page_icon="💼",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1050px; padding-top: 2.5rem;}
    h1 {letter-spacing: -0.04em;}
    div[data-testid="stForm"] {
        border: 1px solid #dce5ef;
        border-radius: 16px;
        padding: 1.5rem;
        background: #ffffff;
    }
    div.stButton > button[kind="primaryFormSubmit"] {
        border-radius: 10px;
        min-height: 2.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LinkedIn Profile Optimizer")
st.write(
    "Build a stronger student profile with three headline ideas and a "
    "role-focused About section."
)
st.link_button("Get a free Gemini API key", "https://aistudio.google.com/apikey")

api_key = st.text_input(
    "Gemini API key",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    help="Used for this session only. The app does not write your key to a file.",
)

with st.form("profile_form"):
    headline_column, role_column = st.columns(2)
    with headline_column:
        current_headline = st.text_input(
            "Current LinkedIn headline",
            placeholder="Student | Aspiring data analyst",
        )
    with role_column:
        target_role = st.text_input(
            "Target role",
            placeholder="Junior data analyst",
        )
    current_about = st.text_area(
        "Current About section",
        height=150,
        placeholder="Paste your current LinkedIn About section...",
    )
    skills = st.text_area(
        "Skills",
        height=100,
        placeholder="Python, data analysis, communication",
    )
    with st.expander("Advanced setting"):
        model = st.text_input(
            "Gemini model",
            value="gemini-2.5-flash",
            help="Use a model name available to your Gemini API key.",
        )
    submitted = st.form_submit_button(
            "Optimize my profile",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state.pop("optimized_profile", None)
    st.session_state.pop("original_profile", None)
    missing = [
        label
        for label, value in (
            ("current headline", current_headline),
            ("About section", current_about),
            ("skills", skills),
            ("target role", target_role),
            ("Gemini API key", api_key),
            ("Gemini model", model),
        )
        if not value.strip()
    ]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    else:
        try:
            with st.spinner("Generating profile suggestions..."):
                optimized_profile = optimize_profile(
                    api_key=api_key.strip(),
                    model=model.strip(),
                    current_headline=current_headline.strip(),
                    current_about=current_about.strip(),
                    skills=skills.strip(),
                    target_role=target_role.strip(),
                )
            st.session_state["optimized_profile"] = optimized_profile
            st.session_state["original_profile"] = {
                "headline": current_headline.strip(),
                "about": current_about.strip(),
                "skills": skills.strip(),
                "target_role": target_role.strip(),
            }
        except (errors.APIError, RuntimeError) as exc:
            st.error(f"Gemini could not generate suggestions: {exc}")

if "optimized_profile" in st.session_state:
    st.divider()
    st.subheader("Before and after")
    before, after = st.columns(2)

    with before:
        st.markdown("#### Before")
        st.markdown("**Headline**")
        st.write(st.session_state["original_profile"]["headline"])
        st.markdown("**About**")
        st.write(st.session_state["original_profile"]["about"])

    with after:
        st.markdown("#### Suggested profile")
        st.markdown(st.session_state["optimized_profile"])

    with st.expander("Skills and target role used"):
        st.markdown("**Skills**")
        st.write(st.session_state["original_profile"]["skills"])
        st.markdown("**Target role**")
        st.write(st.session_state["original_profile"]["target_role"])

    st.caption(
        "Review the suggestions before using them. Keep only claims that "
        "accurately reflect your experience."
    )
