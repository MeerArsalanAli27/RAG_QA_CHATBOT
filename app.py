import streamlit as st


st.set_page_config(
    page_title="Ferrell Solutions | PDF Chat",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("### Ferrell Solutions")

page = st.navigation(
    [
        st.Page(
            "app_pages/introduction.py",
            title="Introduction",
            icon=":material/info:",
            url_path="introduction",
            help=(
                "Learn what the PDF chatbot does and how its retrieval-augmented "
                "generation workflow turns documents into answers."
            ),
        ),
        st.Page(
            "main.py",
            title="PDF Chat",
            icon=":material/forum:",
            url_path="chat",
            help=(
                "Upload one or more PDFs, process them, and ask questions "
                "grounded in their contents."
            ),
        ),
    ],
    position="top",
)

page.run()
