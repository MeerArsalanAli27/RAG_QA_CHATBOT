import streamlit as st


st.title("A simple introduction to RAG", icon=":material/auto_awesome:")
st.caption(
    "How retrieval-augmented generation helps an AI answer questions using "
    "information from your own documents."
)

st.subheader("What does RAG mean?")
st.markdown(
    "**RAG** stands for **retrieval-augmented generation**. It is a way to "
    "connect a language model to a collection of reference material. Instead "
    "of relying only on what the model learned during training, a RAG system "
    "first searches that material for relevant passages and then gives those "
    "passages to the model as context for its answer."
)

with st.container(border=True):
    st.markdown(
        "### In one line\n"
        "**Your question → relevant information is retrieved → an answer is "
        "generated from that information.**"
    )

st.subheader("Why use RAG?")
benefits = st.columns(3, gap="medium")
with benefits[0]:
    with st.container(border=True):
        st.subheader("Use your own sources", icon=":material/library_books:")
        st.write(
            "Ask about PDFs and reference material that are specific to your "
            "task, rather than asking only general questions."
        )
with benefits[1]:
    with st.container(border=True):
        st.subheader("Find relevant context", icon=":material/manage_search:")
        st.write(
            "Search narrows a large collection to passages that are useful "
            "for the current question."
        )
with benefits[2]:
    with st.container(border=True):
        st.subheader("Ground the response", icon=":material/verified:")
        st.write(
            "The retrieved passages give the model evidence to use when "
            "formulating an answer. Retrieval can still miss information, so "
            "answers should be checked against the source."
        )

st.subheader("The RAG process", icon=":material/account_tree:")
st.caption(
    "A RAG system prepares its reference material before questions are asked, "
    "then retrieves relevant context for each question."
)

steps = [
    (
        "1. Prepare documents",
        "Extract text from source files. This chatbot can use OCR when PDF "
        "pages contain little extractable text.",
        ":material/description:",
    ),
    (
        "2. Split into chunks",
        "Divide long text into smaller, overlapping passages so relevant "
        "sections can be found independently.",
        ":material/text_snippet:",
    ),
    (
        "3. Create an index",
        "Convert each passage into an embedding, a numeric representation "
        "that helps compare its meaning with a question.",
        ":material/database:",
    ),
    (
        "4. Retrieve passages",
        "When a question arrives, search the index for passages with similar "
        "meaning and select the most relevant ones.",
        ":material/search:",
    ),
    (
        "5. Generate a response",
        "Give the question and retrieved passages to a language model so it "
        "can write a response grounded in that context.",
        ":material/auto_awesome:",
    ),
]

for start in range(0, len(steps), 3):
    columns = st.columns(3, gap="medium")
    for column, (title, description, icon) in zip(columns, steps[start : start + 3]):
        with column:
            with st.container(border=True):
                st.subheader(title, icon=icon)
                st.write(description)

with st.container(border=True):
    st.subheader("How this PDF chatbot uses RAG")
    st.markdown(
        "Upload up to three PDFs at a time and process them to build a "
        "searchable index. This limit helps keep the retrieved context within "
        "the language model's context capacity. "
        "For each question, the app retrieves up to four relevant text chunks "
        "from those PDFs and sends them with the question to the configured "
        "language model. The model is instructed to answer from that context "
        "or say when the context does not contain the answer."
    )

st.page_link(
    "main.py",
    label="Try it with your PDFs",
    icon=":material/arrow_forward:",
    help="Open the PDF workspace, process documents, and ask a question.",
)
