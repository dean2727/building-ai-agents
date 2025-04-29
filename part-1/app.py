import streamlit as st
from graph import process_question

research_texts = [
    "Research Report: Results of a New AI Model Improving Image Recognition Accuracy to 98%",
    "Academic Paper Summary: Why Transformers Became the Mainstream Architecture in Natural Language Processing",
    "Latest Trends in Machine Learning Methods Using Quantum Computing"
]

development_texts = [
    "Project A: UI Design Completed, API Integration in Progress",
    "Project B: Testing New Feature X, Bug Fixes Needed",
    "Product Y: In the Performance Optimization Stage Before Release"
]

def main():
    st.set_page_config(
        page_title="AI Research & Development Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton > button {
        width: 100%;
        margin-top: 20px;
    }
    .data-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: black;
    }
    .research-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .dev-box {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with Data Display
    with st.sidebar:
        st.header("📚 Available Data")
        
        st.subheader("Research Database")
        for text in research_texts:
            st.markdown(f'<div class="data-box research-box">{text}</div>', unsafe_allow_html=True)
            
        st.subheader("Development Database")
        for text in development_texts:
            st.markdown(f'<div class="data-box dev-box">{text}</div>', unsafe_allow_html=True)

    # Main Content
    st.title("🤖 AI Research & Development Assistant")
    st.markdown("---")

    # Query Input
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the latest advancement in AI research?")

    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("🔍 Get Answer", use_container_width=True):
            if query:
                with st.spinner('Processing your question...'):
                    # Process query through workflow
                    events = process_question(query, {"configurable":{"thread_id":"1"}})
                    
                    # Display results
                    for event in events:
                        if 'agent' in event:
                            with st.expander("🔄 Processing Step", expanded=True):
                                content = event['agent']['messages'][0].content
                                if "Results:" in content:
                                    # Display retrieved documents
                                    st.markdown("### 📑 Retrieved Documents:")
                                    docs_start = content.find("Results:")
                                    docs = content[docs_start:]
                                    st.info(docs)
                        elif 'generate' in event:
                            st.markdown("### ✨ Final Answer:")
                            st.success(event['generate']['messages'][0].content)
            else:
                st.warning("⚠️ Please enter a question first!")

    with col2:
        st.markdown("""
        <div style="color: black;">
            <h3>🎯 How to Use</h3>
            <ol>
                <li>Type your question in the text box</li>
                <li>Click "Get Answer" to process</li>
                <li>View retrieved documents and final answer</li>
            </ol>
            
            Example Questions:
            - What are the latest advancements in AI research?
            - What is the status of Project A?
            - What are the current trends in machine learning?
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()