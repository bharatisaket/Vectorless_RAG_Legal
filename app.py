import streamlit as st
import os
import json
import copy
import concurrent.futures
from pageindex import PageIndexClient
import pageindex.utils as utils
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 1. Configuration & Security ---
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

LAW_DOC_MAPPING = {
    "Bharatiya Nyaya Sanhita (BNS) - Penal Code": "pi-cmoo3zavg011p01qr7mbgdtev",
    "Bharatiya Nagarik Suraksha Sanhita (BNSS) - Procedures": "pi-cmoo55m2w012301qrevyj12j4",
    "Bharatiya Sakshya Adhiniyam (BSA) - Evidence": "pi-cmoo3z9av011n01qrcozuk72f"
}

SYSTEM_PROMPT = """You are an elite Indian Legal AI Assistant. 
Your primary directive is to provide structured legal analysis based STRICTLY on the retrieved context. 

Structure:
1. Executive Summary
2. Statutory Breakdown (with verbatim quotes)
3. Summary Table
4. Procedural & Legal Notes

If the question is not related to Indian criminal law (BNS, BNSS, BSA), or if the answer is not in the text, politely state that you cannot assist with that specific query. Do not hallucinate."""

# --- 2. Caching Engine ---
@st.cache_data(show_spinner="Loading Legal Codes into memory...")
def fetch_law_trees():
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cached_data = {}
    for doc_id in LAW_DOC_MAPPING.values():
        tree_response = pi_client.get_tree(doc_id, node_summary=True)
        tree = tree_response.get("result", tree_response) if isinstance(tree_response, dict) else tree_response
        node_mapping = utils.create_node_mapping(tree)
        clean_tree = copy.deepcopy(tree)
        tree_without_text = utils.remove_fields(clean_tree, fields=['text'])
        cached_data[doc_id] = {
            "mapping": node_mapping,
            "tree_json": json.dumps(tree_without_text)
        }
    return cached_data

legal_trees = fetch_law_trees()

# --- 3. PARALLEL ROUTING (NO LIMITS) ---
def process_law_tree(doc_id, user_query, history_str):
    """Executes tree search. Grabs ALL relevant nodes without starving the AI."""
    tree_data = legal_trees[doc_id]
    tree_json = tree_data["tree_json"]
    node_mapping = tree_data["mapping"]
    
    # UPDATE: Removed the artificial node limits entirely. Just block whole Chapters.
    routing_prompt = f"""
    Analyze the tree and query. Return a valid JSON array of the most relevant node IDs.
    
    CRITICAL RULE:
    1. NEVER select parent Chapters or Parts. Select ONLY specific Section or Schedule nodes.
    2. Select ALL nodes that directly pertain to the query to ensure a complete legal answer.
    
    Latest Query: {user_query}
    Conversation History: {history_str}
    Tree: {tree_json}
    """
    
    route_response = llm.invoke(routing_prompt)
    raw_content = route_response.content
    
    if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict) and "text" in raw_content[0]:
        raw_content = raw_content[0]["text"]
    
    if isinstance(raw_content, list):
        selected_nodes = raw_content
    else:
        cleaned = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            selected_nodes = json.loads(cleaned) if cleaned else []
        except:
            selected_nodes = []
    
    extracted_texts = []
    
    # UPDATE: Loop through all selected nodes and grab their full text. No budget.
    for node_id in selected_nodes:
        if node_id in node_mapping and 'text' in node_mapping[node_id]:
            extracted_texts.append(node_mapping[node_id]['text'])
            
    return extracted_texts

# --- 4. UI Initialization & Sidebar ---
st.set_page_config(page_title="Bharatiya Laws AI", page_icon="⚖️", layout="wide")

with st.sidebar:
    st.header("⚖️ Search Scope")
    selected_laws = st.multiselect(
        "Active Databases:",
        options=list(LAW_DOC_MAPPING.keys()),
        default=list(LAW_DOC_MAPPING.keys())
    )
    st.divider()
    st.warning("**Disclaimer:** This tool is for informational purposes only. It is not a substitute for professional legal advice. AI can hallucinate.")

st.title("Bharatiya Laws AI Assistant")
st.caption("Powered by Vectorless RAG & Gemini Flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(message["context"])

# --- 5. Main Logic Loop ---
if prompt := st.chat_input("E.g., What is the penalty for mob lynching?"):
    
    if not selected_laws:
        st.warning("⚠️ Please select at least one legal code from the sidebar.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            if any(word in prompt.lower() for word in ["recipe", "weather", "movie", "song", "cake"]):
                 message_placeholder.markdown("I am a specialized Legal AI for the BNS, BSA, and BNSS. I cannot assist with non-legal queries.")
                 st.session_state.messages.append({"role": "assistant", "content": "I am a specialized Legal AI for the BNS, BSA, and BNSS. I cannot assist with non-legal queries."})
                 st.stop()

            message_placeholder.markdown("Executing Parallel Tree Search...")
            retrieved_texts = []
            
            chat_history_str = ""
            history_subset = st.session_state.messages[-5:-1] 
            for msg in history_subset:
                chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
            active_doc_ids = [LAW_DOC_MAPPING[law] for law in selected_laws]
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_doc = {executor.submit(process_law_tree, doc_id, prompt, chat_history_str): doc_id for doc_id in active_doc_ids}
                
                for future in concurrent.futures.as_completed(future_to_doc):
                    try:
                        result_texts = future.result()
                        retrieved_texts.extend(result_texts)
                    except Exception as exc:
                        st.error(f"Routing error on one of the documents: {exc}")
            
            message_placeholder.markdown("Reasoning with Gemini...")
            
            # FULL context goes to the AI
            context_text = "\n\n".join(retrieved_texts)
            if not retrieved_texts:
                context_text = "No specific statutes retrieved."
            
            # TRUNCATED context goes to the UI Expander
            ui_display_text = context_text
            if len(ui_display_text) > 5000:
                ui_display_text = ui_display_text[:5000] + "\n\n***\n*... [Remaining text hidden for UI readability. The AI analyzed the complete legal text.]*"
            
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for msg in st.session_state.messages[:-1]:
                role_class = HumanMessage if msg["role"] == "user" else AIMessage
                messages.append(role_class(content=msg["content"]))
                    
            messages.append(HumanMessage(content=f"CONTEXT:\n{context_text}\n\nQUESTION:\n{prompt}"))
            
            final_response = llm.invoke(messages)
            
            raw_answer = final_response.content
            if isinstance(raw_answer, list) and len(raw_answer) > 0 and isinstance(raw_answer[0], dict) and "text" in raw_answer[0]:
                answer = raw_answer[0]["text"]
            else:
                answer = str(raw_answer)
            
            message_placeholder.markdown(answer)
            
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(ui_display_text)
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "context": ui_display_text})
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                st.warning("⏳ The AI is currently handling too many requests. Please wait a moment and try again.")
            else:
                st.error(f"Error: {e}")
