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

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)

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

# --- 3. PARALLEL ROUTING & DYNAMIC BUDGET HELPER ---
def process_law_tree(doc_id, user_query, history_str):
    """Executes tree search for a single document with a smart character budget."""
    tree_data = legal_trees[doc_id]
    tree_json = tree_data["tree_json"]
    node_mapping = tree_data["mapping"]
    
    routing_prompt = f"""
    Analyze the tree and query. Return a valid JSON array of the most relevant node IDs.
    
    CRITICAL RULES:
    1. NEVER select parent Chapters or Parts. Select ONLY specific Section or Schedule nodes.
    2. ORDER MATTERS: You MUST order the array from most critical to least critical. 
    3. Select up to 6 nodes to ensure you capture the core offence and procedures.
    
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
    TOTAL_CHAR_BUDGET = 20000  # Smart safety net (approx 4,000 tokens)
    current_char_count = 0
    
    for node_id in selected_nodes:
        if node_id in node_mapping and 'text' in node_mapping[node_id]:
            node_text = node_mapping[node_id]['text']
            node_len = len(node_text)
            
            # If adding this node breaks the budget, stop reading lower-priority nodes
            if current_char_count + node_len > TOTAL_CHAR_BUDGET:
                # Failsafe: If even the first critical node is too massive, truncate it
                if len(extracted_texts) == 0:
                    extracted_texts.append(node_text[:TOTAL_CHAR_BUDGET] + "\n\n... [Text truncated to preserve system stability]")
                break 
                
            extracted_texts.append(node_text)
            current_char_count += node_len
            
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
            # Guardrail for non-legal queries
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
            
            # --- THE PARALLEL THREAD POOL ENGINE ---
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_doc = {executor.submit(process_law_tree, doc_id, prompt, chat_history_str): doc_id for doc_id in active_doc_ids}
                
                for future in concurrent.futures.as_completed(future_to_doc):
                    try:
                        result_texts = future.result()
                        retrieved_texts.extend(result_texts)
                    except Exception as exc:
                        st.error(f"Routing error on one of the documents: {exc}")
            
            message_placeholder.markdown("Reasoning with Gemini...")
            
            context_text = "\n\n".join(retrieved_texts)
            if not retrieved_texts:
                context_text = "No specific statutes retrieved."
            
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
                st.markdown(context_text)
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "context": context_text})
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                st.warning("⏳ The AI is currently handling too many requests. Please wait a moment and try again.")
            else:
                st.error(f"Error: {e}")
