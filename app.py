import streamlit as st
import requests
import re
import concurrent.futures
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="LegalEdge India", page_icon="⚖️", layout="wide")

# --- SECRETS & SETUP ---
try:
    PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Missing API Keys in secrets.toml.")
    st.stop()

# Initialize Gemini for blazing fast RAG and routing
# Using the standard fast tier model for LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.1 # Strictly deterministic for legal routing
)

# PageIndex Mapping
LAW_DOC_MAPPING = {
    "BNS": "bfbbdf13-b5bb-4de0-b6ec-7df038de8fec",
    "BNSS": "23d11b36-f08a-4933-90d5-71649d212727",
    "BSA": "46906660-f96b-4bd4-99ce-42790dccfc16"
}

# --- CACHING & DATA FETCHING ---
@st.cache_data(show_spinner=False)
def fetch_law_trees():
    """Fetches the full hierarchical trees and creates a stripped 'Map' version."""
    headers = {"x-api-key": PAGEINDEX_API_KEY}
    trees = {}
    maps = {}
    
    def remove_text_fields(node):
        node_copy = {k: v for k, v in node.items() if k not in ['text', 'content']}
        if 'children' in node_copy and isinstance(node_copy['children'], list):
            node_copy['children'] = [remove_text_fields(child) for child in node_copy['children']]
        return node_copy

    for law_name, doc_id in LAW_DOC_MAPPING.items():
        try:
            response = requests.get(f"https://api.pageindex.ai/v1/documents/{doc_id}/tree", headers=headers)
            if response.status_code == 200:
                full_tree = response.json()
                trees[law_name] = full_tree
                maps[law_name] = remove_text_fields(full_tree)
            else:
                st.error(f"Failed to fetch {law_name}")
        except Exception as e:
            st.error(f"Error fetching {law_name}: {e}")
            
    return trees, maps

trees, maps = fetch_law_trees()

# --- RAG LOGIC ---
def expand_query(query: str) -> str:
    """Translates old IPC/CrPC references to new BNS/BNSS terminology before searching."""
    prompt = f"""
    You are a legal translator. The user is asking about Indian Law.
    If the user mentions old penal codes (IPC, CrPC, Indian Evidence Act), translate their query 
    to ask about the new equivalent codes (BNS, BNSS, BSA). 
    If they don't mention old codes, just return their query exactly as is.
    
    User Query: {query}
    Translated Query:
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def process_law_tree(query: str, law_name: str, full_tree: dict, tree_map: dict) -> list:
    """Hybrid routing: Regex for explicit sections, LLM for conceptual search."""
    retrieved_texts = []
    
    # 1. Regex Exact Match
    section_match = re.search(r'section\s+(\d+[a-zA-Z]?)', query.lower())
    if section_match:
        section_num = section_match.group(1)
        
        def find_section(node):
            if "Section " + section_num in str(node.get("title", "")):
                return node
            for child in node.get("children", []):
                result = find_section(child)
                if result: return result
            return None
            
        found_node = find_section(full_tree)
        if found_node and 'text' in found_node:
            retrieved_texts.append(f"**{found_node.get('title', 'Section')}**: {found_node['text']}")
            return retrieved_texts

    # 2. LLM Conceptual Routing
    routing_prompt = f"""
    You are a legal routing engine. 
    Review the following Law Map (which only contains titles and node IDs, no heavy text).
    Find the 1-3 most relevant Node IDs that answer the user's query.
    Return ONLY a comma-separated list of node IDs. Do not write anything else.
    
    User Query: {query}
    Law Map: {tree_map}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=routing_prompt)])
        target_ids = [n_id.strip() for n_id in response.content.split(',')]
        
        # 3. Text Assembly
        def extract_by_ids(node, target_ids):
            results = []
            if str(node.get("id")) in target_ids and 'text' in node:
                results.append(f"**{node.get('title', 'Section')}**: {node['text']}")
            for child in node.get("children", []):
                results.extend(extract_by_ids(child, target_ids))
            return results
            
        retrieved_texts = extract_by_ids(full_tree, target_ids)
        
    except Exception as e:
        pass # Fallback to empty if LLM fails formatting
        
    return list(set(retrieved_texts)) # Deduplicate

def build_html_tree(retrieved_data: dict) -> str:
    """Builds the interactive IDE-style directory tree for the UI."""
    if not retrieved_data:
        return ""
        
    html = """
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; margin: 15px 0; font-family: 'Inter', sans-serif;">
        <h4 style="margin-top: 0; color: #1E293B; font-size: 14px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">📚 Retrieved Legal Statutes</h4>
    """
    
    for law, texts in retrieved_data.items():
        if texts:
            html += f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; font-weight: 600; color: #2563EB; padding: 5px; border-radius: 4px; transition: background 0.2s;">
                    📘 {law} ({len(texts)} excerpts)
                </summary>
                <div style="margin-left: 20px; border-left: 2px solid #E2E8F0; padding-left: 15px; margin-top: 5px;">
            """
            for text in texts:
                # Split title from text if formatted with markdown bold
                parts = text.split("**: ", 1)
                title = parts[0].replace("**", "") if len(parts) > 1 else "Statute"
                content = parts[1] if len(parts) > 1 else text
                
                html += f"""
                <details style="margin-top: 8px;">
                    <summary style="cursor: pointer; color: #475569; font-size: 13px; font-weight: 500;">
                        📄 {title}
                    </summary>
                    <p style="font-size: 13px; color: #334155; background: #FFFFFF; padding: 10px; border-radius: 4px; border: 1px solid #E2E8F0; margin-top: 5px; line-height: 1.5;">
                        {content}
                    </p>
                </details>
                """
            html += "</div></details>"
            
    html += "</div>"
    return html

# --- UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    
    /* Sleek Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #E2E8F0;
        background-color: white;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #2563EB;
        color: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1);
    }
    
    /* Chat Bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
    }
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(odd) {
        background-color: #F1F5F9; /* User bubble */
    }
    [data-testid="stChatMessage"][data-baseweb="card"]:nth-child(even) {
        background-color: #FFFFFF; /* AI bubble */
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Profile Text */
    .profile-text { font-size: 14px; color: #64748B; margin-top: 20px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/1200px-Emblem_of_India.svg.png", width=60)
    st.markdown("<h2 style='font-weight: 700; color: #1E293B;'>LegalEdge India</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 14px;'>Intelligent navigation for the BNS, BNSS, and BSA.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🗑️ Start New Case", use_container_width=True):
        st.session_state.messages = []
        st.session_state.starter_prompt = None
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-weight: 600; font-size: 16px; color: #1F2937;'>⚖️ Search Scope</h3>", unsafe_allow_html=True)
    search_bns = st.checkbox("BNS (Substantive Law)", value=True)
    search_bnss = st.checkbox("BNSS (Procedural Law)", value=True)
    search_bsa = st.checkbox("BSA (Evidence Law)", value=True)
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("""
        <p class='profile-text'>
        <strong>Saket</strong><br>
        Just a product guy who loves tech. Building things here to learn, tinker, and have a little fun.
        </p>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "starter_prompt" not in st.session_state:
    st.session_state.starter_prompt = None

# --- MAIN CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("html"):
            st.markdown(msg["html"], unsafe_allow_html=True)
        st.markdown(msg["content"])

# --- EMPTY STATE SUGGESTIONS ---
if len(st.session_state.messages) == 0:
    st.markdown("<h2 style='text-align: center; color: #1E293B; margin-top: 10vh;'>How can I help with your case today?</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📝 Draft a Notice", use_container_width=True): st.session_state.starter_prompt = "Draft a legal notice for breach of contract."
    with col2:
        if st.button("🔄 IPC to BNS", use_container_width=True): st.session_state.starter_prompt = "What is the BNS equivalent of IPC Section 420?"
    with col3:
        if st.button("💼 Procedural Query", use_container_width=True): st.session_state.starter_prompt = "What is the procedure for issuing a summons under the BNSS?"
    with col4:
        if st.button("🚨 Find Penalties", use_container_width=True): st.session_state.starter_prompt = "What is the specific penalty for rash and negligent driving under the BNS?"

# --- CHAT INPUT HANDLING ---
user_input = st.chat_input("E.g., Search for a specific legal section, penalty, or procedure...")

if st.session_state.starter_prompt:
    user_input = st.session_state.starter_prompt
    st.session_state.starter_prompt = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("🧠 Analyzing Legal Framework...", expanded=True) as status:
            st.write("🔍 Translating legacy codes (IPC/CrPC) to Bharatiya Nyaya Sanhita...")
            translated_query = expand_query(user_input)
            
            st.write("📚 Scanning Bare Acts...")
            retrieved_data = {}
            
            # Parallel Execution for Speed
            active_laws = []
            if search_bns: active_laws.append("BNS")
            if search_bnss: active_laws.append("BNSS")
            if search_bsa: active_laws.append("BSA")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_law = {
                    executor.submit(process_law_tree, translated_query, law, trees[law], maps[law]): law 
                    for law in active_laws
                }
                for future in concurrent.futures.as_completed(future_to_law):
                    law_name = future_to_law[future]
                    try:
                        retrieved_data[law_name] = future.result()
                    except Exception as exc:
                        st.error(f"{law_name} search generated an exception: {exc}")
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # Build Interactive Directory
        html_tree = build_html_tree(retrieved_data)
        if html_tree:
            st.markdown(html_tree, unsafe_allow_html=True)
            
        # Final LLM Synthesis
        synthesis_prompt = f"""
        You are an expert Indian Legal Assistant. Answer the user's query using ONLY the provided legal excerpts.
        If the user asks you to draft something, you can be flexible. Otherwise, structure your response professionally.
        Do not hallucinate external laws.
        
        Context Data: {retrieved_data}
        User Query: {translated_query}
        """
        
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        try:
            for chunk in llm.stream([SystemMessage(content=synthesis_prompt)]):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = "An error occurred while communicating with the AI. Please try again."
            message_placeholder.markdown(full_response)
            
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "html": html_tree})

# --- CASE EXPORT (Main UI) ---
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Center the button nicely using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        chat_export = "# ⚖️ LegalEdge India - Case Notes\n\n"
        for msg in st.session_state.messages:
            # We don't want to export the raw HTML tree, just the text
            if "html" not in msg: 
                role = "🧑‍💼 User Query" if msg["role"] == "user" else "🤖 LegalEdge Analysis"
                chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"
            
        st.download_button(
            label="📄 Download Case Notes",
            data=chat_export,
            file_name="LegalEdge_Case_Notes.md",
            mime="text/markdown",
            use_container_width=True
        )
