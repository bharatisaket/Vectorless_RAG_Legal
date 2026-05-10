import streamlit as st
import os
import json
import copy
import concurrent.futures
import re
from pageindex import PageIndexClient
import pageindex.utils as utils
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 1. Configuration & Security ---
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

LAW_DOC_MAPPING = {
    "BNS": {"name": "Bharatiya Nyaya Sanhita (BNS)", "id": "pi-cmoo3zavg011p01qr7mbgdtev"},
    "BNSS": {"name": "Bharatiya Nagarik Suraksha Sanhita (BNSS)", "id": "pi-cmoo55m2w012301qrevyj12j4"},
    "BSA": {"name": "Bharatiya Sakshya Adhiniyam (BSA)", "id": "pi-cmoo3z9av011n01qrcozuk72f"}
}

SYSTEM_PROMPT = """You are LegalEdge India, an elite Indian Legal AI Assistant. 
Your primary directive is to provide accurate legal analysis based STRICTLY on the retrieved context. 

DEFAULT STRUCTURE:
1. Executive Summary
2. Statutory Breakdown (with verbatim quotes)
3. Summary Table
4. Procedural Notes

FLEXIBILITY RULE: If the user explicitly asks for a different format (e.g., "Draft a legal notice", "Explain this to a beginner", "Compare these two sections"), abandon the default structure and format your response to best serve their specific request.

If the question is not related to Indian criminal law (BNS, BNSS, BSA), politely state that you cannot assist. Do not hallucinate."""

# --- 2. Caching Engine ---
@st.cache_data(show_spinner="Loading Legal Codes into memory...")
def fetch_law_trees():
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cached_data = {}
    for key, data in LAW_DOC_MAPPING.items():
        tree_response = pi_client.get_tree(data["id"], node_summary=True)
        tree = tree_response.get("result", tree_response) if isinstance(tree_response, dict) else tree_response
        node_mapping = utils.create_node_mapping(tree)
        clean_tree = copy.deepcopy(tree)
        tree_without_text = utils.remove_fields(clean_tree, fields=['text'])
        cached_data[data["id"]] = {
            "mapping": node_mapping,
            "tree_json": json.dumps(tree_without_text)
        }
    return cached_data

legal_trees = fetch_law_trees()

# --- 3. HELPER FUNCTIONS ---
def expand_query(user_input):
    translation_prompt = f"""
    You are an expert in Indian Criminal Law transitions. India replaced the IPC, CrPC, and IEA with the BNS, BNSS, and BSA.
    TASK: If the user mentions an old section (e.g., IPC 302, IPC 420), translate it to reference the NEW 2023 Bharatiya codes. 
    If no translation is needed, return the original query. Output ONLY the new query string.
    User Query: {user_input}
    Translated Query:"""
    try:
        response = llm.invoke(translation_prompt)
        return response.content.strip()
    except:
        return user_input

def process_law_tree(doc_id, doc_name, user_query, history_str):
    tree_data = legal_trees[doc_id]
    tree_json = tree_data["tree_json"]
    node_mapping = tree_data["mapping"]
    
    extracted_nodes = []
    regex_matched = False
    
    section_targets = re.findall(r'\b\d+[a-zA-Z]?\b', user_query) 
    for node_id_key, node_data in node_mapping.items():
        if 'text' in node_data:
            node_text = node_data['text']
            for num in section_targets:
                if f" {num}." in node_text[:100] or f"Section {num}" in node_text[:100] or node_text.startswith(f"{num}."):
                    extracted_nodes.append({"doc_name": doc_name, "text": node_text})
                    regex_matched = True

    routing_prompt = f"""
    Analyze the tree and query. Return a valid JSON array of the most relevant node IDs.
    CRITICAL RULE: NEVER select parent Chapters. Select ONLY specific Section or Schedule nodes.
    Latest Query: {user_query}
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
    
    for node_id_key in selected_nodes:
        if node_id_key in node_mapping and 'text' in node_mapping[node_id_key]:
            node_text = node_mapping[node_id_key]['text']
            extracted_nodes.append({"doc_name": doc_name, "text": node_text})
            
    seen = set()
    unique_nodes = []
    for n in extracted_nodes:
        if n["text"] not in seen:
            seen.add(n["text"])
            unique_nodes.append(n)
            
    return unique_nodes, regex_matched

# --- UPDATE: The Flattened Accordion HTML Generator ---
def build_html_tree(retrieved_nodes):
    if not retrieved_nodes:
        return "<i>No specific statutes retrieved.</i>"
        
    grouped = {}
    for node in retrieved_nodes:
        doc = node["doc_name"]
        if doc not in grouped:
            grouped[doc] = []
        grouped[doc].append(node["text"])
        
    html = "<div class='legal-accordion'>"
    for doc, texts in grouped.items():
        # Act Level (Root Card) - Flattened to avoid Markdown code-block rendering
        html += f"<details class='act-group' open><summary class='act-header'><span class='act-title'>{doc}</span><span class='chevron'></span></summary><div class='act-body'>"
        
        for text in texts:
            first_line = text.split('\n')[0].strip()
            title = first_line if len(first_line) < 65 else first_line[:65] + "..."
            clean_text = text.replace('\n', '<br>')
            
            # Section Level (Nested Card) - Flattened
            html += f"<details class='section-item'><summary class='section-header'><span class='section-title'>{title}</span><span class='chevron'></span></summary><div class='section-text'>{clean_text}</div></details>"
            
        html += "</div></details>"
    html += "</div>"
    return html

# --- 4. UI Initialization & Styling ---
st.set_page_config(page_title="LegalEdge India", page_icon="⚖️", layout="wide")

# --- CSS UI UPGRADE (Modern Nested Accordion) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* SaaS Buttons */
    div.stButton > button {
        border-radius: 24px; border: 1px solid #E5E7EB; background-color: #FFFFFF; color: #374151;
        font-weight: 500; transition: all 0.2s ease-in-out; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div.stButton > button:hover {
        border-color: #2563EB; color: #2563EB; background-color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transform: translateY(-1px);
    }
    
    /* Chat Bubbles */
    [data-testid="stChatMessage"] { border-radius: 12px; padding: 15px; margin-bottom: 20px; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #FFFFFF; border: 1px solid #F3F4F6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #F8FAFC; border-left: 4px solid #2563EB;
    }

    /* --- THE MODERN ACCORDION CSS --- */
    .legal-accordion details { margin: 0; padding: 0; }
    .legal-accordion summary { list-style: none; outline: none; cursor: pointer; user-select: none; }
    .legal-accordion summary::-webkit-details-marker { display: none; }

    /* Act Group (Outer Card) */
    .act-group { 
        margin-bottom: 16px; border: 1px solid #E2E8F0; border-radius: 12px; 
        background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden; 
    }
    .act-header { 
        display: flex; justify-content: space-between; align-items: center; 
        padding: 16px 20px; background: #F8FAFC; transition: background 0.2s ease; 
    }
    .act-header:hover { background: #F1F5F9; }
    details[open] > .act-header { border-bottom: 1px solid #E2E8F0; }
    .act-title { font-weight: 700; color: #0F172A; font-size: 1.05em; display: flex; align-items: center; gap: 8px; }
    .act-title::before { content: '🏛️'; }

    /* Section Item (Inner Toggles) */
    .act-body { padding: 12px; background: #FAFAFA; }
    .section-item { 
        margin-bottom: 6px; border: 1px solid transparent; border-radius: 8px; 
        transition: all 0.2s ease; 
    }
    .section-item:hover { 
        border-color: #CBD5E1; background: #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.02); 
    }
    .section-header { 
        display: flex; justify-content: space-between; align-items: center; 
        padding: 12px 16px; border-radius: 8px; transition: background 0.2s; 
    }
    .section-title { font-weight: 600; color: #334155; font-size: 0.95em; display: flex; align-items: center; gap: 8px; }
    .section-title::before { content: '📄'; opacity: 0.8; font-size: 1.1em;}

    /* The Legal Text Inset */
    .section-text { 
        margin: 4px 16px 16px 16px; padding: 16px 20px; background: #FFFFFF; 
        border-left: 3px solid #3B82F6; border-radius: 0 8px 8px 0; 
        color: #475569; font-size: 0.9em; line-height: 1.7; 
        box-shadow: inset 0 0 0 1px #F1F5F9; max-height: 400px; overflow-y: auto; 
    }

    /* Animated Chevrons */
    .chevron { 
        width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; 
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
    }
    .chevron::after { 
        content: ''; width: 7px; height: 7px; border-right: 2px solid #94A3B8; border-bottom: 2px solid #94A3B8; 
        transform: rotate(45deg); margin-bottom: 4px; transition: border-color 0.2s; 
    }
    .act-header:hover .chevron::after, .section-header:hover .chevron::after { border-color: #3B82F6; }
    details[open] > summary .chevron { transform: rotate(225deg); margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR LOGIC ---
with st.sidebar:
    if st.button("🗑️ Start New Case", use_container_width=True):
        st.session_state.messages = []
        st.session_state.starter_prompt = None
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight: 700; color: #1F2937;'>⚖️ Search Scope</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6B7280; font-size: 0.9em;'>Toggle active databases for your search:</p>", unsafe_allow_html=True)
    
    bns_active = st.checkbox("**BNS** (Penal Code)", value=True, help="Bharatiya Nyaya Sanhita")
    bnss_active = st.checkbox("**BNSS** (Criminal Procedures)", value=True, help="Bharatiya Nagarik Suraksha Sanhita")
    bsa_active = st.checkbox("**BSA** (Evidence Act)", value=True, help="Bharatiya Sakshya Adhiniyam")
    
    selected_doc_ids = []
    if bns_active: selected_doc_ids.append(LAW_DOC_MAPPING["BNS"]["id"])
    if bnss_active: selected_doc_ids.append(LAW_DOC_MAPPING["BNSS"]["id"])
    if bsa_active: selected_doc_ids.append(LAW_DOC_MAPPING["BSA"]["id"])

    st.divider()
    st.warning("**Disclaimer:** This tool is for informational purposes only. AI can make mistakes, so please verify important information.")
    
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; font-size: 0.85em; color: #9CA3AF;'>
        <p>Built by <strong>Saket</strong></p>
        <a href='YOUR_GITHUB_URL' target='_blank' style='text-decoration: none; color: #6B7280; margin-right: 15px;'>🐙 GitHub</a>
        <a href='YOUR_LINKEDIN_URL' target='_blank' style='text-decoration: none; color: #6B7280;'>💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# Main Header
st.markdown("<h1 style='font-weight: 800; color: #111827; letter-spacing: -0.02em;'>LegalEdge India</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='font-size: 1.1em; color: #4B5563; margin-bottom: 0;'>
<strong>Your intelligent navigator for India's modern criminal laws.</strong> Seamlessly search across the BNS, BNSS, and BSA. 
Ask complex legal questions, automatically translate legacy IPC/CrPC sections, and retrieve exact statutory citations in seconds.
</p>
""", unsafe_allow_html=True)
st.caption("⚙️ *Powered by Vectorless RAG & Gemini Flash Lite*")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- EMPTY STATE GREETING & CHIPS ---
if len(st.session_state.messages) == 0:
    st.markdown("<h3 style='text-align: center; color: #374151; padding-bottom: 10px; font-weight: 600;'>Where should we start?</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("📱 Digital Evidence", use_container_width=True): st.session_state.starter_prompt = "Under the new BSA, what are the strict conditions for submitting WhatsApp chats or electronic records as evidence?"
    with col2:
        if st.button("⚖️ Compare IPC & BNS", use_container_width=True): st.session_state.starter_prompt = "What is the difference between Murder under the old IPC and the new BNS?"
    with col3:
        if st.button("📖 Explain it Simply", use_container_width=True): st.session_state.starter_prompt = "Explain the rules for electronic evidence (BSA) as if I am a beginner."
    with col4:
        if st.button("🚨 Find Penalties", use_container_width=True): st.session_state.starter_prompt = "What is the specific penalty for mob lynching under the BNS?"
    st.markdown("<br>", unsafe_allow_html=True)

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "badge" in message and message["badge"]:
            st.success(message["badge"])
        if "context_html" in message and message["context_html"]:
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(message["context_html"], unsafe_allow_html=True)

user_input = st.chat_input("E.g., What is the penalty for IPC 420?")
prompt = None

if "starter_prompt" in st.session_state and st.session_state.starter_prompt is not None:
    prompt = st.session_state.starter_prompt
    st.session_state.starter_prompt = None 
elif user_input:
    prompt = user_input

# --- 5. Main Logic Loop ---
if prompt:
    if not selected_doc_ids:
        st.warning("⚠️ Please select at least one legal code from the sidebar.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        badge_text = None
        try:
            if any(word in prompt.lower() for word in ["recipe", "weather", "movie", "song", "cake"]):
                 st.markdown("I am a specialized Legal AI for the BNS, BSA, and BNSS. I cannot assist with non-legal queries.")
                 st.session_state.messages.append({"role": "assistant", "content": "I am a specialized Legal AI for the BNS, BSA, and BNSS. I cannot assist with non-legal queries."})
                 st.stop()

            with st.status("Analyzing Legal Context...", expanded=True) as status:
                st.write("🔍 Translating legacy codes to modern framework...")
                smart_query = expand_query(prompt)
                
                st.write("📚 Searching active legal databases...")
                retrieved_nodes = [] 
                direct_match_found = False
                
                chat_history_str = ""
                for msg in st.session_state.messages[-5:-1]:
                    chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_doc = {}
                    for doc_id in selected_doc_ids:
                        doc_name = next((v["name"] for k, v in LAW_DOC_MAPPING.items() if v["id"] == doc_id), "Legal Database")
                        future_to_doc[executor.submit(process_law_tree, doc_id, doc_name, smart_query, chat_history_str)] = doc_id
                    
                    for future in concurrent.futures.as_completed(future_to_doc):
                        try:
                            nodes, regex_flag = future.result()
                            retrieved_nodes.extend(nodes)
                            if regex_flag:
                                direct_match_found = True
                        except Exception as exc:
                            st.error(f"Routing error: {exc}")
                
                st.write("🧠 Compiling legal response with Gemini...")
                
                raw_texts = [n["text"] for n in retrieved_nodes]
                context_text = "\n\n".join(raw_texts) if raw_texts else "No specific statutes retrieved."
                
                html_tree_ui = build_html_tree(retrieved_nodes)

                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for msg in st.session_state.messages[:-1]:
                    role_class = HumanMessage if msg["role"] == "user" else AIMessage
                    messages.append(role_class(content=msg["content"]))
                
                messages.append(HumanMessage(content=f"CONTEXT:\n{context_text}\n\nQUESTION:\n{smart_query}"))
                final_response = llm.invoke(messages)
                
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
            if smart_query.lower() != prompt.lower():
                st.caption(f"🔄 *Translated legacy query to modern framework:* {smart_query}")
                
            if direct_match_found:
                badge_text = "🎯 Direct Section Match Found"
                st.success(badge_text)
            
            raw_answer = final_response.content
            if isinstance(raw_answer, list) and len(raw_answer) > 0 and isinstance(raw_answer[0], dict) and "text" in raw_answer[0]:
                answer = raw_answer[0]["text"]
            else:
                answer = str(raw_answer)
            
            st.markdown(answer)
            
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(html_tree_ui, unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "context_html": html_tree_ui,
                "badge": badge_text
            })
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                st.warning("⏳ The AI is currently handling too many requests. Please wait a moment and try again.")
            else:
                st.error(f"Error: {e}")
