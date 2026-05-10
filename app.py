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
    "BNS": {"name": "Bharatiya Nyaya Sanhita (BNS) - Penal Code", "id": "pi-cmoo3zavg011p01qr7mbgdtev"},
    "BNSS": {"name": "Bharatiya Nagarik Suraksha Sanhita (BNSS) - Procedures", "id": "pi-cmoo55m2w012301qrevyj12j4"},
    "BSA": {"name": "Bharatiya Sakshya Adhiniyam (BSA) - Evidence", "id": "pi-cmoo3z9av011n01qrcozuk72f"}
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

# --- UPDATE: FLATTENED DIRECTORY TREE HTML ---
# This generates the IDE-style tree but without formatting spaces so Streamlit doesn't break.
def build_html_tree(retrieved_nodes):
    if not retrieved_nodes:
        return "<i>No specific statutes retrieved.</i>"
        
    grouped = {}
    for node in retrieved_nodes:
        doc = node["doc_name"]
        if doc not in grouped:
            grouped[doc] = []
        grouped[doc].append(node["text"])
        
    html = "<ul class='legal-dir-tree'>"
    for doc, texts in grouped.items():
        html += f"<li><details open><summary>📁 <strong>{doc}</strong></summary><ul>"
        for text in texts:
            first_line = text.split('\n')[0].strip()
            title = first_line if len(first_line) < 65 else first_line[:65] + "..."
            clean_text = text.replace('\n', '<br>')
            html += f"<li><details><summary>📄 {title}</summary><div class='dir-tree-leaf'>{clean_text}</div></details></li>"
        html += "</ul></details></li>"
    html += "</ul>"
    return html

# --- 4. UI Initialization & Styling ---
st.set_page_config(page_title="LegalEdge India", page_icon="⚖️", layout="wide")

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
    /* Primary Action CTA (Download Button) */
    [data-testid="stDownloadButton"] > button {
        border-radius: 24px;
        background-color: #10B981;
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #059669;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(16, 185, 129, 0.3);
    }
    /* Chat Bubbles */
    [data-testid="stChatMessage"] { border-radius: 12px; padding: 15px; margin-bottom: 20px; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #FFFFFF; border: 1px solid #F3F4F6; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #F8FAFC; border-left: 4px solid #2563EB;
    }

    /* --- THE DIRECTORY TREE CSS --- */
    .legal-dir-tree, .legal-dir-tree ul { list-style: none; padding-left: 22px; margin: 0; }
    .legal-dir-tree { padding-left: 0; }
    .legal-dir-tree li { position: relative; padding-top: 5px; padding-bottom: 5px; }
    .legal-dir-tree li::before, .legal-dir-tree li::after { content: ''; position: absolute; left: -14px; }
    .legal-dir-tree li::before { border-top: 1px solid #CBD5E1; top: 20px; width: 12px; height: 0; }
    .legal-dir-tree li::after { border-left: 1px solid #CBD5E1; height: 100%; width: 0px; top: -5px; }
    .legal-dir-tree ul > li:last-child::after { height: 25px; }
    .legal-dir-tree details > summary::-webkit-details-marker { display: none; }
    .legal-dir-tree details > summary { list-style: none; cursor: pointer; padding: 5px 8px; border-radius: 4px; font-size: 0.95em; color: #374151; transition: background-color 0.2s; }
    .legal-dir-tree details > summary:hover { background-color: #F1F5F9; }
    .dir-tree-leaf { margin-top: 5px; margin-bottom: 5px; padding: 12px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; font-size: 0.85em; color: #475569; line-height: 1.6; max-height: 300px; overflow-y: auto; }

    /* --- CTA PILLS CSS --- */
    .profile-footer { text-align: center; margin-top: 35px; }
    .profile-text { font-size: 0.85em; color: #64748B; margin-bottom: 12px; }
    .social-pills { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
    .social-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 20px;
        font-size: 0.75em; font-weight: 600; text-decoration: none;
        color: #475569; background: #FFFFFF; border: 1px solid #E2E8F0;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .social-pill svg { width: 12px; height: 12px; fill: currentColor; }
    
    .github-pill:hover { background: #24292E; color: #FFFFFF; border-color: #24292E; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .linkedin-pill:hover { background: #0A66C2; color: #FFFFFF; border-color: #0A66C2; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(10, 102, 194, 0.2); }
    .email-pill:hover { background: #EA4335; color: #FFFFFF; border-color: #EA4335; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(234, 67, 53, 0.2); }
    .phone-pill:hover { background: #10B981; color: #FFFFFF; border-color: #10B981; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); }

    .feedback-link { display: inline-block; margin-top: 18px; font-size: 0.75em; color: #94A3B8; text-decoration: none; transition: color 0.2s; }
    .feedback-link:hover { color: #3B82F6; text-decoration: underline; }
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
    
    # --- UPGRADED DEVELOPER SIGNATURE, PHONE & PRE-FILLED EMAIL ---
    st.markdown("""
    <div class='profile-footer'>
        <p class='profile-text'>Built by <strong>Saket</strong></p>
        <div class='social-pills'>
            <a href='https://github.com/bharatisaket' target='_blank' class='social-pill github-pill'>
                <svg viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub
            </a>
            <a href='https://www.linkedin.com/in/saket-bharati-2a615a148/' target='_blank' class='social-pill linkedin-pill'>
                <svg viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                LinkedIn
            </a>
            <a href='mailto:bharatisaket@gmail.com?subject=LegalEdge%20India%20Inquiry' class='social-pill email-pill'>
                <svg viewBox="0 0 24 24"><path d="M0 3v18h24v-18h-24zm21.518 2l-9.518 7.713-9.518-7.713h19.036zm-19.518 14v-11.817l10 8.104 10-8.104v11.817h-20z"/></svg>
                Email
            </a>
            <a href='tel:+918766623773' class='social-pill phone-pill'>
                <svg viewBox="0 0 24 24"><path d="M20 22.621l-3.521-6.792c-.008.004-1.974.97-2.064 1.011-2.24 1.086-6.799-7.82-4.609-8.994l2.083-1.026-3.493-6.82c-2.106 1.039-8.938 4.8-6.68 11.225 4.24 12.028 17.027 12.446 20.707 10.113l2.423-1.189-4.846-7.528z"/></svg>
                Call
            </a>
        </div>
        <a href="mailto:YOUR_EMAIL@gmail.com?subject=Feedback%20for%20LegalEdge%20India&body=Hi%20Saket%2C%0A%0AHere%20is%20my%20feedback%20on%20LegalEdge%20India%3A%0A%0A1.%20Bug%20%2F%20Legal%20Inaccuracy%20Found%3A%0A%5BDescribe%20issue%20here%5D%0A%0A2.%20Feature%20Request%3A%0A%5BWhat%20should%20be%20added%20next%3F%5D%0A%0A3.%20My%20Role%3A%0A%5BAdvocate%20%2F%20Student%20%2F%20Citizen%5D%0A%0AThanks%21" class='feedback-link'>
            💡 Share Feedback or Suggestions
        </a>
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
        if st.button("🚨 Find Penalties", use_container_width=True): st.session_state.starter_prompt = "What is the penalty for creating a public nuisance under the new BNS?"
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

user_input = st.chat_input("E.g., Search for a specific legal section, penalty, or procedure...")
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

# --- CASE EXPORT (Main UI) ---
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 1. Build the text document cleanly
    chat_export = "# ⚖️ LegalEdge India - Case Notes\n\n"
    for msg in st.session_state.messages:
        role = "🧑‍💼 User Query" if msg["role"] == "user" else "🤖 LegalEdge Analysis"
        chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"
        
    # 2. Create a crisp SaaS export card container
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 25px; text-align: center; margin-top: 10px;">
        <h3 style="color: #1E293B; font-size: 16px; margin-top: 0; font-weight: 600;">Research Session Complete</h3>
        <p style="color: #64748B; font-size: 14px; margin-bottom: 25px;">Download a clean Markdown record of your query, statutory citations, and analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Overlap the button slightly over the card for a connected UI feel
    st.markdown("<div style='margin-top: -50px; position: relative; z-index: 10;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.download_button(
            label="📥 Export Session Notes",
            data=chat_export,
            file_name="LegalEdge_Case_Notes.md",
            mime="text/markdown",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
