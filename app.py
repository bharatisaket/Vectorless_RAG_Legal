import streamlit as st
import os
import json
import copy
from pageindex import PageIndexClient
import pageindex.utils as utils
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 1. Configuration & Security ---
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Using Flash to stay within Free Tier limits 
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)

ALL_LAW_DOC_IDS = [
    "pi-cmoo3zavg011p01qr7mbgdtev", # BNS
    "pi-cmoo3z9av011n01qrcozuk72f", # BSA
    "pi-cmoo55m2w012301qrevyj12j4"  # BNSS
]

SYSTEM_PROMPT = """You are an elite Indian Legal AI Assistant specializing in the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). 

Your primary directive is to provide highly structured, comprehensive, and perfectly formatted legal analysis based STRICTLY on the retrieved context. 

Whenever a user asks about an offence, penalty, procedure, or legal concept, you MUST format your response using the exact structure below:

### 1. Executive Summary
Provide a 1-2 sentence high-level overview of how the law treats the user's query.

### 2. Statutory Breakdown
For every relevant provision retrieved, you must clearly list:
* **Law & Chapter:** (e.g., Bharatiya Nyaya Sanhita, 2023 - Chapter VI)
* **Section:** (e.g., Section 103(2))
* **Statutory Text:** "Extract and output the EXACT verbatim quote from the retrieved text inside quotation marks."
* **Key Elements:** Provide a bulleted list breaking down the core ingredients of the offence.

### 3. Summary Table
Always include a clean Markdown table summarizing the core findings.

### 4. Procedural & Legal Notes
Include procedural classifications (e.g., Cognizable, Bailable, Triable by which court) and any significant legal context or schedules found in the text.

Maintain a strictly formal, precise, and authoritative legal tone. Do not hallucinate outside the provided text. If the text does not contain the answer, explicitly state that it is not present in the codes.
"""

# --- 2. Caching Engine ---
@st.cache_data(show_spinner="Loading Legal Codes into memory...")
def fetch_law_trees():
    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cached_data = {}
    for doc_id in ALL_LAW_DOC_IDS:
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

# --- 3. UI Initialization & History Rendering ---
st.set_page_config(page_title="Bharatiya Laws AI", page_icon="⚖️")
st.title("Bharatiya Laws AI Assistant")
st.caption("Powered by Vectorless RAG & Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Update: Render chat history along with the saved expander context
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(message["context"])

# --- 4. Main Logic Loop ---
if prompt := st.chat_input("E.g., What is the penalty for mob lynching?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            message_placeholder.markdown("Executing Vectorless Tree Search...")
            retrieved_texts = []
            
            chat_history_str = ""
            history_subset = st.session_state.messages[-5:-1] 
            for msg in history_subset:
                chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
            for doc_id in ALL_LAW_DOC_IDS:
                tree_data = legal_trees[doc_id]
                tree_json = tree_data["tree_json"]
                node_mapping = tree_data["mapping"]
                
                routing_prompt = f"""
                Analyze this document tree, the conversation history, and the user's latest query.
                Return ONLY a valid JSON array of the most strictly relevant node IDs.
                
                CRITICAL INSTRUCTIONS:
                1. ONLY select granular "leaf" nodes (specific Sections or Sub-sections).
                2. NEVER select high-level "Chapter" or "Part" parent nodes (doing so will crash the system with too much text).
                3. STRICT LIMIT: Return a maximum of 3 node IDs.
                
                Example: ["N001", "N003"]
                
                Conversation History:
                {chat_history_str}
                
                Tree: {tree_json}
                Latest Query: {prompt}
                """
                
                route_response = llm.invoke(routing_prompt)
                raw_content = route_response.content
                
                if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict) and "text" in raw_content[0]:
                    raw_content = raw_content[0]["text"]
                
                if isinstance(raw_content, list):
                    selected_nodes = raw_content
                else:
                    cleaned = raw_content.replace("```json", "").replace("```", "").strip()
                    selected_nodes = json.loads(cleaned) if cleaned else []
                
                for node_id in selected_nodes:
                    if node_id in node_mapping and 'text' in node_mapping[node_id]:
                        retrieved_texts.append(node_mapping[node_id]['text'])
                    
            message_placeholder.markdown("Reasoning with Gemini...")
            
            context_text = "\n\n".join(retrieved_texts)
            if not retrieved_texts:
                context_text = "No specific statutes retrieved for this query."
            
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
                    
            gemini_prompt = f"RETRIEVED LEGAL TEXT:\n{context_text}\n\nLATEST USER QUESTION:\n{prompt}"
            messages.append(HumanMessage(content=gemini_prompt))
            
            final_response = llm.invoke(messages)
            raw_answer = final_response.content
            
            if isinstance(raw_answer, list) and len(raw_answer) > 0 and isinstance(raw_answer[0], dict) and "text" in raw_answer[0]:
                answer = raw_answer[0]["text"]
            else:
                answer = str(raw_answer)
            
            # Display final answer
            message_placeholder.markdown(answer)
            
            # Update: Render the expander for the current turn
            with st.expander("View Retrieved Legal Statutes"):
                st.markdown(context_text)
            
            # Update: Save the context in the session state so it persists
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "context": context_text
            })
            
        except json.JSONDecodeError:
            st.error("Error: The routing engine failed to format the node IDs correctly.")
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                st.warning("⏳ The AI is currently handling too many requests. Please wait a moment and try again.")
            else:
                st.error(f"Error executing Vectorless RAG: {e}")
