import streamlit as st
import os
import json
import copy
from pageindex import PageIndexClient
import pageindex.utils as utils
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Configuration & Security ---
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

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
* **Key Elements:** Provide a bulleted list breaking down the core ingredients of the offence (e.g., Group size, Motive, Liability).

### 3. Summary Table
Always include a clean Markdown table summarizing the core findings. For example:
| Scenario | Relevant Section | Prescribed Punishment/Rule |
| :--- | :--- | :--- |
| [Data] | [Data] | [Data] |

### 4. Procedural & Legal Notes
Include procedural classifications (e.g., Cognizable, Bailable, Triable by which court) and any significant legal context or schedules found in the text.

Maintain a strictly formal, precise, and authoritative legal tone. Do not hallucinate outside the provided text. If the text does not contain the answer, explicitly state that it is not present in the codes.
"""

# --- 2. UI Initialization ---
st.set_page_config(page_title="Bharatiya Laws AI", page_icon="⚖️")
st.title("Bharatiya Laws AI Assistant")
st.caption("Powered by Vectorless RAG & Gemini 1.5 Pro")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Main Logic Loop ---
if prompt := st.chat_input("E.g., What is the penalty for mob lynching?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            message_placeholder.markdown("Executing Vectorless Tree Search...")
            retrieved_texts = []
            
            for doc_id in ALL_LAW_DOC_IDS:
                # STEP 1: Fetch the Tree Index (Direct method on the client)
                tree_response = pi_client.get_tree(doc_id, node_summary=True)
                
                # The API returns a dict with 'result', grab the actual tree
                tree = tree_response.get("result", tree_response) if isinstance(tree_response, dict) else tree_response
                
                # Create a local map to look up node text instantly (Zero additional API calls)
                node_mapping = utils.create_node_mapping(tree)
                
                # Remove the full text from a copy of the tree to save Gemini context tokens
                clean_tree = copy.deepcopy(tree)
                tree_without_text = utils.remove_fields(clean_tree, fields=['text'])
                tree_json = json.dumps(tree_without_text)
                
                # STEP 2: The Routing Prompt
                routing_prompt = f"""
                Analyze this document tree and the user query.
                Return ONLY a valid JSON array of the most relevant node IDs.
                Example: ["N001", "N003"]
                
                Tree: {tree_json}
                Query: {prompt}
                """
                
                route_response = llm.invoke(routing_prompt)
                raw_content = route_response.content
                
                # Scenario 1: LangChain returned a list of content blocks
                if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict) and "text" in raw_content[0]:
                    raw_content = raw_content[0]["text"]
                
                # Parse the nodes based on what LangChain handed back
                if isinstance(raw_content, list):
                    # Scenario 2: Gemini auto-parsed the JSON into a Python list for us
                    selected_nodes = raw_content
                else:
                    # Scenario 3: It's a normal string, so clean and parse it
                    cleaned = raw_content.replace("```json", "").replace("```", "").strip()
                    selected_nodes = json.loads(cleaned)
                
                # STEP 3: Fetch the raw text directly from our local mapping
                for node_id in selected_nodes:
                    if node_id in node_mapping and 'text' in node_mapping[node_id]:
                        retrieved_texts.append(node_mapping[node_id]['text'])
                    
            message_placeholder.markdown("Reasoning with Gemini...")
            
            # STEP 4: Final Generation with Gemini (Zero Vectify Credits)
            context_text = "\n\n".join(retrieved_texts)
            gemini_prompt = f"RETRIEVED LEGAL TEXT:\n{context_text}\n\nUSER QUESTION:\n{prompt}"
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=gemini_prompt)
            ]
            
            final_response = llm.invoke(messages)
            raw_answer = final_response.content
            
            # Extract the actual text if LangChain wrapped it in a list with metadata
            if isinstance(raw_answer, list) and len(raw_answer) > 0 and isinstance(raw_answer[0], dict) and "text" in raw_answer[0]:
                answer = raw_answer[0]["text"]
            else:
                answer = str(raw_answer)
            
            # Display final answer
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except json.JSONDecodeError:
            st.error("Error: The routing engine failed to format the node IDs correctly. Please try asking again.")
        except Exception as e:
            st.error(f"Error executing Vectorless RAG: {e}")
