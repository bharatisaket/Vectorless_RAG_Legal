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

SYSTEM_PROMPT = "You are an expert Indian Legal AI Assistant specializing in the new criminal laws. Always base your answers strictly on the provided context. You must explicitly cite the Law, Chapter, and Section/Sub-section. Maintain a formal, precise legal tone."

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
                
                # Clean the response to ensure it's just pure JSON
                cleaned_response = route_response.content.replace("```json", "").replace("```", "").strip()
                selected_nodes = json.loads(cleaned_response)
                
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
            answer = final_response.content
            
            # Display final answer
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except json.JSONDecodeError:
            st.error("Error: The routing engine failed to format the node IDs correctly. Please try asking again.")
        except Exception as e:
            st.error(f"Error executing Vectorless RAG: {e}")
