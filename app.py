import streamlit as st
import os
import json
from pageindex import PageIndexClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Configuration & Security ---
# Pull keys securely from Streamlit Secrets
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Initialize the Gemini Brain and the PageIndex Client
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# The Document IDs for the BNS, BSA, and BNSS
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
                # STEP 1: Fetch the Tree Index (0 credits - you already paid to index this)
                tree = pi_client.documents.get_tree(doc_id)
                tree_json = tree.to_json()
                
                # STEP 2: The Routing Prompt (Gemini reads the map and chooses the nodes)
                routing_prompt = f"""
                Analyze this document tree and the user query.
                Return ONLY a valid JSON array of the most relevant node IDs.
                Example: ["N001", "N003"]
                
                Tree: {tree_json}
                Query: {prompt}
                """
                
                route_response = llm.invoke(routing_prompt)
                
                # Clean the response to ensure it's just pure JSON
                cleaned_response = route_response.content.replace("
```json", "").replace("```", "").strip()
                selected_nodes = json.loads(cleaned_response)
                
                # STEP 3: Fetch the raw text for the specifically selected nodes
                for node_id in selected_nodes:
                    node_data = pi_client.documents.get_node(doc_id, node_id)
                    retrieved_texts.append(node_data.text)
                    
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
