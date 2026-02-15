import streamlit as st
import requests
import uuid
import time

# --- CONFIG ---
API_URL = "http://localhost:8000/chat"
st.set_page_config(page_title="Diet & Cheat AI", page_icon="🥗", layout="centered")

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "اهلا بيك يا بطل! اقدر اساعدك ازاي النهاردة في الدايت او التمرين؟"}
    ]

# --- UI HEADER ---
st.title("🥗 Diet & Cheat AI Coach")
st.markdown("---")

# --- CHAT DISPLAY ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT HANDLER ---
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # 1. Add User Message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get Bot Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Captain is thinking..."):
                payload = {"message": prompt, "session_id": st.session_state["session_id"]}
                resp = requests.post(API_URL, json=payload, timeout=60)
                
                if resp.status_code == 200:
                    data = resp.json()
                    full_response = data["answer"]
                    
                    # Show sources if available
                    if data.get("sources"):
                        full_response += "\n\n**Sources:**\n"
                        for s in data["sources"]:
                            full_response += f"- *{s['source']}*: {s['snippet'][:50]}...\n"
                            
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.error(f"Error: {resp.text}")
        except Exception as e:
            message_placeholder.error(f"Connection Error: Is the API running? ({e})")

    # 3. Add Assistant Message to History
    if full_response:
        st.session_state["messages"].append({"role": "assistant", "content": full_response})

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("Debug Info")
    st.text(f"Session ID: {st.session_state['session_id'][:8]}...")
    if st.button("New Session"):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()
