"""Streamlit interface:  streamlit run app.py"""
import cv2
import streamlit as st

from netra.agent import run_query
from netra.common import BASE, IMAGE_IN
from netra.image_tools import assess_image_quality, enhance_low_light

st.set_page_config(page_title="NetraAI", layout="wide")
st.title("NetraAI: assistive-vision image and video processing")
st.caption("Research prototype. Not a certified mobility or safety device.")

upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
filename = None
if upload:
    filename = f"upload_{abs(hash(upload.name)) % 10**8}.{upload.name.rsplit('.', 1)[-1].lower()}"
    (IMAGE_IN / filename).write_bytes(upload.getvalue())
    st.sidebar.image(cv2.cvtColor(cv2.imread(str(IMAGE_IN / filename)), cv2.COLOR_BGR2RGB))
    st.sidebar.write(f"Stored as `{filename}`")

tab_tools, tab_chat = st.tabs(["Tools", "Chat agent"])

with tab_tools:
    if not filename:
        st.info("Upload an image in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        if c1.button("Assess quality"):
            st.json(assess_image_quality(filename))
        if c2.button("Enhance low light"):
            res = enhance_low_light(filename, gamma=1.8)
            st.json(res)
            st.image(cv2.cvtColor(cv2.imread(str(BASE / res["output"])), cv2.COLOR_BGR2RGB))
        if st.button("Detect objects and speak guidance"):
            from netra.pipeline import navigation_assist
            res = navigation_assist(filename)
            st.subheader(res["guidance"])
            st.image(cv2.cvtColor(cv2.imread(str(BASE / res["annotated"])), cv2.COLOR_BGR2RGB))
            if res["audio"]:
                st.audio(str(BASE / res["audio"]))

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for m in st.session_state.history:
        role = m["role"] if isinstance(m, dict) else m.role
        text = m["content"] if isinstance(m, dict) else m.content
        if role in {"user", "assistant"} and text:
            st.chat_message(role).write(text)
    prompt = st.chat_input("e.g. Assess dark_scene.png and enhance it if needed")
    if prompt:
        st.chat_message("user").write(prompt)
        with st.spinner("Working..."):
            answer, st.session_state.history = run_query(prompt, st.session_state.history[-24:])
        st.chat_message("assistant").write(answer)
