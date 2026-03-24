import os

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8080/")


st.set_page_config(page_title="Cliente API", page_icon="API", layout="centered")

st.title("Interfaz para llamar a la API")
st.write("Esta pantalla realiza una llamada real al endpoint configurado.")

api_url = st.text_input("URL del endpoint", value=DEFAULT_API_URL)

if st.button("llamar a api", type="primary"):
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"No se pudo llamar a la API: {exc}")
    else:
        st.success("api llamada exitosamente")
        st.write("Respuesta del endpoint:")
        st.code(response.text)
