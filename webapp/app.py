import streamlit as st

from db import get_pending_count, init_database
from pages_logic import (
    page_forside,
    page_innsending,
    page_foreslatte_avvik,
    page_avviksoversikt,
    page_personinnboks,
    page_om_modell,
)

init_database()

pending_count = get_pending_count()

st.set_page_config(
    page_title="HMS-avvikssystem",
    page_icon="🦺",
    layout="centered",
)

with st.sidebar:
    st.image("webapp/logo.png", use_container_width=True)
    st.markdown("---")


innsending_page = st.Page(
    page_innsending,
    title="Send inn bilde",
    icon="📤",
)

foreslatte_page = st.Page(
    page_foreslatte_avvik,
    title=f"Foreslåtte avvik ({pending_count})",
    icon="🚨",
)

oversikt_page = st.Page(
    page_avviksoversikt,
    title="Avviksoversikt",
    icon="📋",
)

personinnboks_page = st.Page(
    page_personinnboks,
    title="Innboks",
    icon="📥",
)

forside_page = st.Page(
    page_forside,
    title="Forside",
    icon="🏠",
)

om_modell_page = st.Page(
    page_om_modell,
    title="Om AI-modellen",
    icon="ℹ️",
)

pg = st.navigation(
    [forside_page, innsending_page, foreslatte_page, oversikt_page, personinnboks_page, om_modell_page],
    position="sidebar",
)

pg.run()