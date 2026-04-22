from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import v2

from db import (
    create_report,
    get_all_reports,
    get_reports_for_review,
    reject_report,
    report_with_comment,
    assign_report_to_person,
    get_reports_for_person,
    close_report,
    delete_report,
)

def format_status(status: str) -> str:
    if status == "Til vurdering":
        return "Til vurdering av HMS-ansvarlig"
    elif status == "Rapportert":
        return "Avvik registrert"
    elif status == "Sendt til person":
        return "Tildelt ansvarlig"
    elif status == "Avvist":
        return "Avvist av HMS-ansvarlig"
    elif status.startswith("Lukket av"):
        return status
    else:
        return status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ppe_model.pth"
UPLOAD_DIR = PROJECT_ROOT / "webapp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

transform = v2.Compose([
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_image(model, pil_image: Image.Image):
    image_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.sigmoid(output)[0]

    helmet_conf = float(probs[0].item())
    vest_conf = float(probs[1].item())
    glasses_conf = float(probs[2].item())

    helmet = 1 if helmet_conf >= 0.5 else 0
    vest = 1 if vest_conf >= 0.5 else 0
    glasses = 1 if glasses_conf >= 0.5 else 0

    return {
        "helmet": helmet,
        "vest": vest,
        "glasses": glasses,
        "helmet_conf": helmet_conf,
        "vest_conf": vest_conf,
        "glasses_conf": glasses_conf,
    }


def build_deviation_text(result: dict) -> str:
    missing = []

    if result["helmet"] == 0:
        missing.append("hjelm")
    if result["vest"] == 0:
        missing.append("vest")
    if result["glasses"] == 0:
        missing.append("vernebriller")

    if not missing:
        return "Ingen avvik oppdaget"
    if len(missing) == 1:
        return f"Mangler {missing[0]}"
    if len(missing) == 2:
        return f"Mangler {missing[0]} og {missing[1]}"
    return f"Mangler {missing[0]}, {missing[1]} og {missing[2]}"


def page_innsending():
    st.title("Send inn bilde")
    st.write("Velg om du vil laste opp et bilde eller ta et bilde.")

    if "capture_mode" not in st.session_state:
        st.session_state.capture_mode = None

    col1, col2 = st.columns(2)

    if col1.button("Last opp bilde", use_container_width=True):
        st.session_state.capture_mode = "upload"

    if col2.button("Ta bilde", use_container_width=True):
        st.session_state.capture_mode = "camera"

    file_to_use = None

    if st.session_state.capture_mode == "upload":
        uploaded_file = st.file_uploader(
            "Velg bilde",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_file"
        )
        if uploaded_file is not None:
            file_to_use = uploaded_file

    elif st.session_state.capture_mode == "camera":
        camera_file = st.camera_input("Ta bilde", key="camera_file")
        if camera_file is not None:
            file_to_use = camera_file

    if file_to_use is not None:
        image = Image.open(file_to_use)
        st.image(image, caption="Valgt bilde", use_container_width=True)

        if st.button("Send inn", type="primary"):
            model = load_model()
            result = predict_image(model, image)
            deviation = build_deviation_text(result)

            save_path = UPLOAD_DIR / file_to_use.name
            image.save(save_path)

            create_report(
                image_name=file_to_use.name,
                image_path=str(save_path),
                helmet=result["helmet"],
                vest=result["vest"],
                glasses=result["glasses"],
                helmet_conf=result["helmet_conf"],
                vest_conf=result["vest_conf"],
                glasses_conf=result["glasses_conf"],
                deviation=deviation,
                comment="",
            )

            st.success("Bildet er sendt inn. Forslaget er lagt til under Foreslåtte avvik.")
            st.session_state.capture_mode = None
            st.rerun()

from db import (
    create_report,
    get_all_reports,
    get_reports_for_review,
    reject_report,
    report_with_comment,
    assign_report_to_person,
    get_reports_for_person,
    close_report,
    delete_report,
)

def page_foreslatte_avvik():
    st.title("Foreslåtte avvik")
    st.write("Her vurderer HMS-ansvarlig forslag som AI-modellen har opprettet.")

    all_reports = get_all_reports()

    pending_reports = [r for r in all_reports if r[4] == "Til vurdering"]

    if not pending_reports:
        st.success("Ingen saker venter på vurdering.")
        return

    st.subheader("Til vurdering")

    for report in pending_reports:
        report_id, image_name, image_path, deviation, status, comment, created_at, assigned_to = report

        with st.expander(f"Sak {report_id} – {deviation}"):
            st.markdown(f"### Sak {report_id} – {deviation}")

            image_file = Path(image_path)
            if image_file.exists():
                st.image(str(image_file), caption=image_name, use_container_width=True)

            if deviation == "Ingen avvik oppdaget":
                st.success("✅ Ingen avvik oppdaget")
            else:
                st.error(f"🚨 {deviation}")

            st.write(f"**Opprettet:** {created_at}")

            hms_comment = st.text_area(
                "Kommentar fra HMS-ansvarlig",
                value=comment if comment else "",
                key=f"comment_{report_id}"
            )

            selected_person = st.selectbox(
                "Tildel til person",
                ["Person 1", "Person 2", "Person 3"],
                key=f"assign_select_{report_id}"
            )

            col1, col2 = st.columns(2)

            if col1.button("Avvis forslag", key=f"reject_{report_id}"):
                reject_report(report_id, hms_comment)
                st.rerun()

            if col2.button("Bekreft og tildel", key=f"confirm_assign_{report_id}"):
                assign_report_to_person(report_id, selected_person, hms_comment)
                st.rerun()

def page_avviksoversikt():
    st.title("Avviksoversikt")
    st.write("Alle registrerte saker vises her, delt inn i åpne, lukkede og avviste avvik.")

    filter_status = st.selectbox(
        "Filtrer visning",
        ["Alle", "Åpne", "Lukkede", "Avviste"]
    )

    reports = get_all_reports()

    if not reports:
        st.info("Ingen saker registrert ennå.")
        return

    open_reports = []
    closed_reports = []
    rejected_reports = []

    for report in reports:
        status = report[4]

        if status in ["Rapportert", "Sendt til person"]:
            open_reports.append(report)
        elif status.startswith("Lukket av"):
            closed_reports.append(report)
        elif status == "Avvist":
            rejected_reports.append(report)
        # "Til vurdering" ignoreres her og vises bare under Foreslåtte avvik

    # -------------------------
    # ÅPNE AVVIK
    # -------------------------
    if filter_status in ["Alle", "Åpne"]:
        st.subheader(f"Åpne avvik ({len(open_reports)})")

        if not open_reports:
            st.info("Ingen åpne avvik.")
        else:
            for report in open_reports:
                report_id, image_name, image_path, deviation, status, comment, created_at, assigned_to = report

                with st.expander(f"Sak {report_id} – {deviation}"):
                    image_file = Path(image_path)
                    if image_file.exists():
                        st.image(str(image_file), caption=image_name, use_container_width=True)

                    if deviation == "Ingen avvik oppdaget":
                        st.success("✅ Ingen avvik oppdaget")
                    else:
                        st.error(f"🚨 {deviation}")

                    if status == "Rapportert":
                        st.warning(format_status(status))
                    elif status == "Sendt til person":
                        st.warning(format_status(status))
                    else:
                        st.write(format_status(status))

                    if assigned_to:
                        st.write(f"**Tildelt til:** {assigned_to}")

                    st.write(f"**Kommentar:** {comment if comment else 'Ingen kommentar'}")
                    st.write(f"**Opprettet:** {created_at}")

                    if st.button("Slett (kun demo)", key=f"delete_open_{report_id}"):
                        delete_report(report_id)
                        st.rerun()

        st.markdown("---")

    # -------------------------
    # LUKKEDE AVVIK
    # -------------------------
    if filter_status in ["Alle", "Lukkede"]:
        st.subheader(f"Lukkede avvik ({len(closed_reports)})")

        if not closed_reports:
            st.info("Ingen lukkede avvik.")
        else:
            for report in closed_reports:
                report_id, image_name, image_path, deviation, status, comment, created_at, assigned_to = report

                with st.expander(f"Sak {report_id} – {deviation}"):
                    image_file = Path(image_path)
                    if image_file.exists():
                        st.image(str(image_file), caption=image_name, use_container_width=True)

                    if deviation == "Ingen avvik oppdaget":
                        st.success("✅ Ingen avvik oppdaget")
                    else:
                        st.error(f"🚨 {deviation}")

                    st.success(format_status(status))

                    if assigned_to:
                        st.write(f"**Tildelt til:** {assigned_to}")

                    st.write(f"**Kommentar:** {comment if comment else 'Ingen kommentar'}")
                    st.write(f"**Opprettet:** {created_at}")

                    if st.button("Slett (kun demo)", key=f"delete_closed_{report_id}"):
                        delete_report(report_id)
                        st.rerun()

        st.markdown("---")

    # -------------------------
    # AVVISTE AVVIK
    # -------------------------
    if filter_status in ["Alle", "Avviste"]:
        st.subheader(f"Avviste avvik ({len(rejected_reports)})")

        if not rejected_reports:
            st.info("Ingen avviste avvik.")
        else:
            for report in rejected_reports:
                report_id, image_name, image_path, deviation, status, comment, created_at, assigned_to = report

                with st.expander(f"Sak {report_id} – {deviation}"):
                    image_file = Path(image_path)
                    if image_file.exists():
                        st.image(str(image_file), caption=image_name, use_container_width=True)

                    if deviation == "Ingen avvik oppdaget":
                        st.success("✅ Ingen avvik oppdaget")
                    else:
                        st.error(f"🚨 {deviation}")

                    st.error(format_status(status))

                    if assigned_to:
                        st.write(f"**Tildelt til:** {assigned_to}")

                    st.write(f"**Kommentar:** {comment if comment else 'Ingen kommentar'}")
                    st.write(f"**Opprettet:** {created_at}")

                    if st.button("Slett (kun demo)", key=f"delete_rejected_{report_id}"):
                        delete_report(report_id)
                        st.rerun()

def page_personinnboks():
    st.title("Innboks")
    st.write("Her kan du velge hvem sin innboks du vil se.")

    selected_person = st.selectbox(
        "Velg person",
        ["Person 1", "Person 2", "Person 3"]
    )

    reports = get_reports_for_person(selected_person)

    # Vis bare saker som faktisk fortsatt er åpne for personen
    active_reports = [r for r in reports if r[4] == "Sendt til person"]

    if not active_reports:
        st.info(f"Ingen åpne saker er sendt til {selected_person}.")
        return

    st.subheader(f"Åpne saker for {selected_person}")

    for report in active_reports:
        report_id, image_name, image_path, deviation, status, comment, created_at, assigned_to = report

        with st.expander(f"Sak {report_id} – {deviation}"):

            image_file = Path(image_path)
            if image_file.exists():
                st.image(str(image_file), caption=image_name, use_container_width=True)

            if deviation == "Ingen avvik oppdaget":
                st.success("✅ Ingen avvik oppdaget")
            else:
                st.error(f"🚨 {deviation}")

            st.write(f"**Status:** {status}")
            st.write(f"**Kommentar:** {comment if comment else 'Ingen kommentar'}")
            st.write(f"**Opprettet:** {created_at}")
            st.write(f"**Tildelt til:** {assigned_to}")

            if st.button("Lukk avvik", key=f"close_{report_id}"):
                close_report(report_id, selected_person)
                st.rerun()


def page_om_modell():
    st.title("Om AI-modellen")

    st.write("""
    Denne AI-modellen analyserer bilder og vurderer om en person bruker:
    - Hjelm
    - Vest 
    - Vernebriller
    """)

    st.write("""
    Modellen er kun trent på disse tre typene verneutstyr.
    Andre typer verneutstyr (som hansker, vernesko og hørselsvern) er ikke inkludert, og avvik knyttet til dette må derfor registreres manuelt.
    AI-modellen har en treffsikkerhet på 60-90%, og er designet som et hjelpemiddel for å øke rapporteringsgraden ved å komme med foreslåtte avvik. 
    """)

    st.write("""
    Løsningen er basert på reelle HMS-krav til personlig verneutstyr på byggeplass fra Oslobygg KF sitt faktaark. 
    For mer informasjon om PVU, last ned:
    """)
    # 👇 Sørg for at PDF ligger i samme mappe som app.py
    with open("webapp/Faktaark PVU - Oslobygg KF.pdf", "rb") as f:
        st.download_button(
            "Faktaark PVU - Oslobygg KF",
            f,
            file_name="Faktaark PVU - Oslobygg KF.pdf"
        )

    st.write("""
    
    """)

    st.subheader(f"Kontakt oss")
    st.write("""
    Har du spørsmål om løsningen? Ta gjerne kontakt:
    """)

    col1, col2, right = st.columns([2, 2, 3])

    with col1:
        st.image("webapp/heddasp.jpg", width=120)
        st.markdown("**Hedda Spangberg**")
        st.markdown("heddasp@uia.no")

    with col2:
        st.image("webapp/matildenk.jpg", width=120)
        st.markdown("**Matilde Nyheim Kristoffersen**")
        st.markdown("matildenk@uia.no")




def page_forside():
    st.image("webapp/banner.png", width=1000)
    st.title("Rapportering av HMS-avvik med AI")

    st.write("""
    Dette systemet bruker kunstig intelligens til å oppdage manglende verneutstyr på byggeplass. 
    
    Et bilde av en person sendes inn og analyseres automatisk. Dersom modellen oppdager manglende verneutstyr, opprettes et forslag til avvik.
    Forslaget kvalitetssikres av HMS-ansvarlig, som kan velge å registrere avviket og tildele det til ansvarlig person. 
    Den ansvarlige mottar deretter et varsel i sin innboks, og når avviket er rettet, kan saken lukkes.
    
    Systemet gir en oversikt over åpne, lukkede og avviste avvik, noe som bidrar til økt transparens.
    """)

    st.image("webapp/systemet.png", width=1000)







