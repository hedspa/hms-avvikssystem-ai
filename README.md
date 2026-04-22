# PPE Detection MVP

En enkel Python-startpakke for studentprosjekt som oppdager personlig verneutstyr i bilder fra byggeplass. MVP-en håndterer tre klasser:

- `person`
- `helmet`
- `vest`

Systemet bruker Ultralytics YOLO til objektgjenkjenning og en enkel regelmotor til å avgjøre om hver person bruker hjelm og vest.

## Hva prosjektet gjør

For hver detektert person:

1. Person-boksen deles i en `head-region`.
2. Person-boksen deles i en `torso-region`.
3. `helmet` må ligge i head-region for å telle som hjelm på personen.
4. `vest` må ligge i torso-region for å telle som vest på personen.
5. Resultatet lagres som JSON og som et annotert bilde.

Eksempel på output:

```json
[
  {
    "person_id": 1,
    "helmet": true,
    "vest": false,
    "deviation": ["no-vest"]
  }
]
```

## Mappestruktur

```text
data/
  dataset.yaml
src/
  config.py
  utils.py
  rules.py
  train.py
  predict.py
  main.py
requirements.txt
README.md
```

## Dataset-format

`data/dataset.yaml` peker på et YOLO-datasett med denne strukturen:

```text
data/
  ppe_dataset/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```

Klassenavnene i label-settet må være:

```text
0 person
1 helmet
2 vest
```

## Installasjon

Opprett og aktiver et virtuelt miljø, og installer avhengigheter:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Trening

Standard trening:

```powershell
python -m src.main train
```

Eksempel med egne parametere:

```powershell
python -m src.main train --model yolov8n.pt --data data/dataset.yaml --epochs 100 --imgsz 640 --project runs/ppe --name exp1
```

Etter trening ligger vektene typisk her:

```text
runs/ppe/exp1/weights/best.pt
```

## Prediksjon

Kjør inferens på ett bilde:

```powershell
python -m src.main predict --image path\to\image.jpg --weights runs/ppe\train\weights\best.pt
```

Valgfri output-mappe:

```powershell
python -m src.main predict --image path\to\image.jpg --weights runs/ppe\train\weights\best.pt --output-dir outputs
```

Resultatet blir:

- JSON-fil med ett objekt per person
- annotert bilde med personboks, head-region og torso-region

## Filansvar

- `src/config.py`: felles konfigurasjon og konstanter
- `src/utils.py`: fil- og bildehjelpere
- `src/rules.py`: regelmotor for kobling mellom person, hjelm og vest
- `src/train.py`: YOLO-trening
- `src/predict.py`: inferens, JSON-output og annotert bilde
- `src/main.py`: enkel CLI for trening og prediksjon

## Videre utvidelse

Koden er bevisst enkel og delt opp slik at du senere kan legge til flere klasser, for eksempel `glasses`, ved å:

1. Legge til klassen i datasettet.
2. Oppdatere konfigurasjon og regler.
3. Utvide output-formatet per person.

## Begrensninger i MVP

- Regelmotoren kobler utstyr til person basert på om senterpunktet til utstyrsboksen ligger inne i riktig region.
- Ved overlappende personer eller vanskelige vinkler kan koblingen bli feil.
- Kvaliteten avhenger direkte av hvor godt YOLO-modellen er trent.
