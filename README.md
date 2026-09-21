# Independent Insider Radar LIVE v1.2

Versione operativa con **Forward Registry automatico**.

## Regole congelate
- WATCH = primo cluster con almeno 2 insider distinti
- CORE = primo cluster con almeno 3 insider distinti
- VALUE 100–250k (2026$) = tag informativo, non filtro obbligatorio
- Orizzonte empirico principale = 5 sedute

## Novità v1.2 — Forward Registry
La v1.2 salva automaticamente ogni CORE/WATCH con contesto completo in `data/live_v1/forward_registry_v1_2.csv`.

Per evitare di chiamare *forward* ciò che era già noto prima della v1.2:
- i segnali già presenti al primo avvio vengono marcati **BASELINE**;
- soltanto i segnali comparsi dopo l'inizializzazione vengono marcati **FORWARD**;
- quando un segnale raggiunge 5 sedute e dispone di `return_5` ed `excess_5`, l'esito viene **congelato** e non viene riscritto dai successivi refresh Yahoo.

Il registro conserva anche entry, stato corrente, CORE/WATCH, VALUE, ruolo CEO/CFO, link SEC e TradingView.

## Flusso quotidiano
1. `Sync / continua SEC Live` fino a Pendenti = 0.
2. `Costruisci / aggiorna Radar`.
3. `Aggiorna prezzi / performance`.
4. La sezione **Forward Registry — v1.2** si aggiorna automaticamente.
5. Scarica periodicamente `insider_forward_registry_v1_2.csv` e il backup ZIP.

## Backup
Il backup stato live include ora anche:
- `forward_registry_v1_2.csv`
- `forward_registry_meta_v1_2.json`

I vecchi backup v1.0/v1.1 restano importabili; se non contengono un registry, il primo avvio v1.2 crea una nuova BASELINE.

## Deploy Streamlit
Sostituire insieme:
- `app.py`
- `engine.py`
- `live_engine.py`
- `requirements.txt`

Lo stato dati resta nella stessa cartella `data/live_v1`, quindi un deploy sopra la v1.1 può riutilizzare radar e prezzi già presenti finché lo storage della piattaforma li conserva.

## Test
9 test automatici superati, inclusi:
- parsing SEC e filtri P/A;
- costruzione FIRST CORE;
- bucket operativi;
- inizializzazione BASELINE;
- registrazione di nuovi segnali FORWARD;
- idempotenza del registro;
- congelamento immutabile del risultato a 5 sedute;
- separazione BASELINE/FORWARD nelle statistiche.

CORE/WATCH restano classificazioni quantitative del pattern studiato, non raccomandazioni di investimento.
