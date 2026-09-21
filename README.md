# Independent Insider Radar LIVE v1.3.2.2

Versione operativa con **Forward Registry automatico**.

## Regole congelate
- WATCH = primo cluster con almeno 2 insider distinti
- CORE = primo cluster con almeno 3 insider distinti
- VALUE 100–250k (2026$) = tag informativo, non filtro obbligatorio
- Orizzonte empirico principale = 5 sedute

## Forward Registry
La v1.3.2 salva automaticamente ogni CORE/WATCH con contesto completo in `data/live_v1/forward_registry_v1_2.csv`.

Per evitare di chiamare *forward* ciò che era già noto prima della v1.3.2.2:
- i segnali già presenti al primo avvio vengono marcati **BASELINE**;
- soltanto i segnali comparsi dopo l'inizializzazione vengono marcati **FORWARD**;
- quando un segnale raggiunge 5 sedute e dispone di `return_5` ed `excess_5`, l'esito viene **congelato** e non viene riscritto dai successivi refresh Yahoo.

Il registro conserva anche entry, stato corrente, CORE/WATCH, VALUE, ruolo CEO/CFO, link SEC e TradingView.

## Flusso quotidiano
1. `Sync / continua SEC Live` fino a Pendenti = 0.
2. `Costruisci / aggiorna Radar`.
3. `Aggiorna prezzi / performance`.
4. La sezione **Forward Registry — v1.3.2** si aggiorna automaticamente.
5. Scarica periodicamente `insider_forward_registry_v1_2.csv` e il backup ZIP.

## Backup
Il backup stato live include ora anche:
- `forward_registry_v1_2.csv`
- `forward_registry_meta_v1_2.json`

I vecchi backup v1.0/v1.1 restano importabili; se non contengono un registry, il primo avvio v1.3.2.2 crea una nuova BASELINE.

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


## Novità v1.3.2.2 — gerarchia visiva

La v1.3.2.2 non cambia l'algoritmo. Migliora solo la leggibilità operativa:

- guida **Uso quotidiano** nella sidebar;
- promemoria del flusso in alto: Sync → Radar → Prezzi → CORE;
- nuova sezione **🎯 DA GUARDARE OGGI** con soli **NUOVI CORE + CORE ATTIVI (1–4/5)**;
- WATCH e COMPLETATI restano visibili nelle sezioni successive ma non occupano la lista primaria;
- link TV e SEC direttamente nella lista del giorno.

Interpretazione: CORE (≥3 insider) è la priorità di monitoraggio; WATCH (2 insider) è una preallerta; COMPLETATI alimentano soprattutto il Forward Registry. Nessuna di queste etichette è una raccomandazione di investimento.


### Gerarchia della schermata
- **🎯 DA GUARDARE OGGI** è la lista primaria: solo NUOVI CORE + CORE ATTIVI 1–4/5.
- Subito sotto compare un richiamo esplicito: **Questa è la lista principale da controllare oggi.**
- La sezione **Dettaglio Radar** contiene NUOVI / ATTIVI / COMPLETATI / anomalie e serve a spiegare lo stato dei segnali, non a creare una seconda priorità.


## v1.3.2
Nella tabella **DA GUARDARE OGGI** sono state aggiunte due colonne informative, senza modificare il segnale:
- **P/L da Entry %**: rendimento dall'OPEN della prima seduta successiva al filing SEC all'ultimo close disponibile.
- **Vs SPY %**: excess return rispetto a SPY sullo stesso intervallo.
Per i segnali 0/5 le colonne restano vuote finché non esiste una seduta di ingresso.
