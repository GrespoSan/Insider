# Independent Insider Radar — v0.12.1

Prototipo indipendente basato su SEC Form 4 e prezzi Yahoo Finance. La v0.12.1 aggiunge la **Replica storica indipendente 2006–2021** con regole congelate prima di guardare quel periodo.

## Regole congelate della replica

Fonte e pulizia:
- SEC Form 4 originali;
- transazioni non-derivate codice `P`, `A` (acquired);
- common/ordinary shares;
- prezzo e quantità positivi;
- solo Officer/Director;
- filing marcati 10b5-1 esclusi quando disponibili;
- valore minimo del singolo componente: **$10.000**;
- filing con più reporting owner esclusi per evitare attribuzioni ambigue nel dataset piatto SEC;
- finestra originale delle transazioni: **±10 giorni**;
- nuovo episodio FIRST_CLUSTER dopo **10 giorni**.

Configurazioni pre-dichiarate:
- **A** — FIRST_CLUSTER con almeno 2 insider;
- **B** — FIRST_CLUSTER con almeno 3 insider;
- **C** — FIRST_CLUSTER con almeno 3 insider e controvalore aggregato **$100k–250k**.

Backtest:
- ingresso: OPEN della prima seduta successiva al filing SEC;
- guardia ticker/storico: entry entro 7 giorni di calendario dal filing;
- benchmark: SPY;
- orizzonti congelati: **1 e 5 sedute**;
- confronto robusto tramite bootstrap a livello issuer.

## Workflow v0.12.1

1. Avvia l'app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

2. Nella sezione **Replica storica indipendente 2006–2021**:
   - `H1` scarica/aggiorna i 64 trimestri SEC 2006Q1–2021Q4;
   - `H2` costruisce i componenti e i segnali storici con le regole congelate;
   - `H3` esegue il backtest Yahoo 1/5 sedute con **checkpoint persistente per batch**;
   - `H4` carica il checkpoint completato e applica le tre configurazioni senza retuning.

## Checkpoint

La v0.12.1 salva automaticamente ogni batch Yahoo in:

`data/historical_replication_v0_12/insider_event_study_2006_2021.csv`

Se Streamlit si riavvia durante il test, `H3` riprende dagli eventi non ancora elaborati. Una firma del set di segnali impedisce di riutilizzare per errore un checkpoint appartenente a dati/regole differenti.

Anche i componenti SEC vengono salvati trimestre per trimestre in modo che `H2` possa riutilizzare quelli già elaborati.

## Output diagnostici

La replica mostra:
- copertura Yahoo per anno;
- numerosità e copertura per configurazione;
- media, trimmed mean 1%, mediana e win rate;
- bootstrap issuer-level configurazione − SOLO;
- diagnostica temporale fissa 2006–2010 / 2011–2015 / 2016–2021;
- criteri pre-dichiarati di direzione e robustezza.

## Limite importante

Yahoo può non conservare prezzi storici per molti ticker delistati o riutilizzati. La copertura prezzi è quindi parte integrante dell'interpretazione: una replica positiva con copertura bassa non costituisce da sola una validazione definitiva.

## Test

```bash
pytest -q
```

La v0.12.1 include test per causalità dei cluster, ticker mancanti, timezone, guardia anti-storico incoerente, caricamento CSV e ripresa/checkpoint del backtest storico.


## Fix v0.12.1
H3 non dipende più dalla presenza di `backtest_signals_checkpointed` nel modulo `engine.py`: `app.py` contiene un fallback locale deployment-safe che usa gli helper Yahoo già presenti nelle versioni v0.8–v0.11. Questo evita il falso blocco “engine.py non è aggiornato” dovuto a deploy/cache disallineati su Streamlit Cloud.
