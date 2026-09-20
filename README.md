# Independent Insider Radar v0.9

Prototipo indipendente basato su SEC Form 4 e prezzi Yahoo Finance.

## Correzione critica rispetto a v0.4
La v0.4 sceglieva la prima data Yahoo successiva al filing SEC senza imporre una distanza massima. Per ticker delistati, riutilizzati o con storico Yahoo incompleto, un segnale del 2023 poteva essere associato alla prima quotazione disponibile del 2026, creando rendimenti artificiali enormi.

La v0.9 impone quindi:
- entry = OPEN della prima seduta successiva al filing SEC;
- la seduta deve cadere entro **7 giorni di calendario** dal filing;
- altrimenti `price_status = stale_symbol_or_gap` e il record non entra nei rendimenti.

La sintesi aggiunge anche:
- numero di issuer unici;
- media semplice;
- **media tagliata 1%** (rimozione dell'1% di coda per lato, solo a fini diagnostici);
- mediana;
- win rate vs SPY;
- percentile 1 e 99.

Queste statistiche sono ancora descrittive. Il passo successivo corretto è aggiungere intervalli di confidenza/bootstrapping con dipendenza per issuer e separare il **primo trigger di cluster** dalle ripetizioni ravvicinate dello stesso episodio.


## Fix v0.9
- Normalizza date SEC/Yahoo a timezone-naive prima del confronto.
- Evita TypeError con pd.NA nei ticker.
- Gestisce celle Yahoo duplicate/non scalari in modo conservativo.
- Mantiene il filtro anti-stale a massimo 7 giorni dal filing SEC.


## Riprendi da CSV (v0.9)
Puoi caricare `insider_signals_all.csv` dalla sidebar e premere **Usa CSV segnali caricato**. In questo modo salti il passo 2 e vai direttamente all'Event Study. `insider_signals_view.csv` è accettato, ma se contiene solo cluster non permette il confronto SOLO vs CLUSTER.


## v0.9
- Event study Yahoo realmente streaming: un batch viene scaricato, elaborato e liberato prima del successivo.
- Barra di avanzamento per batch ed eventi elaborati.
- Riduce drasticamente il picco RAM su Streamlit Cloud con universi >2.000 ticker.


## Novità v0.9
- classificazione SOLO / FIRST_CLUSTER / REPEAT_CLUSTER;
- bootstrap a cluster di issuer per FIRST_CLUSTER - SOLO;
- sensibilità soglia insider 2/3/4 sulla finestra SEC originaria;
- caricamento diretto di insider_event_study.csv per evitare di rieseguire Yahoo;
- distinzione esplicita tra finestra transazioni SEC e distanza tra episodi cluster.
