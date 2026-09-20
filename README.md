# Independent Insider Radar v0.5

Prototipo indipendente basato su SEC Form 4 e prezzi Yahoo Finance.

## Correzione critica rispetto a v0.4
La v0.4 sceglieva la prima data Yahoo successiva al filing SEC senza imporre una distanza massima. Per ticker delistati, riutilizzati o con storico Yahoo incompleto, un segnale del 2023 poteva essere associato alla prima quotazione disponibile del 2026, creando rendimenti artificiali enormi.

La v0.5 impone quindi:
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
