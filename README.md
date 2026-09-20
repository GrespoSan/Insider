# Independent Insider Radar v0.3

Prototipo indipendente per verificare l'idea degli acquisti insider senza dati a pagamento.

## Cosa fa

1. Scarica i dataset trimestrali ufficiali SEC Forms 3/4/5.
2. Tiene solo Form 4 originali (non 4/A).
3. Tiene solo transazioni non derivate con:
   - `TRANS_CODE = P`
   - `TRANS_ACQUIRED_DISP_CD = A`
   - common/ordinary shares
   - quantità e prezzo positivi.
4. Per prudenza elimina i filing con più reporting owner, perché nel dataset SEC appiattito le righe transazione non hanno una foreign key verso uno specifico owner.
5. Opzionalmente:
   - limita a Officer/Director;
   - elimina i filing marcati `AFF10B5ONE`.
6. Costruisce un'etichetta **cluster** usando solo informazioni già pubbliche alla filing date.
7. Può eseguire un event study gratuito con `yfinance`:
   - ingresso all'OPEN della prima seduta successiva al filing;
   - orizzonti 1 / 5 / 21 / 63 sedute;
   - rendimento in eccesso rispetto a SPY.

## Installazione

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## User-Agent SEC

L'app chiede una email di contatto e invia un User-Agent del tipo:

`IndependentInsiderRadar/0.1 nome@email.it`

Non inserire credenziali o password: serve soltanto a identificare correttamente l'accesso automatizzato al sito SEC.

## Definizione cluster v0.3

Per ogni nuovo giorno di filing SEC di una società, l'algoritmo guarda soltanto le transazioni che a quel momento risultano già pubbliche. Intorno alle date di transazione appena divulgate cerca acquisti di owner distinti entro `± window_days`. Se gli owner distinti raggiungono `min_insiders`, l'issuer-day è marcato come cluster.

Questa scelta evita il look-ahead più grave: non viene usato un filing futuro per trasformare retroattivamente un vecchio acquisto in un segnale che all'epoca il pubblico non poteva conoscere.

## Cosa NON fa ancora

- Non risolve in modo completo Form 4/A e supersessioni.
- Non interpreta le footnote.
- Non distingue automaticamente acquisti open-market da private purchase: il codice SEC `P` comprende entrambi.
- Non applica ancora filtri di liquidità, market cap, drawdown o fondamentali.
- Non fornisce ancora inferenza statistica robusta (clustered standard errors / bootstrap per issuer-tempo).
- Il dataset trimestrale SEC non copre il trimestre in corso in tempo reale; il live radar richiede un modulo EDGAR giornaliero separato.

## Perché v0.3 è volutamente conservativa

L'obiettivo non è costruire subito un “Insider Score”. Prima vogliamo rispondere a domande verificabili:

- gli acquisti P hanno davvero vantaggio dopo la pubblicazione?
- il vantaggio sopravvive usando la **filing date** e non la transaction date?
- 2 o 3 insider sono migliori di uno solo?
- l'effetto è concentrato nei primi giorni o persiste per mesi?

Solo dopo questi test ha senso aggiungere market cap, fondamentali o scoring.


## Correzione v0.3
La SEC usa due percorsi diversi per gli ZIP trimestrali Insider Transactions.
Il downloader prova automaticamente prima `structureddata` e poi `datastandardsinnovation`,
così gestisce sia i trimestri storici sia quelli recenti senza modifiche manuali.
