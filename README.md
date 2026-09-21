# Independent Insider Radar LIVE — v1.0

Versione operativa separata dalla fase di ricerca/backtest. La v0.14.1 può essere mantenuta come archivio della ricerca; questa app serve a monitorare i nuovi Form 4 SEC.

## Regole congelate

- Solo Form 4 originali.
- Solo transazioni non-derivative con codice `P` e acquisizione `A`.
- Common/ordinary shares; quantità e prezzo positivi.
- Componente minimo: **$10.000**.
- Solo Officer/Director.
- Filing con più reporting owner esclusi per prudenza attributiva.
- 10b5-1 escluso quando esplicitamente marcato.
- Finestra cluster: **±10 giorni** sulle transaction date, senza usare filing futuri.
- `WATCH`: primo cluster che raggiunge **≥2 insider**.
- `CORE`: primo cluster che raggiunge **≥3 insider**.
- `VALUE 100–250k (2026$)`: **tag informativo**, non filtro obbligatorio.
- Orizzonte empirico principale osservato: **5 sedute**.

## Dati live SEC

La SEC pubblica il dataset Form 3/4/5 bulk trimestralmente; al 21 settembre 2026 la pagina ufficiale arriva al 2026 Q2. Per non perdere gli ultimi filing, v1.0 usa gli **indici giornalieri EDGAR** e poi scarica i singoli Form 4 completi.

Per ragioni di carico e perché il pattern operativo usa ±10 giorni e 5 sedute, il sync live è volutamente recente (14/21/30 giorni) e checkpointed. Non serve riscaricare tutto il trimestre corrente per trovare i segnali ancora operativamente rilevanti.

SEC richiede un User-Agent identificabile e limita l'accesso automatizzato a non più di 10 richieste al secondo. L'app usa richieste sequenziali moderate e salva lo stato ogni 25 filing.

Fonti ufficiali:
- https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
- https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- https://www.sec.gov/about/developer-resources

## Uso

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Inserisci una email nella sidebar per il User-Agent SEC.
2. Lascia 21 giorni come lookback iniziale.
3. Premi **1. Sync / continua SEC Live**. Se restano filing pendenti, premi di nuovo: il lavoro riprende dal checkpoint.
4. Quando il sync è completo, premi **2. Costruisci / aggiorna Radar**.
5. Premi **3. Aggiorna prezzi / performance** per aggiungere il contesto Yahoo.
6. Filtra `CORE` e `WATCH`; il link `SEC` apre il filing ufficiale e `TV` apre TradingView.

## Backup

Il pulsante **Backup stato live ZIP** salva indice, checkpoint, componenti, radar e snapshot prezzi. Dopo un redeploy Streamlit puoi caricare il file nella sidebar e premere **Ripristina stato**.

## Nota su TradingView

L'app contiene solo un link TradingView per il ticker. Il plugin TradingView disponibile dentro ChatGPT è separato da Streamlit e può essere usato qui in chat per una verifica qualitativa dei ticker emersi dal radar.

## Limiti

- Il primo sync può richiedere più esecuzioni perché ogni Form 4 deve essere letto per sapere se contiene un acquisto `P` qualificante.
- Yahoo può non avere prezzi per ticker delistati/rinominati.
- `CORE`, `WATCH` e `VALUE` sono etichette di ricerca/monitoraggio, non raccomandazioni d'investimento.
