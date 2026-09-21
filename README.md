# Independent Insider Radar LIVE — v1.1

Versione operativa separata dalla fase di ricerca/backtest. La v0.14.1 resta l'archivio della ricerca; questa app monitora i nuovi Form 4 SEC con regole congelate.

## Regole congelate

- Solo Form 4 originali.
- Solo transazioni non-derivative con codice `P` e acquisizione `A`.
- Common/ordinary shares; quantità e prezzo positivi.
- Componente minimo **$10.000**.
- Solo Officer/Director.
- Filing con più reporting owner esclusi per prudenza attributiva.
- 10b5-1 escluso quando esplicitamente marcato.
- Finestra cluster **±10 giorni** sulle transaction date, senza usare filing futuri.
- `WATCH`: primo cluster che raggiunge **≥2 insider**.
- `CORE`: primo cluster che raggiunge **≥3 insider**.
- `VALUE 100–250k (2026$)`: tag informativo, non filtro obbligatorio.
- Orizzonte empirico principale osservato: **5 sedute**.

## Novità v1.1

La dashboard operativa è divisa in stati meccanici, non in nuovi segnali:

- **NUOVI**: il filing è pubblico ma non esiste ancora una seduta successiva utile per l'entry OPEN.
- **ATTIVI 1–4/5**: entry disponibile e finestra empirica ancora in corso.
- **COMPLETATI 5/5**: almeno 5 sedute osservate, con `Return 5D` e `Excess 5D` quando disponibili.
- **DA VERIFICARE**: Yahoo non risolve correttamente ticker/prezzo/storico.
- **DA PREZZARE**: il radar è stato costruito ma lo snapshot Yahoo non è ancora stato aggiornato.

Per ogni segnale vengono mostrati inoltre:

- numero insider e banda `3 / 4 / 5+`;
- tag `VALUE`;
- presenza informativa di `CEO`, `CFO` o `CEO+CFO` (nessuno di questi modifica CORE/WATCH);
- progresso `0/5 ... 5/5`;
- performance da entry e vs SPY;
- link SEC e TradingView.

## Export distinti

- **Radar completo CSV**: tutti i CORE + WATCH presenti nello stato live, incluso il margine iniziale con contesto cluster incompleto.
- **Radar operativo CSV**: CORE + WATCH con `context_complete = True`.
- **Vista filtrata CSV**: esattamente ciò che è selezionato nell'interfaccia.

Questa distinzione evita di confondere l'universo completo con la vista operativa filtrata.

## Uso

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Inserisci una email nella sidebar per il User-Agent SEC.
2. Lascia 21 giorni come lookback iniziale.
3. Premi **1. Sync / continua SEC Live** finché `Pendenti = 0`.
4. Premi **2. Costruisci / aggiorna Radar**.
5. Premi **3. Aggiorna prezzi / performance**.
6. Usa le sezioni NUOVI / ATTIVI / COMPLETATI e salva periodicamente il backup ZIP.

## TradingView

L'app contiene un link TradingView per il ticker. Il plugin TradingView disponibile dentro ChatGPT è separato da Streamlit e può essere usato in chat per una verifica qualitativa dei pochi CORE emersi dal radar.

## Limiti

- Il primo sync può richiedere più esecuzioni.
- Yahoo può non avere prezzi per ticker delistati/rinominati.
- `CORE`, `WATCH` e `VALUE` sono etichette di ricerca/monitoraggio, non raccomandazioni d'investimento.
