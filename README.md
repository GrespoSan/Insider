# Independent Insider Radar v0.14 — Value Normalization Audit

Questa versione **non modifica il CORE signal** e non cerca una nuova soglia ottimale.
Aggiunge un controllo economico sulla fascia di controvalore `$100k–250k`.

## Regole congelate

- **B / CORE**: `FIRST_CLUSTER >= 3 insider`
- **C-NOMINAL**: B + controvalore nominale del cluster tra `$100k` e `$250k`
- **C-REAL**: B + controvalore del cluster equivalente a `$100k–250k` in **dollari 2026**
- Orizzonti analizzati: **1 e 5 sedute**
- Nuovo episodio: **10 giorni**

## Correzione per inflazione

Usiamo CPI-U, U.S. city average, All items, non destagionalizzato.

- 2006–2025: **media annuale CPI-U**
- riferimento 2026: **indice CPI-U agosto 2026 = 334.980**, ultimo dato disponibile quando è stata costruita la v0.14

Formula:

`cluster_value_2026 = cluster_value_nominale * CPI_2026 / CPI_anno`

La fascia C-REAL include i cluster con `cluster_value_2026 >= 100000` e `< 250000`.
Non viene cercata nessuna fascia alternativa.

Fonti BLS utilizzate per la tabella statica CPI:
- Historical CPI-U tables, U.S. Bureau of Labor Statistics
- CPI-U August 2026 release, U.S. Bureau of Labor Statistics

## Come usarla

Per il controllo completo caricare nella sezione **Value Normalization Audit — v0.14**:

1. `insider_event_study_2006_2021.csv`
2. `insider_event_study_first_cluster.csv` oppure l'Event Study completo 2022–2026

L'app mostra:

- soglie nominali equivalenti anno per anno;
- B vs C-NOMINAL vs C-REAL;
- overlap tra C-NOMINAL e C-REAL;
- bootstrap issuer-level di ciascuna fascia contro il resto di B;
- coerenza di media, trimmed mean, mediana e win rate;
- export `insider_value_normalization_audit_v0_14.csv`.

## Interpretazione corretta

C-REAL serve a verificare se il controvalore aggiunge informazione economica dopo aver tolto l'effetto dell'inflazione. Se non mostra un vantaggio stabile rispetto al CORE B, il filtro monetario non va reso obbligatorio nel radar operativo.
