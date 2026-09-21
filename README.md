# Independent Insider Radar — v0.13

La v0.13 aggiunge un **Robustness Audit** alla replica storica 2006–2021. Le regole A/B/C restano congelate: questa versione non ricerca nuove soglie e non ottimizza il segnale.

## Regole congelate

- A — FIRST_CLUSTER ≥2 insider
- B — FIRST_CLUSTER ≥3 insider
- C — FIRST_CLUSTER ≥3 insider + controvalore aggregato $100k–250k
- finestra SEC originale ±10 giorni
- nuovo episodio dopo 10 giorni
- componente minimo $10.000
- Officer/Director
- esclusione 10b5-1 quando marcato
- ingresso OPEN prima seduta successiva al filing
- guardia entry entro 7 giorni
- benchmark SPY
- orizzonti 1 e 5 sedute

## Nuovo: Robustness Audit v0.13

Può usare direttamente `insider_event_study_2006_2021.csv`, senza rifare H1/H2/H3.

Controlli inclusi:

1. **Copertura Yahoo e missing-data bias** per anno e configurazione.
2. **Break-even dei dati mancanti**: rendimento medio ipotetico degli eventi non prezzati necessario ad azzerare la media osservata del campione selezionato.
3. **Stabilità annuale** 2006–2021 con flag automatico per micro-campioni (<30 eventi prezzati o <20 issuer).
4. **Stress years predefiniti**: 2008 e 2020, più campione esclusi 2008/2020.
5. **Outlier audit**: media, trimmed 1%, winsorized 1%, mediana, p01/p99, estremi, media senza top 1% winner e quota dei profitti positivi attribuibile al top 1%.
6. **C vs B-restante**: confronto corretto tra gruppi mutuamente esclusivi, con bootstrap issuer-level. C non viene confrontata contro B completo perché C è un sottoinsieme di B.

Output: `insider_robustness_audit_v0_13.csv`.

## Workflow consigliato

Se hai già completato la v0.12.1:

```text
Carica insider_event_study_2006_2021.csv
→ Usa Event Study per Robustness Audit
→ leggi le sezioni 1–5
→ scarica insider_robustness_audit_v0_13.csv
```

Non è necessario ripetere Yahoo.

## Nota metodologica

La v0.13 è un audit, non una nuova fase di ottimizzazione. Se una regola congelata fallisce un controllo, il risultato va registrato come limite; non si deve scegliere una nuova soglia osservando il 2006–2021.
