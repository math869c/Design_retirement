# Design_retirement

Master thesis

Plan for updating:

Done:

- Model shocks either as discrete or continous (also in simulation)
- Send første udkast til model
- Har tænkt over DC-EGM (Thomas siger det er den vej)
- Send første udkast til model
- 3d-interp, 2d-interp and 1d-interp
- Data for:
  1. Timelønninger P done
  2. Dødssandsynligheder M done
  3. Almindelig formue M/P
  4. Pensionsformue M DONE
  5. Pensionsindbetalinger done
  6. Folkepensionssats P
  7. Kontanthjælp P
  8. Arbejdstimer om året P done
- Numba njit
- livrente og ratepension
- njit simulation
- tilføjet tau
- means-testing
- første kalibrering
- Vi kører kvinder og mænd

Immediate to do:
- Clean up code
- Opdatér datadefinitioner vi vil køre og skriv ind
- Opdatér model-matematikken
- Opdatér den måde, vi løser modellen på (numerisk optimering, parallelisering, pre-computed shocks, gauss-hermite, interpolering, analytisk sidste løsning)
- Kalibrér renterne (proxy)
- Find en god forklaring på par.m
- Definer ét eller to velfærdsmål (indkomstækvivalent, forbrugsækvivalens, finanseffekt, replacement rate)


Maybe to do:
- Rente på pensionsformuen
- EGM?
- Effekstudie
- Aldersopsparing






First steps:

- Tænk mere over DC-EGM (timer er diskrete?)
- Set reasonable parameters (kig i DST)

Next week:

- Modeuludvidelser (andre tilbagetrækningsformer, andre formuer, present bias, endogene overlevelsessandsynligheder)
- Overvej andre counterfactuals

Next next week:

- Skriv paper
- Heterogenitet?
- Overveje mere, hvad præcis vi vil svare på
- Modeller samfundsnytte, optimale dækningsgrader osv...
