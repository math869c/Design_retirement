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
- Opdatér model-matematikken
- Clean up code
- indfør extensive margin
- Definer ét eller to velfærdsmål (indkomstækvivalent, forbrugsækvivalens, finanseffekt, replacement rate)
- Clean up code
- Tilføjet udbetaling som choice
- Probabilities for fyret/hyret
- Initial draws
- Overvej andre counterfactuals (udbetalingsalder, pensionsalder, indbetalingssatser)
- Skrevet motivation og regler om pensionssystemet (skal self skrives rent)
- Ændre den forventede levetid som funktion af pensionstidspunktet
- Effekt på intensiv og ekstensiv margin

Immediate to do:
- Opdatér datadefinitioner vi vil køre og skriv ind
- Opdatér den måde, vi løser modellen på (numerisk optimering, parallelisering, pre-computed shocks, gauss-hermite, interpolering, analytisk sidste løsning)
- Kalibrér renterne (proxy)
- Find en god forklaring på par.m
- Rente på pensionsformuen
- Ventetillæg (kræver choice ti)
- Skriv modellen op igen, gennemgå matematik, lave variabeltabel
- Overvej hvordan vi sætter unemployment benefits og folkepension

Maybe to do:
- Effekstudie
- Modeuludvidelser (andre tilbagetrækningsformer, andre formuer, present bias, endogene overlevelsessandsynligheder)
- Aldersopsparing

First steps:
- Set reasonable parameters (kig i DST)



Next next week:

- Skriv paper
- Heterogenitet?
- Overveje mere, hvad præcis vi vil svare på
- Modeller samfundsnytte, optimale dækningsgrader osv...
