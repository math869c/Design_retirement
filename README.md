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

Immediate to do:
- Clean up code
- Opdatér datadefinitioner vi vil køre og skriv ind
- Opdatér den måde, vi løser modellen på (numerisk optimering, parallelisering, pre-computed shocks, gauss-hermite, interpolering, analytisk sidste løsning)
- Kalibrér renterne (proxy)
- Find en god forklaring på par.m
- Definer ét eller to velfærdsmål (indkomstækvivalent, forbrugsækvivalens, finanseffekt, replacement rate)
- indfør extensive margin


Maybe to do:
- Rente på pensionsformuen
- EGM?
- Effekstudie
- Aldersopsparing


Plan for at indfære extensive margin:
del 1:
- tilføj ekstra chocie-variable: binary om at arbejd eller ikke at arbejde.
- ændre på hours, så den ikke længere går fra 0. bound for h: [7/37,1.2]
- lav to value functions: 
  1. klassisk med valg af timer
  2. kun vlag af forbrug
- vælg bedste value funktion og tilsvarende valg 

del 2:
- tilføj state om man har arbejde, hvor der er ssh for at blive fyret, og tilføjet state om ssh for at få job. Denne ssh afhænger også af dit valg og noget eksternt
- Ssh er eksogene og afhænger kun af din alder. Derfor modellerer vi ikke arbejdsmarkedet
- 







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
