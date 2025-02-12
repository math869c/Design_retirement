# Design_retirement

Master thesis

Plan for updating:

Done:

* Model shocks either as discrete or continous (also in simulation)
* Send første udkast til model
* Har tænkt over DC-EGM (Thomas siger det er den vej)
* Clean up code
* Send første udkast til model

First steps:

- Numba njit
- Tænk mere over DC-EGM (timer er diskrete?)
- 3d-interp, 2d-interp and 1d-interp
- Set reasonable parameters (kig i DST)
  1. Timelønninger P
  2. Dødssandsynligheder M
  3. Almindelig formue M/P
  4. Pensionsformue M
  5. Pensionsindbetalinger (Har vi)
  6. Folkepensionssats P
  7. Kontanthjælp P
  8. Arbejdstimer om året P

Next week:

- Modeuludvidelser (andre tilbagetrækningsformer, andre formuer, present bias, andre udbetalingsformer som rate, livrente, means-testing, endogene overlevelsessandsynligheder)
- Mere empiri: Formue (både pension, formue og bolig), lønindkomst, arbejdstid, forbrug

Next next week:

- Tænk over estimering/kalibering

Next next next week:

- Skriv paper
- Heterogenitet?
- Overveje mere, hvad præcis vi vil svare på
- Modeller samfundsnytte, optimale dækningsgrader osv...
