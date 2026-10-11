# Validation status — company dev candidate

- 18 offline tests passed: original 11 plus 7 WAM integration tests.
- WAM full-chunk commit, terminal partial-chunk commit, execution failure discard,
  empty output discard, wall-budget exit, config mapping and facade reset/lazy loading tested.
- Original Astra initialization, ICL loading, router guards and episode caching tested.
- Python compilation and shell syntax checks recorded in VALIDATION.json.
- No real WAM checkpoint loaded, GPU inference or native simulator run in this validation.
- Plain/encoded network debug loops not run; custom RPC transport needs live smoke testing.
- JEV trained checkpoint and public asset hosting remain pending.
- Astra custom command envelope is retained; this is an eval-only integration draft,
  not a validated leaderboard release or a performance claim.

## Balls success update (2026-10-11 v2)

Includes the own-success unified-v2 ICL (75 images), exact r23 task skill and shared main_route rotation guard from RSI-final-3 v13. 18 offline tests plus 8 guard cases pass. The source run succeeded at seed1/layout1 in 374/600 steps; this integrated package has not been run in simulation. The shared guard can affect other main_route rotation tasks; cross-task native regression remains pending. Other task skills and legacy Open implementations are unchanged. r24 remains archived outside the active release.
