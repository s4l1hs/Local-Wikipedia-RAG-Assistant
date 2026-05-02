# Retrieval Evaluation Report

Generated: 2026-05-02 12:45 UTC  |  Top-K: 5  |  Threshold: 0.55  |  Indexed entities: 44

## Summary

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Routing Accuracy | 100.0% (20 queries) | ≥80% | ✅ PASS |
| Hit@5 | 100.0% (15 in-corpus) | ≥80% | ✅ PASS |
| MRR | 0.967 (15 in-corpus) | — | ℹ️ info |
| OOC Rejection Rate | 100.0% (5 queries) | ≥60% | ✅ PASS |

**Indexed entities**: abraham_lincoln, acropolis_of_athens, ada_lovelace, albert_einstein, amazon_rainforest, angkor_wat, aristotle, cleopatra, colosseum, cristiano_ronaldo, eiffel_tower, florence_nightingale, frida_kahlo, grand_canyon, great_wall_of_china, hagia_sophia, isaac_newton, istanbul, leonardo_da_vinci, lincoln_memorial, lionel_messi, machu_picchu, mahatma_gandhi, marie_curie, martin_luther_king_jr, mount_everest, napoleon_bonaparte, nelson_mandela, nikola_tesla, pablo_picasso, petra, pompeii, pyramids_of_giza, statue_of_liberty, stephen_hawking, stonehenge, taj_mahal, taylor_swift, tokyo, venice, victoria_falls, william_shakespeare, wolfgang_amadeus_mozart, yellowstone_national_park

## Per-Query Results

| # | Query | Exp. Routing | Got | R✓ | Expected Entity | Hit@K | RR | Reject |
|---|-------|-------------|-----|-----|-----------------|-------|-----|--------|
| 1 | Who discovered the theory of relativity? | person | person | ✅ | albert_einstein | ✅ | 1.000 | — |
| 2 | What did Albert Einstein study? | person | person | ✅ | albert_einstein | ✅ | 1.000 | — |
| 3 | Who published the special theory of relativity in 1905? | person | person | ✅ | albert_einstein | ✅ | 1.000 | — |
| 4 | Which physicist won the Nobel Prize for the photoelectric effect? | person | person | ✅ | albert_einstein | ✅ | 1.000 | — |
| 5 | Who discovered polonium and radium? | person | person | ✅ | marie_curie | ✅ | 1.000 | — |
| 6 | Which scientist won two Nobel Prizes in different fields? | person | person | ✅ | marie_curie | ✅ | 0.500 | — |
| 7 | Tell me about Marie Curie's research on radioactivity | person | person | ✅ | marie_curie | ✅ | 1.000 | — |
| 8 | Who invented the alternating current electrical system? | person | person | ✅ | nikola_tesla | ✅ | 1.000 | — |
| 9 | What patents did Nikola Tesla hold? | person | person | ✅ | nikola_tesla | ✅ | 1.000 | — |
| 10 | Which inventor worked on wireless power transmission? | person | person | ✅ | nikola_tesla | ✅ | 1.000 | — |
| 11 | How tall is the Eiffel Tower? | place | place | ✅ | eiffel_tower | ✅ | 1.000 | — |
| 12 | When was the iron lattice tower in Paris built? | place | place | ✅ | eiffel_tower | ✅ | 1.000 | — |
| 13 | What landmark was built for the 1889 World's Fair? | place | place | ✅ | eiffel_tower | ✅ | 1.000 | — |
| 14 | Compare Einstein and the Eiffel Tower | both | both | ✅ | albert_einstein, eiffel_tower | ✅ | 1.000 | — |
| 15 | What did Tesla invent near the Eiffel Tower era? | both | both | ✅ | nikola_tesla, eiffel_tower | ✅ | 1.000 | — |
| 16 | What is the best recipe for chocolate cake? | unknown | unknown | ✅ | (ooc) | — | — | ✅ |
| 17 | How do I bake sourdough bread? | unknown | unknown | ✅ | (ooc) | — | — | ✅ |
| 18 | How do I train a puppy not to bite? | unknown | unknown | ✅ | (ooc) | — | — | ✅ |
| 19 | What are the rules of poker? | unknown | unknown | ✅ | (ooc) | — | — | ✅ |
| 20 | How do I change a car tire step by step? | unknown | unknown | ✅ | (ooc) | — | — | ✅ |
## Interpretation

- **Routing Accuracy ≥ 80%**: Router correctly classifies query intent.
- **Hit@5 ≥ 80%**: At least one expected entity appears in top-5 results.
- **MRR**: Mean Reciprocal Rank — higher is better (1.0 = always top-1).
- **OOC Rejection ≥ 80%**: Out-of-corpus queries correctly rejected (is_empty=True).

If any threshold fails, investigate in this order:
1. **Routing failures** → review `_PERSON_KW` / `_PLACE_KW` in `src/router.py`
2. **Hit@K failures** → check chunk quality in `data/processed/_chunking_report.md`
3. **OOC not rejected** → lower `similarity_threshold_low` in `src/config.py`
   or test with `--threshold 0.35` / `--threshold 0.40`
