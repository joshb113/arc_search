# plan-003 – Precision (weeks 3–6)

**Status:** 📋 Planned
**Goal:** Make the results trustworthy. Every threshold traces to a measurement.

**Exit criterion:** thresholds in `config.py` no longer marked UNCALIBRATED, and
`calibrated: bool` flips to True from a real `eval.calibrate` run.

---

## Phase 1 – The labeled set ⬜

- [ ] Build it from FOSDEM's `alt="Photo of NAME"` — a weak label supplied by the
      publisher, which the crawler already records. **40 of 40** images in the first
      Postgres run carried one.
      ⚠️ **The label is year-bounded: 2015–2025 carry it, 2013–2014 do not.**
      Measured directly against the archive, one speaker page per year. So the
      eval set draws on 11 of 13 crawled years; the 2013–2014 photos are
      detection corpus only. A run whose labels read zero is not necessarily
      broken — check which years it has reached first.
- [ ] Populate `eval_pair` — same-person pairs are the same speaker across years and
      across conferences

## Phase 2 – Derive the thresholds ⬜

- [ ] `eval/calibrate.py` → ROC → `t_plausible`, `t_strong`, `t_near_certain`
- [ ] Replace the UNCALIBRATED block in `config.py` with the output

## Phase 3 – Ensemble and clustering ⬜

- [ ] AdaFace second model, ensemble agreement re-rank
- [ ] Face-level canonicalization at cosine > 0.92
- [ ] Identity clustering of results
- [ ] Scale to 1–10M

---

## Open questions

- **PDQ near-dup threshold (31) is conventional, not tuned for our corpus.** It is
  the accepted "same image" boundary for PDQ's 256 bits. It is also meaningless
  against the 64-bit dHash fallback, which is why `pdqhash` is now a CI dependency.
- **Crops for all faces, or only above quality p60?**
  [[ADR-001-crop-only-storage]] rejected the hybrid for now; revisit past 50M.
- **ccc-media carries no weak labels** — thumbnail alt text is the talk title, not a
  speaker name. It is a detection corpus (on-stage faces, off-axis, variably lit),
  deliberately, as a counterweight to FOSDEM's clean portraits. Do not expect eval
  pairs from it.
