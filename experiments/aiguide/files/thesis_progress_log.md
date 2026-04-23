# THESIS PROGRESS LOG
## Suhas Venkat | M.Sc. Data Science | University of Europe
## Supervisor: Prof. Dr. Raja Hashim Ali
## Topic: Intelligent Email Triage and Automated Response Generation Using RAG-Augmented LLM Agents

---

## ✅ EXPERIMENTS — COMPLETED

| Experiment | Status | Real Value | Date |
|---|---|---|---|
| Twitter dataset (1,200 emails) | ✅ Done | n=1,200 (6 intents × 200) | Apr 23 |
| FAISS KB rebuild (18 chunks) | ✅ Done | 6 domain docs created | Apr 23 |
| Fix 1: Mistral-7B pipeline (100 emails) | ✅ Done | ROUGE-L=0.104, BLEU=0.012 | Apr 23 |
| Fix 2: RAGAS | ⚠ Approx | Faithfulness=0.84 (approx) | Apr 23 |
| Fix 3: BERT fine-tuning (CPU, 3 epochs) | ✅ Done | F1=0.559 (full), F1=0.683 (400-sample) | Apr 23 |
| Fix 4: Locust load test | ✅ Done | 7800ms@10u, 99.24% error@100u | Apr 23 |
| Fix 5: McNemar's test | ✅ Done | p=0.00145 (significant) | Apr 23 |
| Fix 6: Multilingual | ❌ Skipped | No DeepL credit card | Apr 23 |
| Fix 7: Human annotation | ✅ Done | 3 annotators, mean=4.48, κ=-0.021 | Apr 23 |
| 16 figures generated | ✅ Done | experiments/results/figures/ | Apr 23 |
| 12 tables generated | ✅ Done | experiments/results/tables/ | Apr 23 |

---

## ✅ WRITING — COMPLETED

| Chapter | Status | Word Count (est.) |
|---|---|---|
| Title page | ✅ | — |
| Abstract | ✅ | ~300 |
| Chapter 1 — Introduction | ✅ | ~800 |
| Chapter 2 — Literature Review | ✅ | ~1,200 |
| Chapter 3 — Methodology | ✅ | ~1,000 |
| Chapter 4 — Implementation | ✅ | ~600 |
| Chapter 5 — Experimental Setup | ✅ | ~700 |
| Chapter 6 — Results & Discussion | ✅ (needs figures) | ~2,000 |
| Chapter 7 — Conclusion | ✅ | ~700 |
| References | ✅ | 20 citations |
| **TOTAL** | **~8,300 words** | **Target: 15,000–20,000** |

**⚠ Need ~7,000 more words! Expand each chapter — see "How to expand" below.**

---

## ❌ STILL TO DO

### Figures (insert into thesis Word doc):
- [ ] fig01 → After Section 1.3
- [ ] fig02 → Section 6.1
- [ ] fig03 → Section 6.2
- [ ] fig04 → Section 6.2
- [ ] fig05 → Section 6.5
- [ ] fig06 → Section 6.1
- [ ] fig07 → Section 6.3
- [ ] fig08 → Section 6.4
- [ ] fig09 → Section 6.4
- [ ] fig10 → Section 6.4
- [ ] fig11 → Section 6.5
- [ ] fig12 → Section 6.5
- [ ] fig13 → Section 6.6
- [ ] fig14 → Section 6.6
- [ ] fig15 → Section 6.7
- [ ] fig16 → Section 6.8

### Admin:
- [ ] Sign Declaration of Authorship (file: Declaration_of_Authorship pdf)
- [ ] Fill + sign AI Usage Declaration (file: TemplateDeclaration pdf)
- [ ] Update TOC page numbers
- [ ] Spell check
- [ ] Word count check
- [ ] GitHub repo → make public
- [ ] Thesis registration form → submit to UE admin
- [ ] Export to PDF

---

## HOW TO EXPAND WORD COUNT

Each section below needs more content:

### Chapter 2 (Literature Review) — add these subsections:
- 2.6 Customer Support Datasets — discuss Twitter, Enron, ASAP
- 2.7 Ethical Considerations in AI Customer Support
- 2.8 Related Systems (commercial chatbots: Zendesk, Intercom)

### Chapter 3 (Methodology) — add:
- 3.7 State Schema Details (AgentState TypedDict fields explained)
- 3.8 Prompt Engineering Decisions
- 3.9 Error Handling and Fallback Strategy

### Chapter 5 (Experimental Setup) — add:
- 5.5 Implementation Details (hardware spec, run time)
  - Hardware: Apple M4, 16GB unified memory
  - Mistral inference: ~17s per email
  - Total experiment run time: ~28 minutes (100 emails)

### Chapter 6 (Results) — add more analysis:
- Per-section: discuss WHY results are what they are
- Compare to literature values
- Discuss implications for deployment

---

## REAL VALUES — DO NOT CHANGE

```
Intent F1 (Proposed):     0.689
Intent F1 (Keyword):      0.565
Intent F1 (BERT):         0.683 (400-sample) / 0.559 (full)
Intent F1 (GPT-4†):       0.825 (proxy)
ROUGE-1:                  0.148
ROUGE-L:                  0.104
BLEU-4:                   0.012
Escalation Rate:          38%
Locust @10 users:         7,800ms mean, 0% errors
Locust @100 users:        16,293ms mean, 99.24% errors
Human mean overall:       4.48 / 5.0
Cohen's κ:               -0.021
McNemar p:                0.00145
BERT F1 (full dataset):   0.559
```

---

## FILE LOCATIONS

```
Project root:     /Users/suhasvenkat/Projects/SupportMailAgent/
Twitter data:     data/twitter_support/twcs.csv
Processed:        data/processed/twitter_support_processed.csv
FAISS index:      data/faiss_index/
KB documents:     knowledge_base/docs/ (6 files)
Cache:            data/processed/fix_all_cache.json
Figures:          experiments/results/figures/ (16 PNGs)
Tables:           experiments/results/tables/ (12 CSVs)
Annotations:      experiments/annotations/ (3 CSV files)
Thesis doc:       ~/Downloads/SuhasVenkat_MasterThesis_Final.docx
Experiments:      experiments/run_real_experiments_FINAL.py
                  experiments/fix_all_free.py
```

---

## KNOWN ISSUES (disclosed in thesis)

1. **GPT-4 proxy**: LogReg C=8.0 used instead of real GPT-4 API
2. **RAGAS approx**: Keyword co-occurrence, not LLM-judge
3. **Mistral latency**: 16s/email on M4 CPU — needs GPU for production
4. **BERT training**: Only 400 samples used (CPU constraint)
5. **Multilingual**: Skipped (DeepL key not obtained)
6. **Low κ**: -0.021 — annotator calibration needed
7. **Twitter data**: Tweets ≠ formal emails — generalisability limited
