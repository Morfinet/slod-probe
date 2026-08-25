# AI assistance

I used ChatGPT and Claude as sounding boards while planning the experiment and working through a few implementation choices. I made the final decisions, ran the experiments locally, inspected the data and errors, and checked the reported numbers against the saved outputs.

I did not keep complete chat exports. The prompts below are therefore reconstructed from my notes and describe what I asked, but they are not verbatim transcripts.

## Experimental plan

**Tool:** ChatGPT

**Prompt (reconstructed):**

> Help me turn this assignment into an experiment I can finish in a week. What outputs are essential, and what mistakes could invalidate the result?

The response proposed a pipeline consisting of weak-label generation, frozen MiniLM embeddings, three logistic-regression conditions, evaluation metrics, and qualitative error analysis. The warning about document leakage was particularly useful. I used `paper_id`, rather than individual spans, as the train/test grouping variable. After implementation, I checked the repository against `task.md` and confirmed that every required condition and output was present.

## Weak labels from paper structure

**Tool:** Claude

**Prompt (reconstructed):**

> My S2ORC input has titles, abstracts, headings, and paragraphs on separate lines. Suggest understandable rules for macro, meso, and micro labels while excluding references and appendices.

The response suggested treating titles and abstracts as macro candidates, using the opening sentence of an ordinary section for meso examples, and drawing micro examples from later methods/results paragraphs. I adapted those rules, added minimum-length checks, and retained the metadata needed for later analysis. I manually inspected examples from each source type and rebuilt the dataset from the pinned shard; the rebuild produced the same 1,500 selected spans.

## Parser expressions

**Tool:** ChatGPT

**Prompt (reconstructed):**

> Help me make the regular expressions in `src/dataset.py` precise enough to identify NLP and CV papers, recognize common section names, skip references and appendices, and normalize section numbers such as `2.1` or `IV.`.

I used the suggestions as a starting point for `DOMAIN_PATTERNS`, `INTRO`, `CONCLUSION`, `EXCLUDED`, `DETAIL`, and `SECTION_NUMBER`. I kept domain assignment only when exactly one domain pattern matched and included spelling variants such as `summari[sz]ation`. I then read samples of the matched titles and headings, checked that excluded sections produced no spans, and confirmed that all six domain/class cells still contained 250 examples.

## Reproducible balancing

**Tool:** ChatGPT

**Prompt (reconstructed):**

> I need 500 spans per class across NLP and CV, but long papers generate many more candidates. What simple deterministic sampling scheme will keep a few papers from dominating?

The proposed approach was to shuffle paper IDs with a fixed seed, cap the number of spans contributed by one paper, and sample round-robin across papers. I used seed 42 and a limit of four spans per paper and label. The resulting dataset contains 500 macro, 500 meso, and 500 micro spans from 590 papers. I also checked that every row contains the required metadata.

## Length control

**Tool:** Claude

**Prompt (reconstructed):**

> The suggested 100–150-token range removes nearly all titles and many section-opening sentences. How can I control length without losing whole label categories?

The response suggested choosing a token count supported by all classes, truncating retained spans to that exact length, and embedding them again. I used 24 model tokens and rebalanced the classes within the fixed paper split. This is not a perfect control because truncation removes later context, so I state that limitation in the report. I verified that all 1,223 controlled records have `model_token_count = 24`; the final controlled evaluation contains 423 training examples and 102 test examples, with 34 test examples per class
