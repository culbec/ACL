# Seminar 4 — Plan

Goal: a small IR system over the CACM collection. It has an inverted index, a stop list and one-word query retrieval. The notebook reports build/answer times, corpus/index statistics and 3 test queries with P@3 / P@7 for the Teams docx.

## 1. Data

### What CACM is

CACM is a classic IR test collection. It holds the titles and abstracts of articles published in *Communications of the ACM* from 1958 to 1979. The topics are computer science of that period: programming languages, compilers, algorithms, numerical methods, operating systems and data structures. The text is short. Many records are a title only.

- Location: `data/cacm/CACM-0001.html` … `CACM-3204.html`. The archive is `data/cacm.tar.gz`.
- 3,204 documents, 2,217,926 bytes in total.
- The search-engines-book site also ships 64 queries and their relevance judgments (`cacm.query`, `cacm.rel`). They are **not** in this folder and are not used (see §8).

### Document layout

Each file is `<html><pre> … </pre></html>`. Inside are blocks separated by blank lines, in this order:

| # | Block | Present in | Example (CACM-2685) |
|---|-------|-----------|---------------------|
| 1 | title | all | `The Parallel Execution of DO Loops` |
| 2 | abstract | ~49% (1,584 docs) | `Methods are developed for the parallel execution …` |
| 3 | date line | all (6 docs without the `CACM` prefix) | `CACM February, 1974` |
| 4 | authors | most | `Lamport, L.` |
| 5 | keywords | ~44% | `parallel computing, multiprocessor computers, …` |
| 6 | CR category numbers | most | `4.12 5.24` |
| 7 | record line | all | `CA740204 JB January 18, 1978  1:14 PM` |
| 8 | citation block | most | lines of 3 tab-separated integers: `2685	5	2685` |

- Doc id = file stem (`CACM-2685`).
- Length after tokenization (`[a-z]+`, before stop words): min 5, median 32, max 415 tokens per document.
- Most frequent raw terms: `the 10,985`, `of 9,168`, `a 6,478`, `and 4,595`, `to 3,782`, `is 3,711`, `in 3,465`, `cacm 3,199`, `for 3,165`, `are 1,984`, `algorithm 1,619`. `cacm` comes from the date line, not from the text.

## 2. Repo setup

- `.gitignore` ignores `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` and `data/cacm/`. `data/cacm.tar.gz` stays tracked.
- `.python-version` contains `3.14.10`.
- Dependencies: none beyond the stdlib. `ipykernel` is already present. Commands for the user:
  ```
  uv sync
  uv run python src/ir.py          # self-check
  ```
- `main.py` is not used.

## 3. Files

| Path | Role |
|------|------|
| `src/ir.py` | all logic: load, preprocess, index, search, stats, precision. Stdlib only. `__main__` = assert self-check |
| `ir.ipynb` | demo notebook. Imports `src/ir.py`, prints docx values |
| `docs/plan.md` | this file. The Results section is filled after the run |

## 4. Preprocessing

Per document:
1. Read the file and strip `<html>`, `<pre>` and their closing tags.
2. Drop bookkeeping lines. These are the citation lines (regex `^\d+\t\d+\t\d+$`), the record line (`^CA\d{6}`) and the date line (`^(CACM )?\w+,? \d{4}$`). The date line puts `cacm` and a month name into every document. It is metadata, not text.
3. Lowercase.
4. Tokenize with `[a-z]+`. This drops numbers (CR categories, years) and punctuation.
5. Remove stop words.

- **Tokens before preprocessing**: count of `\S+` over the tag-stripped raw file text, all blocks included.
- **Tokens after preprocessing**: tokens left after step 5.

Title and keyword lines are also kept separately per doc (`fields[doc_id] = set of terms`) for the relevance rule in §8.

## 5. Stop list

**Choice:** the 7 most frequent terms after steps 1–4, computed in code with `top_terms(docs, 7)`. This is the instructor's suggestion. Expected list: `the, of, a, and, to, is, in`. The notebook prints the real list, and that list goes in the docx.

**Why it is a sound choice here:**
- All 7 are pure function words. They carry no topic.
- Together they are ~21% of all tokens (42,184 of ~201,500), so removing them shrinks the index noticeably.
- The list comes from the corpus itself, so it needs no external file.
- Raising the cutoff hurts. Ranks 8–15 already include `algorithm` and `system`, which are content words a user could search for. The sample docx output stop-listed `system` and got wrong answers for that query.

**Ceiling:**
- Function words below rank 7 stay in the index: `for, are, an, on, by, with, …`. For one-word queries this costs index space, not ranking quality. IDF already gives these words a near-zero weight.
- Upgrade path: a standard list (NLTK English, 179 words) behind the same `stop_words` set. Not done, because it adds a dependency and the instructor's suggestion is the top 7.

## 6. Inverted index

The index stores **raw counts**. TF-IDF is not stored; it is computed at query time from the stored counts.

- `index: dict[str, dict[str, int]]` maps term → {doc_id: tf}, where tf is how many times the term occurs in that doc.
- `doc_len: dict[str, int]` maps doc_id → token count after preprocessing.
- df is not stored separately. It is `len(index[term])`.
- Storing counts keeps the index simple and lets the scoring change without a rebuild.
- Build = load + preprocess + index, timed end to end.
- **Index size**: `len(pickle.dumps(index))` bytes, compared with the corpus size (sum of file sizes). Report as `X / Y bytes (Z%)`.
- **Vocabulary size**: `len(index)`.

## 7. Retrieval and ranking

`search(word)` works as follows:
1. Preprocess the word the same way as the documents (lowercase, `[a-z]+`).
2. A stop word or unknown term returns an empty result.
3. Score each document in `index[word]` with TF-IDF:
   ```
   score(d) = tf(t, d) / doc_len(d) * log(N / df(t))
   ```
   - `tf / doc_len`: how much of document d is about t. Dividing by length keeps long abstracts from beating short titles just by being long.
   - `log(N / df)`: how rare t is in the collection. N = 3,204.
4. Sort by score descending. Ties are broken by doc id.
5. Return `[(doc_id, score), …]` with every match.

For a one-word query, `log(N / df)` is the same for every match. The ranking is therefore effectively tf / doc_len. The IDF factor only matters if scores are compared across queries or multi-word queries are added. It stays in because it is one multiplication and makes the score standard TF-IDF.

## 8. Evaluation

**Metric:** P@k = (relevant docs in the top k) / k, for k = 3 and k = 7.
- Divide by k even when there are fewer than k results, as the sample does (`2 matches; P@3 = 0.67`).
- 0 matches gives `top 3: none; P@3 = 0.00; P@7 = 0.00`.

**Relevance rule:** a document is relevant to word t if t occurs in its **title or keyword** lines. These are the user-facing fields, so a user who searched for t can see right away why the document came back. A document that has t only in its abstract matched, but it is not counted as relevant.

**Alternatives considered:**
- *Official CACM qrels* (`cacm.rel`). These are human judgments, but they cover 64 full natural-language queries, not single words, so they do not fit one-word retrieval. Rejected.
- *Any match counts as relevant*. Then P@k is always 1.00 whenever there are k matches, so the metric says nothing. Rejected.
- *Manual judgment of the top 7 per query*. This is the most faithful option, but it is subjective, cannot be repeated by colleagues, and must be redone whenever the query words change. Kept as an option: if the instructor asks for it, override the automatic rule in the notebook with a hand-written `{query: {doc_ids}}` dict.

## 9. Timing

- Index build time: `time.perf_counter()` around the full build. Report the median of 5 runs, in seconds.
- Answer time per query: `timeit` over 1,000 calls per word. Report the mean in ms.

## 10. Test data

The query words are **configurable**, so you can pick words that no colleague has used. Set them in the first notebook cell:

```python
QUERIES = ["compiler", "sorting", "blockchain"]
```

Every later cell (results, P@k, timing, docx lines) reads `QUERIES`. Nothing else needs editing.

The defaults cover three cases and do not repeat the sample words (`computer, space, recipes, system`):

| Case | Query | Expected |
|------|-------|----------|
| frequent term | `compiler` | many matches |
| medium term | `sorting` | tens of matches |
| absent term | `blockchain` | 0 matches |

The self-check in `src/ir.py` uses its own toy corpus and does not depend on `QUERIES`.

The notebook prints one docx line per query:
`compiler — N matching documents; top 3: CACM-…, CACM-…, CACM-…; P@3 = …; P@7 = …`

## 11. Notebook layout (`ir.ipynb`)

1. Config: `QUERIES`, data path. Imports, with `sys.path` pointing to `src`.
2. Build the index (timed) and print the build time.
3. Stop words: source and list.
4. Stats: tokens before/after, vocabulary size, index vs corpus size.
5. Test queries: number of matches, top 3, P@3, P@7.
6. Answer time per query.
7. Docx block: every value in the sample format, ready to paste.
8. Interactive cell: `search("…")` for the live demo with a word chosen in class.

Markdown cells: section headings only.

## 12. Self-check (`python src/ir.py`)

Use a tiny in-memory corpus of 3 docs. Assert:
- stop words are not in the index
- citation, record and date lines are not indexed
- tf counts are correct
- search returns the doc with the higher tf/len first
- an unknown word or a stop word returns `[]`
- P@k divides by k when there are fewer than k results
- the relevance rule uses only title and keyword terms

## 13. Results

Run on 2026-11-20. Timings vary between machines and runs.

```txt
Index build time: 0.319670 s
System answer time per query:
compiler: 0.0162 ms
sorting: 0.0093 ms
blockchain: 0.0002 ms
Stop words: CACM collection; 7 words: the, of, a, and, to, is, in.
Preprocessing: HTML tags, citation block, record line and date line removed; lowercase; [a-z]+ tokens; stop words removed.
Index size vs corpus size: 957,888 / 2,217,926 bytes (43.19%).
Vocabulary size: 11,078 distinct terms.
Tokens after preprocessing: 159,105.
Tokens before preprocessing: 374,478.
compiler — 103 matching documents; top 3: CACM-0061, CACM-0404, CACM-1149; P@3 = 1.00; P@7 = 1.00.
sorting — 61 matching documents; top 3: CACM-2191, CACM-0563, CACM-0648; P@3 = 1.00; P@7 = 1.00.
blockchain — 0 matching documents; top 3: none; P@3 = 0.00; P@7 = 0.00.
```

Notes:
- "Tokens before" counts `\S+` over all blocks, citation numbers included. That is why it is much larger than the `[a-z]+` count in §1.
- Keyword lines are detected as the block right before the CR-number block. When that block is the only one after the date line, it counts as keywords only if it does not look like an author list (an author line has exactly one comma).
