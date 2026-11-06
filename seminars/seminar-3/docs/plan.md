# Seminar 3 — Plan

Goal: get the Zipf's law representation for an English corpus and a Spanish corpus from the same domain (malicious URL detection) and compare them. Then report the results in the MS Teams docx table (Files / Sem3, your semigroup's table).

The assignment's "compare it with the results obtained by using another corpus" means **comparing with your colleagues' work**: the corpora and results they report in the same docx table. It does not mean processing a third corpus yourself.

## 1. Corpora

| # | Role | Language | Title | Author / Source | Raw words* |
|---|------|----------|-------|-----------------|-----------|
| A | Main corpus | English | *NURLP: Detection of malicious URLs using NLP and Deep Learning techniques* (Diploma thesis, 2025) | Profir Alexandru. Babeș-Bolyai University Cluj-Napoca, Faculty of Mathematics and Computer Science. Supervisor: Assist. PhD. Briciu Anamaria | 22,866 |
| B | Other language, same domain | Spanish | *Detección de URLs maliciosas por medio de técnicas de aprendizaje automático* (Master's thesis, 2021) | María Martha Céspedes Maestre. Universidad Nacional de Colombia, Facultad de Ingeniería, Bogotá. Director: Ph.D. Octavio José Salcedo Parra. Codirector: MSc. Carlos Alberto Salazar Herrera. Institutional repository of UNAL | 29,728 |

\* Raw count = `pdftotext file.pdf - | wc -w`. This still includes numbers, tables, references and code. The cleaned counts go in the notebook and the docx table.

Files are in `seminars/seminar-3/corpus/`:
- `NURLP-Diploma_Thesis-Profir_Alexandru.pdf`
- `1094269813.2021.pdf`

Each document is a corpus on its own; no preloaded corpora are used.

**Size note.** Corpus A is below the suggested ~50k words. The instructor approved using the thesis alone to keep it original. To keep the comparison fair, both corpora are truncated to the size of the smaller one: N = min(cleaned token count of A, cleaned token count of B), and each corpus keeps its first N tokens. Zipf curves depend on corpus size, so curves of unequal sizes cannot be compared directly. The results reported in the docx use the truncated corpora (both of size N). The notebook also computes every value on the original (full) corpora and compares the two, to show how much truncation changes the results.

## 2. Zipf's law

**Tokens vs types.** A *token* is one occurrence of a word in the text. A *type* is a distinct word form. In "the model and the data", there are 5 tokens but only 4 types, because *the* appears twice. Tokens measure corpus size. Types measure vocabulary size. The docx table asks for both, and the hapax column is measured both ways:
- A word that appears once (a *hapax legomenon*) is 1 type and 1 token.
- So the number of hapaxes is the same either way. Only the denominator changes: hapaxes / types is usually large (around 40–60%), while hapaxes / tokens is small (a few %).
- After lowercasing, *Model* and *model* count as the same type. With lemmas, *model* and *models* also merge into one type, so lemmas give fewer types.

Count how often each distinct word (each *type*) occurs in a corpus, then sort the types from most to least frequent. The most frequent word gets rank r = 1, the next r = 2, and so on. **Zipf's law** (George K. Zipf, 1935/1949) says a word's frequency is roughly inversely proportional to its rank:

```
f(r) ≈ C / r^s      with s ≈ 1
```

so `r · f(r) ≈ C` (constant). The second word occurs about half as often as the first, the third about a third as often, and so on.

**Representation.** Take logs of both sides:

```
log f(r) = log C − s · log r
```

On a **log–log plot** of frequency against rank, Zipf's law gives a straight line with slope −s ≈ −1. The standard representation is:
1. A rank–frequency table: rank, word, frequency f, relative frequency, and the product r·f.
2. A log–log plot of rank against frequency, with a fitted line using least squares on log10 values.
3. The fitted exponent s and the fit quality R².

**Known deviations to look for:**
- **Head (low ranks).** Function words (*the, of, and* / *de, la, que, el*) often flatten the curve. Mandelbrot's correction `f(r) = C / (r + b)^s` accounts for this.
- **Tail (high ranks).** Many words occur only once (*hapax legomena*), which produces a "staircase" of equal frequencies.
- **Language.** Spanish has richer morphology: verb conjugation, gender and number agreement. That gives more types and more hapaxes for the same number of tokens. Expect a higher type/token ratio, a longer tail and a slightly different slope than English.
- **Domain.** Technical text repeats domain terms (*URL, malicious, model, dataset* / *URL, maliciosas, modelo*). These terms reach unusually high ranks compared with general-language text.

## 3. Prerequisites

**System**
- Python **3.14.10** (pinned in `.python-version`, both at the repo root and in `seminars/seminar-3/`), managed with `uv`
- `pdftotext` (poppler-utils, already installed) to extract PDF text. It needs no extra Python dependency.

**Python packages**

| Package | Use | Python 3.14 support (checked on PyPI) |
|---|---|---|
| `spacy` | Tokenization (`token.is_alpha`) and lemmatization, with one model per language | 3.8.18, cp314 wheels |
| `seaborn` | Log–log plots (pulls in `matplotlib` and `numpy`) | 0.13.2, pure Python |
| `numpy` | Least-squares fit of the exponent s (`numpy.polyfit`); installed with seaborn | 2.5.4, cp314 wheels |
| `pandas` | Tables (rank–frequency, metrics); installed with seaborn | 3.1.1 |
| `ipykernel` | Kernel for running the notebook from the editor (no JupyterLab) | 7.4.0 |
| `en_core_web_sm`, `es_core_news_sm` | spaCy models, one per language | 3.8.0, pure Python wheels |

Counting uses the standard library's `collections.Counter`. NLTK is not needed: spaCy handles tokenization and lemmas, and no preloaded corpora are used.

The spaCy models are project dependencies pinned to their GitHub wheel URLs (`[tool.uv.sources]` in `pyproject.toml`). Do not use `spacy download`: models installed that way are not in the lockfile, so the next `uv sync` uninstalls them.

Setup (already done):

```bash
cd seminars/seminar-3
uv init --python 3.14.10
uv add spacy seaborn numpy ipykernel
uv add "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
       "es_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl"
uv sync
```

Run `zipf.ipynb` from the editor, with the kernel from `seminars/seminar-3/.venv`. The `main.py` created by `uv init` is unused.

## 4. Steps

1. **Extract text.** Run `pdftotext <pdf> -` (plain mode, no `-layout`) for corpora A and B, called from the notebook.
2. **Clean.** Remove the front matter (title page, table of contents), bibliography, appendices, code listings, tables, page numbers and headers/footers. Then remove URLs, emails, numbers and punctuation. Keep only alphabetic tokens, including Spanish accented letters (`á é í ó ú ü ñ`). Lowercase everything.
3. **Tokenize.** Run the text through the spaCy model for its language and keep tokens where `token.is_alpha` is true. Raise `nlp.max_length` if a text is longer than spaCy's default limit.
4. **Equalize size.** Set N = the smaller of the two token counts. Truncate both corpora to their first N tokens; only the larger one actually changes. Keep the full token lists as well. Steps 5–9 run on both versions: **full** (original size) and **truncated** (size N).
5. **Count.** Build a `collections.Counter` for each corpus, sort with `most_common()` and assign ranks.
6. **Fit.** Fit `log10 f` against `log10 r` with `numpy.polyfit(deg=1)` to get the slope −s and R². Optionally, fit only ranks 10–1000 to skip the head and tail effects.
7. **Plot.** Use seaborn (`sns.lineplot`, with log scale on both axes) to draw a log–log figure with both curves plus the ideal line `C/r`. The plots show inline in the notebook and are also saved to `plots/` with `savefig` (see section 7).
8. **(Optional) Lemmas.** Repeat steps 5–7 on `token.lemma_` instead of word forms. This shows how much of the English/Spanish difference comes from morphology.
9. **Compute the docx table values.** For each corpus, compute one value for each column of the Teams table:

   | docx column | What to compute | How |
   |---|---|---|
   | Corpus source | Title, author, institution and year (section 1) | — |
   | Corpus dimension | Number of tokens N and number of types V | `N = sum(counts.values())`, `V = len(counts)` |
   | Zipf graphic (top 200 words) | Log–log rank–frequency plot of the 200 most frequent words | `counts.most_common(200)` with seaborn, saved to `plots/` |
   | The 7 most frequent words (in order) | Top 7 types by frequency | `counts.most_common(7)` |
   | % of tokens covered by the 7 most frequent words, excluding stop words | Remove stop words, take the top 7 remaining words, then divide their total frequency by N | Filter with `token.is_stop` (spaCy stop list for each language) |
   | % covered by words that appear once, as types | Hapaxes / V × 100 | `h = sum(1 for c in counts.values() if c == 1)` |
   | % covered by words that appear once, as tokens | Hapaxes / N × 100 | Same `h`; only the denominator changes |

   The column "The 7 most frequent words" does not say whether stop words count. It will almost certainly list only function words (*the, of, and* / *de, la, que*). Record both versions: the plain top 7 and the top 7 excluding stop words, which is the same list used for the coverage column. Check with the instructor which version goes in the docx.

   In the notebook, also show the Zipf exponent s and R² from step 6.

   Compute every value for 4 cases: EN full, EN truncated, ES full, ES truncated. **The docx gets the truncated values**, so both corpora are reported at the same size N. In the docx "Corpus dimension" column, give N and also note the original size (e.g. "N tokens, truncated from M"). The full values are only for the full-vs-truncated comparison in the notebook.

## 5. Comparisons to discuss

- **A vs B (English vs Spanish, same domain).** Differences in type/token ratio, hapax share, slope s and the most frequent words. Function words should dominate both lists. Domain terms should be shared or translated equivalents.
- **Against colleagues' work (the docx table).** This is the "another corpus" comparison. Once colleagues fill in their rows, compare your values with theirs: corpora from other domains and genres, such as novels or news. Expect your technical corpora to concentrate domain vocabulary at higher ranks and to show lower lexical variety.
- **Consistency.** Check whether s stays close to 1 in every corpus. That shows Zipf's law holds across languages and genres.

## 6. Deliverables

All paths are under `~/uni/ACL/seminars/seminar-3/`.

| Path | Content |
|---|---|
| `zipf.ipynb` | The work and the report together. Code cells do steps 1–9; markdown cells explain each step and record the observations. There is no separate report file. |
| `plots/` | PNG copies of the figures, saved from the notebook with `fig.savefig(PLOTS_DIR / "<name>.png", dpi=150)`. |

**Plots** (in `plots/`):
- `zipf_top200_en.png` and `zipf_top200_es.png`: the docx Zipf graphic. A log–log rank–frequency plot of the 200 most frequent words, with the fitted line and the values of s and R². Paste one into each docx row.
- `zipf_en_es.png`: both corpora on one figure (all ranks), with the ideal `C/r` line, for the A vs B comparison.
- `zipf_full_vs_trunc.png`: full vs truncated curves of the larger corpus.
- `zipf_lemmas_en_es.png`: the same as `zipf_en_es.png` but on lemmas (only if step 8 is done).

Also deliver:
- The completed row in the MS Teams docx table for your semigroup.
- If you were absent from the seminar: a PDF export of the notebook (from the editor: export to HTML, then print to PDF).

## 7. Notebook backbone (`zipf.ipynb`)

Cells in order. **MD** = markdown cell, **PY** = code cell. Each entry says what the cell must contain; you write the cell. Names in `code` are suggestions, so keep them consistent across cells.

### Part 0 — Setup

**Cell 1 · MD — Title and goal**
- Title: *Seminar 3 — Zipf's law on English and Spanish CS theses*.
- Author and date.
- Two or three sentences on the goal: Zipf representation for one English and one Spanish corpus from the same domain, a comparison between them, then a comparison with colleagues' corpora from the docx table.

**Cell 2 · MD — Corpora**
- The corpora table from section 1: title, author, institution, year, source and raw word count.
- One line: the instructor approved using the thesis alone, even though it is below 50k words.

**Cell 3 · PY — Imports and configuration**
- Imports: `re`, `subprocess`, `collections.Counter`, `pathlib.Path`, `numpy as np`, `pandas as pd` (installed with seaborn), `matplotlib.pyplot as plt`, `seaborn as sns`, `spacy`.
- `BASE = Path.home() / "uni/ACL/seminars/seminar-3"`, `CORPUS_DIR = BASE / "corpus"`, `PLOTS_DIR = BASE / "plots"`. Create the plots directory with `mkdir(exist_ok=True)`.
- `CORPORA`: a dict with keys `"en"` and `"es"`. Each value holds `pdf` (path), `model` (`"en_core_web_sm"` / `"es_core_news_sm"`), and `start` / `end` line numbers (filled in after cell 6).
- `sns.set_theme()`.

### Part 1 — Text extraction and cleaning

**Cell 4 · MD — Step 1: extraction**
- Why plain `pdftotext` and not a Python PDF library: it is already installed and adds no dependency.
- Plain mode (no `-layout`) keeps reading order, so words are not split across columns.

**Cell 5 · PY — Extract**
- `extract(pdf: Path) -> list[str]`: run `subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True)` and return `.stdout.splitlines()`.
- Call it for both corpora and print the line count of each.

**Cell 6 · PY — Find body boundaries (exploration)**
- For each corpus, print `(line number, line)` for lines matching the chapter headings. English: `Introduction|Conclusions|Bibliography`. Spanish: `Introducción|Conclusiones|Bibliografía`.
- From the output, choose `start` (the first body line of the Introduction chapter, after the table of contents) and `end` (the first line of the Bibliography), then write them into `CORPORA` in cell 3.
- Indices from `extract()` (Python's `splitlines()` also splits on the form feed `\f` that `pdftotext` puts at each page break, so these do not match `grep -n` line numbers):
  - EN: `Introduction` chapter heading at index 178, first `Bibliography` after it at 4438 → `start=178`, `end=4438`.
  - ES: first `Introducción` after the table of contents at index 259, first `Bibliografía` at 4027 → `start=259`, `end=4027`. Both headings repeat later because they are running page headers.

**Cell 7 · MD — Step 2: cleaning rules**
- Keep only the body (Introduction up to the Bibliography): no title page, abstract, table of contents, bibliography or appendices.
- Drop running headers and footers: any line that repeats 4 or more times in the body, plus English `CHAPTER n. ...` headers (`CHAPTER 1. INTRODUCTION` appears only 3 times). The threshold 4 also catches the Spanish `Introducción` header (4 times) and the `o` / `§` bullet lines.
- Drop lines that are only numbers (page numbers).
- Remove URLs and email addresses. The theses are about URLs, so many URLs appear in the text; they are data, not words.
- Numbers and punctuation are removed later at the token level.
- Known limit: text inside tables, figure captions and code still gets through. Mention it in the limitations cell.

**Cell 8 · PY — Clean**
- `clean(lines, start, end) -> str`:
  1. `body = lines[start:end]`
  2. `repeated = {l for l, c in Counter(l.strip() for l in body).items() if c >= 4 and l}`
  3. Keep lines where `l.strip() not in repeated`, `not l.strip().isdigit()` and `not re.match(r"CHAPTER \d+\.", l.strip())`.
  4. Join with `"\n"`, then `re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)`.
- Store the result in `texts = {"en": ..., "es": ...}`.
- Print the first 500 characters of each and the number of repeated lines removed, to check the result by eye.

### Part 2 — Tokens

**Cell 9 · MD — Step 3: tokens vs types, and tokenization**
- The definitions from section 2 (a token is an occurrence, a type is a distinct word, with the "the model and the data" example).
- Tokenization rule: spaCy model for each language; keep only `token.is_alpha` tokens (letters only, including accented Spanish letters); lowercase them.

**Cell 10 · PY — Tokenize**
- `nlp = spacy.load(model, disable=["parser", "ner"])` for each language. Keep the lemmatizer for step 8. Set `nlp.max_length = len(text) + 1` if a text is longer than the default.
- `words[lang] = [t.lower_ for t in doc if t.is_alpha]`
- `lemmas[lang] = [t.lemma_.lower() for t in doc if t.is_alpha]`
- `stops[lang] = nlp.Defaults.stop_words` (a set of lowercase words).
- Print the token count of each corpus.

**Cell 11 · MD — Step 4: equal size**
- Why: Zipf curves, the type count and the hapax share all grow with corpus size, so a comparison is only fair at the same N.
- Rule: N = the smaller of the two token counts. Both corpora are truncated to their first N tokens; only the larger one changes.
- Both versions are kept: **full** (original size) and **truncated** (size N). The docx reports the truncated values; the full values are used for the full-vs-truncated comparison (cell 22).

**Cell 12 · PY — Equalize**
- Keep the full lists: `words_full = words`, `lemmas_full = lemmas`.
- `N = min(len(w) for w in words_full.values())`
- `words_trunc = {lang: w[:N] for lang, w in words_full.items()}`, and the same for `lemmas_trunc`.
- `assert all(len(w) == N for w in words_trunc.values())`
- Print a small table: language, full size, truncated size N, and the percentage of tokens dropped.

### Part 3 — Zipf

**Cell 13 · MD — Step 5: Zipf's law**
- A short version of section 2: rank, `f(r) ≈ C / r^s`, `r·f ≈ C`, straight line with slope −s on a log–log plot, and the head/tail deviations.

**Cell 14 · PY — Rank–frequency table**
- `counts[(lang, version)] = Counter(...)` for `version` in `"full"` and `"trunc"`, so there are 4 counters: EN/ES × full/truncated.
- `rank_table(counts) -> pd.DataFrame` with columns `rank`, `word`, `f`, `rel_f` (f / N) and `r_f` (rank × f), built from `counts.most_common()`.
- Display the first 20 rows for each language (truncated version).

**Cell 15 · MD — Step 6: fitting s**
- Fit a straight line to `log10 f` against `log10 r` by least squares. The slope is −s and the intercept is log10 C. R² measures how close the points lie to the line.

**Cell 16 · PY — Fit**
- `fit_zipf(freqs: list[int]) -> tuple[float, float, float]` returning `(s, C, r2)`:
  - `x = np.log10(np.arange(1, len(freqs) + 1))`, `y = np.log10(freqs)`
  - `slope, intercept = np.polyfit(x, y, 1)`, then `s = -slope`, `C = 10**intercept`, `r2 = np.corrcoef(x, y)[0, 1] ** 2`
- Self-check in the same cell: on synthetic data `freqs = [round(1000 / r) for r in range(1, 101)]`, assert `abs(s - 1) < 0.05` and `r2 > 0.99`.
- Call it on all 4 counters, for the top 200 words (the docx graphic) and for all ranks. Print s and R² for each.

**Cell 17 · MD — Step 7: plots**
- What each plot shows (list from section 6) and how to read it: points close to the line mean the corpus follows Zipf's law; a flat head or a staircase tail are the expected deviations.

**Cell 18 · PY — Top-200 Zipf plots (docx graphic)**
- `plot_zipf(counts, lang, top=200)`: `sns.scatterplot` of rank against frequency for `counts.most_common(top)`, with `ax.set(xscale="log", yscale="log")`. Overlay the fitted line `C / r**s` and put `s` and `R²` in the title or legend. Label the axes "rank" and "frequency".
- Use the truncated counters (these go in the docx). Save as `plots/zipf_top200_en.png` and `plots/zipf_top200_es.png`.

**Cell 19 · PY — EN vs ES on one figure**
- Both truncated corpora (all ranks) on one log–log figure, using `hue` for the language, plus the dashed ideal line `C / r` (C = the frequency of rank 1).
- Save as `plots/zipf_en_es.png`.

### Part 4 — docx table values

**Cell 20 · MD — Step 9: docx columns**
- The column-to-computation table from step 9 in section 4.
- The note on the "7 most frequent words" column: both versions are shown and the instructor decides which one goes in the docx.

**Cell 21 · PY — Metrics**
- `metrics(counts, stop_words) -> dict` with keys:
  - `tokens` = `sum(counts.values())`, `types` = `len(counts)`
  - `top7` = `[w for w, _ in counts.most_common(7)]`
  - `cover7_%` = 100 × the sum of the `top7` frequencies / tokens (stop words included)
  - `top7_nostop` = the first 7 words of `counts.most_common()` that are not in `stop_words`
  - `cover7_nostop_%` = 100 × the sum of their frequencies / tokens
  - `hapax` = the number of words with count 1, then `hapax_types_%` = 100 × hapax / types and `hapax_tokens_%` = 100 × hapax / tokens
  - `s` and `r2` from cell 16 (top 200)
- Show all 4 cases side by side: `pd.DataFrame({"en full": ..., "en trunc": ..., "es full": ..., "es trunc": ...})`.
- Copy only the **truncated** columns into the docx, with the original size noted in the "Corpus dimension" column.

**Cell 21b · PY — Top 7 words with counts**
- `top7_df`: a separate DataFrame, rows = rank 1–7, MultiIndex columns (case, `top7` / `top7_nostop`, `word` / `f`), for all 4 cases. Display the truncated columns for the docx.

**Cell 22 · MD — Full vs truncated: what to expect**
- Only the larger corpus changes. For it, truncation lowers the token and type counts. Hapax shares usually rise slightly, because in a shorter text more words have appeared only once so far.
- The percentages (top-7 coverage, hapaxes as tokens) and the slope s should change only a little if the text is homogeneous. A large change means the dropped part of the text differs in style or topic (e.g. a different chapter).

**Cell 23 · PY — Full vs truncated comparison**
- For each language, a table with the metrics from cell 21 for full vs truncated, plus a `diff` column (truncated − full).
- One log–log figure with the full and truncated curves of the larger corpus (`hue` = version). Save as `plots/zipf_full_vs_trunc.png`.

### Part 5 — Optional: lemmas

**Cell 24 · MD — Step 8: why lemmas**
- Spanish has more inflection than English, so it has more word forms for each lemma. Counting lemmas instead of forms shows how much of the EN/ES difference comes from morphology.

**Cell 25 · PY — Lemma Zipf**
- Repeat cells 14, 16 and 19 on `Counter(lemmas_trunc[lang])`. Save as `plots/zipf_lemmas_en_es.png`. Print the type count and s for forms vs lemmas. Done: the cells 24–28 were written into the notebook directly; cell 27 (colleagues) is a template to fill in.

### Part 6 — Analysis

**Cell 26 · MD — Observations** (written after running everything)
- How well each corpus follows Zipf's law: s and R², and where the points leave the line (head, tail).
- EN vs ES (truncated, same N): types, hapax shares and slope, and whether the gap shrinks with lemmas.
- Full vs truncated: which values changed for the larger corpus and by how much (from cell 23).
- The top 7 words with and without stop words: which domain terms appear (*url, malicious, model* / *url, maliciosas, modelo*) and how much of the text they cover.

**Cell 27 · MD — Comparison with colleagues** (filled in after the docx table has other rows)
- Compare your values with colleagues' corpora (other domains, genres or languages): the type count at a similar size, hapax shares, top-7 coverage and the shape of the Zipf plot.
- Expectation to check: technical text repeats domain terms, so it has fewer types and higher top-7 coverage than literary text.

**Cell 28 · MD — Limitations**
- The English corpus is below 50k tokens; both corpora are truncated to the smaller size N, so the larger one loses part of its text.
- Some noise from tables, captions and code remains after cleaning.
- spaCy's English and Spanish stop lists differ in size, which affects the "excluding stop words" values.
- The English thesis was written by the analyst (originality approved by the instructor).
