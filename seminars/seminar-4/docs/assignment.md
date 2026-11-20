# Seminar 4 - Assignment

## IR system. Part 1

Implement a (basic) small IR system.

Use CACM Document Collections. (see : http://www.search-engines-book.com/collections/)

- The system should use an inverted index
- Use a stop list; for example, you could the most freq. 7 words (from experiments_2_results)
- Implement a retrieval functionality, given an one word query

Present 3 test data.
Get estimation for index build time and system answer time.

Add the required information in the *.docx file. The *.docx file is uploaded on Teams,
a description of the required data is presented below.

**Does not apply!** If you are NOT present in class during the seminar, upload a pdf document presenting  the way you processed the data and get the results. (At least 200 words.)

## Docx Required Data

- Time: index build time and system answer time (for a word)
- Stop words (source – nr.of words) Preprocessing (short description)
- Index size vs corpus size; Vocabulary size; Tokens after preprocessing;Tokens before preprocessing
- 3 Test data; (Query word  – Answer-nr.of matching documents; top 3 documents, P@3 , P@7)

### Sample results

```txt
Harangus Paul

Index build time: 0.681757 s

System answer time per query:

computer: 0.6194 ms

space: 0.0576 ms

recipes: 0.0018 ms

system: 0.003 ms

Stop words: CACM collection; 8 words: the, and, cacm, for, are, march, algorithm, system.

Index size vs corpus size: 1,486,899 / 2,080,955 bytes (71.45%).

Vocabulary size: 9,030 distinct terms.

Tokens after preprocessing: 192,057.

Tokens before preprocessing: 220,757.

computer — 710 matching documents; top 3: CACM-2685, CACM-1771, CACM-2329; P@3 = 1.00; P@7 = 1.00.

space — 104 matching documents; top 3: CACM-1901, CACM-2389, CACM-2711; P@3 = 1.00; P@7 = 1.00.

recipes — 0 matching documents; top 3: none; P@3 = 0.00; P@7 = 0.00.

system — 2 matches; P@3 = 0.67; P@7 = 0.29.
```
```
```

## In-class instructions

- preferably we should not query using the test data the same words.
- the demo will be ran using a ipynb notebook; I've created the project before-hand.

# Indications

- data is located in [cacm](../data/cacm/); this should be Git ignored.
- I manage [pyproject.toml](../pyproject.toml), you just offer commands to install packages and manage the uv toolkit.
- helper scripts will be created under the `src` folder, and will be used in the notebook.
- you own the code; markdown cells are also handled by you, but no yapping in them
- the system is a small IR system that complies with the assignment provided.
- a separate `plan.md` will be built as an implementation plan, designed for post-implementation.
