"""Small IR system over CACM: inverted index, stop list, one-word TF-IDF search."""

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TAGS = re.compile(r"</?(html|pre)>")
CITATION = re.compile(r"^\d+\t\d+\t\d+$")
RECORD = re.compile(r"^CA\d{6}")
DATE = re.compile(r"^(CACM )?\w+ ?,? ?\d{4}$")
CR_NUMBERS = re.compile(r"^[\d.,\s]+$")
AUTHOR_LINE = re.compile(r"^[A-Z][^,]*,[^,]*$")  # "Lamport, L." -- exactly one comma
WORD = re.compile(r"[a-z]+")


def strip_tags(text: str) -> str:
    return TAGS.sub("", text)


def is_bookkeeping(line: str) -> bool:
    line = line.strip()
    return bool(CITATION.match(line) or RECORD.match(line) or DATE.match(line))


def clean(text: str) -> str:
    """Tag-stripped text without citation, record and date lines."""
    return "\n".join(l for l in strip_tags(text).splitlines() if not is_bookkeeping(l))


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def title_and_keywords(text: str) -> str:
    """Title block + keyword block (if any). Keywords sit right before the CR-number block."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", strip_tags(text)) if b.strip()]
    if not blocks:
        return ""
    title, rest = blocks[0], blocks[1:]
    dates = [i for i, b in enumerate(rest) if DATE.match(b)]
    if dates:
        rest = rest[dates[0] + 1 :]
    rest = [b for b in rest if not RECORD.match(b) and not CITATION.match(b.splitlines()[0])]
    cr = [i for i, b in enumerate(rest) if CR_NUMBERS.match(b)]
    if not cr or cr[0] == 0:
        return title  # keywords only appear together with CR categories
    kw = rest[cr[0] - 1]
    # ponytail: author/keyword split by "one comma per line" heuristic; misreads rare one-comma keyword blocks
    if cr[0] == 1 and all(AUTHOR_LINE.match(l.strip()) for l in kw.splitlines()):
        return title  # the only block before CR numbers is the author list
    return title + "\n" + kw


def top_terms(docs_tokens: dict[str, list[str]], k: int) -> list[str]:
    counts = Counter(t for toks in docs_tokens.values() for t in toks)
    return [t for t, _ in counts.most_common(k)]


@dataclass
class IR:
    index: dict[str, dict[str, int]]  # term -> {doc_id: tf}
    doc_len: dict[str, int]  # doc_id -> tokens after preprocessing
    stop_words: list[str]
    fields: dict[str, set[str]]  # doc_id -> title + keyword terms
    tokens_before: int
    tokens_after: int
    corpus_bytes: int

    @classmethod
    def from_texts(cls, texts: dict[str, str], stop_words: list[str] | None = None, n_stop: int = 7, corpus_bytes: int = 0):
        toks = {d: tokenize(clean(t)) for d, t in texts.items()}
        if stop_words is None:
            stop_words = top_terms(toks, n_stop)
        stop = set(stop_words)
        index: dict[str, dict[str, int]] = {}
        doc_len = {}
        for d, ts in toks.items():
            ts = [t for t in ts if t not in stop]
            doc_len[d] = len(ts)
            for t, n in Counter(ts).items():
                index.setdefault(t, {})[d] = n
        fields = {d: set(tokenize(title_and_keywords(t))) - stop for d, t in texts.items()}
        return cls(
            index=index,
            doc_len=doc_len,
            stop_words=stop_words,
            fields=fields,
            tokens_before=sum(len(strip_tags(t).split()) for t in texts.values()),
            tokens_after=sum(doc_len.values()),
            corpus_bytes=corpus_bytes or sum(len(t.encode()) for t in texts.values()),
        )

    @classmethod
    def build(cls, path: str | Path, n_stop: int = 7):
        files = sorted(Path(path).glob("CACM-*.html"))
        texts = {f.stem: f.read_text(encoding="utf-8", errors="replace") for f in files}
        return cls.from_texts(texts, n_stop=n_stop, corpus_bytes=sum(f.stat().st_size for f in files))

    @property
    def vocabulary_size(self) -> int:
        return len(self.index)

    @property
    def index_bytes(self) -> int:
        return len(pickle.dumps(self.index))

    def search(self, word: str) -> list[tuple[str, float]]:
        terms = tokenize(word)
        if len(terms) != 1 or terms[0] not in self.index:
            return []  # stop words never enter the index
        postings = self.index[terms[0]]
        idf = math.log(len(self.doc_len) / len(postings))
        scored = [(d, tf / self.doc_len[d] * idf) for d, tf in postings.items()]
        return sorted(scored, key=lambda x: (-x[1], x[0]))

    def relevant(self, word: str) -> set[str]:
        terms = tokenize(word)
        return {d for d, f in self.fields.items() if terms and terms[0] in f}


def precision_at(results: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d, _ in results[:k]) / k


def docx_line(ir: IR, word: str, results=None, relevant=None) -> str:
    results = ir.search(word) if results is None else results
    relevant = ir.relevant(word) if relevant is None else relevant
    top = ", ".join(d for d, _ in results[:3]) or "none"
    p3, p7 = precision_at(results, relevant, 3), precision_at(results, relevant, 7)
    return f"{word} — {len(results)} matching documents; top 3: {top}; P@3 = {p3:.2f}; P@7 = {p7:.2f}."


if __name__ == "__main__":
    toy = {
        "D1": "<html><pre>\nParsing compilers\n\ncompiler compiler design\n\nCACM May, 1970\n\nSmith, A.\n\n"
        "compiler, parsing\n\n4.12\n\nCA700501 JB March 1, 1978\n\n1\t5\t1\n</pre></html>",
        "D2": "<html><pre>\nSorting methods\n\nthe compiler of a sort with many many many other words here\n\n"
        "CACM June, 1971\n\nJones, B.\n\nCA710601 JB March 1, 1978\n</pre></html>",
        "D3": "<html><pre>\nThe Fast Sort\n\nCACM July, 1972\n\nCA720701 JB March 1, 1978\n</pre></html>",
    }
    ir = IR.from_texts(toy, stop_words=["the", "of", "a"])
    assert not {"the", "of", "a"} & ir.index.keys()
    assert not {"cacm", "may", "june", "jb", "march", "ca"} & ir.index.keys(), "bookkeeping lines indexed"
    assert ir.index["compiler"] == {"D1": 3, "D2": 1}  # title "compilers" is a different token
    assert [d for d, _ in ir.search("Compiler")] == ["D1", "D2"]
    assert ir.search("zebra") == [] and ir.search("the") == []
    assert precision_at([("D1", 1.0), ("D2", 0.5)], {"D1"}, 3) == 1 / 3
    assert ir.relevant("compiler") == {"D1"}  # D2 has it only in the abstract
    assert "design" not in ir.fields["D1"]  # abstract terms are not in fields
    assert ir.relevant("sort") == {"D3"}
    assert ir.tokens_after < ir.tokens_before
    print("self-check ok")
