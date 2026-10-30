# data/

Empty on purpose. The textbooks, their OCR text and the ontology built from them are not redistributed (publisher copyright).
* Benchmark: the ViEduHop dataset on Hugging Face (questions, options, keys, source pointers; no passages).
* Corpus and ontology: build them from your own legal copies of the books with `python -m otter.builder` (see the top-level README).
* Expected layout when you do: `data/textbooks/parsed/<book_id>/{pages,spans}.jsonl`, `data/textbooks/ontology.sqlite3`, `data/ontology/ontology.sqlite3`.
