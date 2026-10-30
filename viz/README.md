# OTTER ontology explorer

`ontology_explorer.html` is a single self-contained page (no CDN, no external fonts, works from `file://`). It draws one lesson graph per subject from the frozen OTTER ontology (`data/ontology/ontology.sqlite3`, read-only) and shows, for every node, its type, id, support label and **source pointers** (book, lesson heading, printed page, span id). Spans are not shown: the point is that every item links to a span you can open in the book.

Open it with any browser: `xdg-open viz/ontology_explorer.html`. Regenerate with `.venv/bin/python scripts/release/viz_build_explorer.py` from the repository root of the research code.

## Which lessons are shown (selection rule)

For each of the six subjects, one lesson is shown. Rule (stated here, applied mechanically by the script, no manual picking): build the supported subgraph of every lesson (a node is an item with at least one evidence span in that lesson; edges are typed relations, fact/rule-to-concept mentions and `teaches` links between such nodes); take the largest connected component; among the lessons whose largest component has **at most 150 nodes**, take the lesson with the **largest** component (ties: smaller lesson id). Only that component is drawn; other nodes of the lesson are not shown. A seventh *demo* entry is the lesson cited by the video's example question (`VIEDUHOP-C2-PHY-017`, gold lesson `nxbgd_g10_physics_d0daa774:ch03:l14`); it is not chosen by the rule.

| subject | lesson (as in the book) | book | nodes drawn | edges | nodes of the lesson in the DB (supported) | role |
|---|---|---|---|---|---|---|
| Vật lí (Physics) | Bài 8 MÔ TẢ SÓNG | Vật lí 11 | 44 | 53 | 55 | selected |
| Hóa học (Chemistry) | BÀI 6 MỘT SỐ HỢP CHẤT CỦA NITROGEN VỚI OXYGEN | Hóa học 11 | 66 | 70 | 68 | selected |
| Sinh học (Biology) | 12 MIỄN DỊCH Ở NGƯỜI VÀ ĐỘNG VẬT | Sinh học 11 | 120 | 140 | 149 | selected |
| Lịch sử (History) | Bài 4 KHÁI NIỆM VĂN MINH. MỘT SỐ NỀN VĂN MINH PHƯƠNG ĐÔNG THỜI CỔ – TRUNG ĐẠI | Lịch sử 10 | 140 | 156 | 234 | selected |
| Địa lí (Geography) | BÀI 1 VỊ TRÍ ĐỊA LÍ VÀ PHẠM VI LÃNH THỔ | Địa lí 12 | 70 | 83 | 72 | selected |
| Ngữ văn (Literature) | Bài 6 NGUYỄN TRÃI “DÀNH CÒN ĐỂ TRỢ DÂN NÀY”(*) | Ngữ văn 10, tập hai | 125 | 143 | 288 | selected |
| Vật lí (Physics) | Bài 14 ĐỊNH LUẬT 1 NEWTON | Vật lí 10 | 24 | 30 | 41 | demo |

## What is and is not drawn

* Only items labelled `supported` by the DB (source-support check of Qwen3-8B, an automatic label, **not human verification**) are used. `partially_supported` and `unsupported` items are excluded; their counts are in the Stats tab. Learning objectives and `teaches` links carry no support label in the DB; they are drawn in their own shape and marked *not labelled*.
* Node types: concept (circle), fact (square), rule (diamond), objective (triangle). Edges: typed relation (solid, arrow), fact/rule mentions concept (dashed), `teaches` (dotted).
* No textbook passage or quote is shipped. A statement (fact text, rule atom text, formula, objective line, concept label) is shown only if it has <= 25 words **and** shares no run of 8 consecutive words with any of its evidence spans, nor (stricter than required) with any other span of the 18 books (NFC, lower case, word tokens; checked programmatically against `data/textbooks/parsed/*/spans.jsonl`). Free-text rule arguments get the same check. Otherwise the page shows only the id/type/pointers.
* Vietnamese: concept names are the textbook's own terms and are shown untranslated (the DB has no English); subject names, lesson headings and the legend carry English glosses.

## Verbatim check: how many statements passed

Shipped in the 7 lesson graphs (all text fields that could have been shown):

| statement kind | passed (shown) | failed (id only) |
|---|---|---|
| concept_label | 348 | 13 |
| fact | 31 | 148 |
| objective | 1 | 20 |
| rule | 14 | 14 |
| rule_arg | 3 | 0 |
| rule_atom | 41 | 17 |

All supported facts / rules / objectives in the whole ontology (for reference; not all are shipped):

| kind | passed | failed |
|---|---|---|
| fact | 678 | 2934 |
| rule | 130 | 548 |
| objective | 28 | 692 |

## Whole-ontology numbers vs. the paper (`paper/sections/method.tex`)

| item | paper | DB (v1 frozen) | equal |
|---|---|---|---|
| concepts | 5,939 | 5,939 | yes |
| relations | 2,929 | 2,929 | yes |
| facts | 3,830 | 3,830 | yes |
| rules | 787 | 787 | yes |
| rules_strict_executable | 650 | 650 | yes |
| objectives | 720 | 720 | yes |
| evidence_links | 18,111 | 18,111 | yes |
| rejected | 1,371 | 1,371 | yes |

The paper's counts include items of every support label (supported / partially / unsupported); the explorer draws only `supported` ones.

## Files

* `ontology_explorer.html`: the page (data inlined, so it also works offline from a file).
* `data/lessons.json`: the 7 lesson graphs (nodes, edges, positions, pointers).
* `data/rule_gallery.json`: 2 strict-executable rules per subject (rule view).
* `data/ontology_stats.json`: counts, support breakdown, comparison with the paper, text-check counters.
* `data/selection.json`: the selection rule and the top candidates per subject.

