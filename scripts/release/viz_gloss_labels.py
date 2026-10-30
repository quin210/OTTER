#!/usr/bin/env python3
"""Optional: add short English glosses for the concept labels shown in the explorer, using the LOCAL vLLM server
(Qwen3-32B at http://127.0.0.1:8004/v1, see configs/default.yaml). No external API is used.

Command (after viz_build_explorer.py has written viz/data/lessons.json):

    .venv/bin/python scripts/release/viz_gloss_labels.py && .venv/bin/python scripts/release/viz_build_explorer.py

Writes release/github/OTTER/viz/data/gloss_en.json  {concept_id: english gloss}. The glosses are machine translations
of short concept names, NOT checked by a human, and the explorer labels them as such. Only labels already shown
(they passed the copyright check) are sent. Deterministic (temperature 0); existing glosses are kept.
"""
import json, os, re, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ = os.path.join(ROOT, "release/github/OTTER/viz/data")
URL = "http://127.0.0.1:8004/v1/chat/completions"


def ask(batch):
    lines = "\n".join("%d\t%s" % (i + 1, t) for i, (_, t) in enumerate(batch))
    prompt = ("Translate each Vietnamese school-textbook concept name into a short English term (1-6 words). "
              "Keep proper names in their usual English form. Output exactly one line per input, as '<number>\\t<English>', nothing else.\n\n" + lines)
    body = {"model": "Qwen/Qwen3-32B", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    res = {}
    for ln in out.splitlines():
        m = re.match(r"\s*(\d+)\s*[\t:.\-]?\s*(\S.*)", ln)
        if m and 1 <= int(m.group(1)) <= len(batch):
            res[batch[int(m.group(1)) - 1][0]] = m.group(2).strip()
    return res


def main():
    lessons = json.load(open(os.path.join(VIZ, "lessons.json"), encoding="utf-8"))["lessons"]
    path = os.path.join(VIZ, "gloss_en.json")
    gloss = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    todo, seen = [], set()
    for l in lessons:
        for n in l["nodes"]:
            if n["k"] == "concept" and n.get("label") and n["id"] not in gloss and n["id"] not in seen:
                seen.add(n["id"]); todo.append((n["id"], n["label"]))
    print("to gloss:", len(todo))
    for i in range(0, len(todo), 40):
        gloss.update(ask(todo[i:i + 40]))
    json.dump(gloss, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    print("glosses:", len(gloss))


if __name__ == "__main__":
    main()
