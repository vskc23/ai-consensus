import glob
import os
import json

import main

assert main.needs_search("What is Oracle's stock rating today?")
assert main.needs_search("Latest Gemini model released?")
assert not main.needs_search("Explain how a hash table works")
# the eval finding: this question must trigger a search now
assert main.needs_search(
    "Is it good to invest in the upcoming Anthropic IPO? "
    "When can an everyday investor register for IPO allotment?"
)
assert "release date" in main.search_hints("What is the release date of Gemini 4?")

# --- config.json is the source of truth; env vars can still override ---
assert main.MODELS["Gemini"] == "google/gemini-4-argon"
assert main.JUDGE_MODEL == "anthropic/claude-opus-5.5"
assert main.ANSWER_MAX_TOKENS == 1024
assert main.load_config()["models"]["DeepSeek"] == "deepseek/deepseek-v4-pro"
os.environ["MODEL_GEMINI"] = "tmp/override-model"
try:
    assert main.load_config()["models"]["Gemini"] == "tmp/override-model"
finally:
    del os.environ["MODEL_GEMINI"]

# --- the judge LITERALLY sees the question, the evidence, every
#     answer, TODAY, and the date rules (provability check) ---
judge_prompt = main.build_evaluation_prompt(
    "When is the concert?",
    {"OpenAI": "The concert is on [S1] November 1st."},
    "evidence text about the concert [S1]",
    [{"title": "T", "url": "https://example.org/x", "snippet": "hi",
      "date": "2026-10-20"}],
    [{"question": "q?", "answer": "a"}],
)
assert "When is the concert?" in judge_prompt
assert "The concert is on [S1] November 1st." in judge_prompt
assert "evidence text about the concert [S1]" in judge_prompt
assert "https://example.org/x" in judge_prompt
assert "(published 2026-10-20)" in judge_prompt
assert "TODAY" in judge_prompt
assert "Date integrity" in judge_prompt and "Future dates" in judge_prompt

sources = [
    {"title": "T", "url": "https://x", "snippet": "s1", "date": ""},
    {"title": "U", "url": "https://y", "snippet": "s2", "date": "2026-01-01"},
]
ev = main.format_evidence(sources)
assert "[S1] T" in ev and "[S2] U (2026-01-01)" in ev

item = {"accuracy": 8, "grounding": 7, "completeness": 7,
        "reasoning": 8, "relevance": 9, "clarity": 8}
assert main.weighted_score(item) == 7.7, main.weighted_score(item)

history = [{"question": "q1", "answer": "a1"}]
msgs = main.build_messages("q2", ev, history)
assert msgs[0]["role"] == "system"
assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
tail = msgs[-1]["content"]
assert tail.startswith("Today's date and time:")  # models get a "now"
assert "2026" in main.current_datestamp()
assert "WEB EVIDENCE" in tail and "QUESTION:\nq2" in tail

from main import extract_json
assert extract_json('prefix\n```json\n{"a": 1}\n```\nsuffix') == {"a": 1}

# --- slugify / unique_path ---
assert main.slugify("  What is Oracle's rating TODAY?! ") == \
    "what-is-oracles-rating-today"
assert main.unique_path("/nope/x.json") == "/nope/x.json"

main.ask_model = lambda model, messages: f"answer from {model}\n[S2] used"
main.evaluate_answers = lambda q, a, e, s, h: {
    "scores": [
        {"answer_number": i+1, "accuracy": 8, "grounding": 7,
         "completeness": 7, "reasoning": 8, "relevance": 9,
         "clarity": 8, "cited_sources": [2], "explanation": "ok"}
        for i in range(len(a))
    ],
    "final_answer": "synthesized using [S2].",
    "confidence": "high",
    "disagreements": [],
    "missing_information": [],
}
# --- results/ and logs/ got one timestamped file each ---
# Snapshot anything that existed before, so cleanup below can never
# delete pre-existing records (the eval history is the source of truth).
pre_results = set(glob.glob(f"{main.RESULTS_DIR}/*.json"))
pre_logs = set(glob.glob(f"{main.LOGS_DIR}/*.json"))

main.search_web = lambda q, max_results=None: (sources, None)
fa, answers, src = main.run_turn("q?", [], "on")
assert fa == "synthesized using [S2]."
assert list(answers) == list(main.MODELS)
assert len(src) == 2

# --- multi-part questions are searched as focused sub-queries ---
assert main.split_questions(
    "When is Benson Boone in Arkansas ? Also Taylor Swift?") == [
        "When is Benson Boone in Arkansas",
        "Taylor Swift",
    ]
assert main.split_questions(
    "What are the best restaurants in Paris and London?") == [
        "What are the best restaurants in Paris and London",
    ]
searched_queries = []

def fake_search(query, max_results=None):
    searched_queries.append(query)
    return ([{"title": query[:8], "url": f"https://r/{len(searched_queries)}",
              "snippet": "s", "date": ""}], None)

main.search_web = fake_search
main.evaluate_answers = lambda q, a, e, s, h: {
    "scores": [], "final_answer": "ok", "confidence": "low",
    "disagreements": [], "missing_information": [],
}
fa2, _, src2 = main.run_turn(
    "When is Benson Boone in Arkansas? Also Taylor Swift?", [], "on"
)
assert searched_queries == [
    "When is Benson Boone in Arkansas",
    "Taylor Swift",
], searched_queries
assert len(src2) == 2 and fa2 == "ok"
for folder in (main.RESULTS_DIR, main.LOGS_DIR):
    for path in glob.glob(f"{folder}/*.json"):
        data = json.load(open(path, encoding="utf-8"))
        if data.get("question", "").startswith("When is Benson Boone"):
            assert data["search"]["sub_queries"] == [
                "When is Benson Boone in Arkansas",
                "Taylor Swift",
            ]
            assert data["search"]["result_count"] == 2

created_results = sorted(set(glob.glob(f"{main.RESULTS_DIR}/*.json")) -
                         pre_results)
created_logs = sorted(set(glob.glob(f"{main.LOGS_DIR}/*.json")) - pre_logs)
assert created_results and created_logs, (created_results, created_logs)

for path in created_results + created_logs:
    # private permissions
    assert os.stat(path).st_mode & 0o777 == 0o600, path

with open(created_results[0], encoding="utf-8") as file:
    result_json = json.load(file)
assert result_json["question"] == "q?"
assert "answers" in result_json and "evaluation" in result_json
assert result_json["failed_models"] == {}

with open(created_logs[0], encoding="utf-8") as file:
    audit_json = json.load(file)
assert "search" in audit_json and "sources_used" in audit_json
assert audit_json["search"]["attempted"] is True
assert audit_json["search"]["result_count"] == 2
assert audit_json["search"]["error"] is None
assert audit_json["search"]["sub_queries"] == ["q"]
assert audit_json["failed_models"] == {}
assert audit_json["model_citations"]["OpenAI"] == [2]
assert audit_json["final_answer_citations"] == [2]
assert audit_json["sources_used"][0]["source"] == "S1"

# nothing secret ever lands in the records
secret_hint = ("api_key", "secret", "token", "tvly-", "sk-or-")
for path in created_results + created_logs:
    text = open(path, encoding="utf-8").read().lower()
    assert not any(hint in text for hint in secret_hint), path

# clean up ONLY the records this test created
for path in created_results + created_logs:
    os.remove(path)
for folder in (main.RESULTS_DIR, main.LOGS_DIR):
    if os.path.isdir(folder) and not glob.glob(f"{folder}/*.json"):
        os.rmdir(folder)


# ---- REPL loop: commands, history, follow-ups (all offline) ----
calls = []

def fake_run_turn(q, h, m):
    calls.append((q, list(h), m))
    return "SYNTH", {"OpenAI": "a"}, [{"title": "T", "url": "u",
                                      "snippet": "s", "date": ""}]

main.run_turn = fake_run_turn
main.API_KEY = "test-key"  # stubbed: the real .env key is empty right now
script = iter([
    "/help",
    "first question",
    "/sources",
    "/search on",
    "second followup",
    "/answers",
    "/clear",
    "/exit",
])
main.input = lambda prompt="": next(script)
main.main()

assert len(calls) == 2, calls
assert calls[0] == ("first question", [], "auto")
assert calls[1] == ("second followup",
                    [{"question": "first question", "answer": "SYNTH"}],
                    "on")
print("ALL TESTS PASSED (unit + run_turn + REPL loop)")
