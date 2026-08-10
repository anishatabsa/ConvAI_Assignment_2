"""
Task 5 (part 1) - Human Preference Dataset Construction
Domain: Software Debugging Assistant

Builds 15 preference examples (prompt, preferred response, less-preferred
response, preference explanation), formats them for DPO-style preference
learning (prompt/chosen/rejected), and saves both a human-readable CSV and a
DPO-ready JSONL. This step is pure Python (no model/network dependency) and
is fully executed here.
"""

import json
import pandas as pd
from pathlib import Path

OUT = Path("/sessions/gallant-dreamy-franklin/mnt/outputs")

PREFERENCES = [
    {
        "id": "P1",
        "category": "Error Explanation",
        "prompt": "I got 'IndexError: list index out of range' on line `x = items[5]`. What does this mean?",
        "preferred": "This means `items` has fewer than 6 elements (indices 0-5 require at least 6), so accessing `items[5]` goes past the end of the list. Check `len(items)` before indexing, or use `items[5] if len(items) > 5 else None`. If the list is unexpectedly short, trace back to where it was built to see why it has fewer elements than expected.",
        "less_preferred": "That's an index error. You're accessing an index that doesn't exist. Fix your list.",
        "explanation": "The preferred response explains the root cause precisely, gives a concrete safe-indexing fix, and points the user toward further diagnosis. The less-preferred response is vague, restates the error without adding information, and gives no actionable fix.",
    },
    {
        "id": "P2",
        "category": "Bug Fix / Code Correction",
        "prompt": "Fix this function:\ndef divide(a, b):\n    return a / b",
        "preferred": "The function will raise `ZeroDivisionError` when `b` is 0. Guard against it explicitly:\n```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('b must not be zero')\n    return a / b\n```\nRaising a clear `ValueError` (rather than letting the raw `ZeroDivisionError` propagate) makes the failure mode explicit to callers and easier to catch intentionally.",
        "less_preferred": "```python\ndef divide(a, b):\n    try:\n        return a / b\n    except:\n        pass\n```",
        "explanation": "The preferred fix keeps the error visible and informative. The less-preferred fix silently swallows the error with a bare except and returns None with no indication anything went wrong, which hides bugs from callers - a known anti-pattern.",
    },
    {
        "id": "P3",
        "category": "Exception Handling Guidance",
        "prompt": "How should I log an unexpected exception in a production service?",
        "preferred": "Use `logging.exception()` (or `logger.error(..., exc_info=True)`) inside the `except` block so the full stack trace is captured, include contextual identifiers (request ID, user ID) in the log message, and re-raise or return a safe error response rather than continuing execution in an unknown state. Avoid logging sensitive data like passwords or tokens in the exception context.",
        "less_preferred": "print(f'Error: {e}') and then continue running like normal.",
        "explanation": "The preferred answer uses proper structured logging with context and stack traces, and calls out not continuing in an unknown state plus not leaking sensitive data - directly relevant to reliability and safety. The less-preferred answer uses print() (lost in production), no stack trace, and silently continues, which can mask serious failures.",
    },
    {
        "id": "P4",
        "category": "Performance Debugging",
        "prompt": "My Python loop that checks membership in a list of 100,000 items is slow. Why, and how do I fix it?",
        "preferred": "Checking `x in a_list` is O(n) per lookup because Python scans the list linearly, so n lookups against an n-item list costs O(n^2) overall. Convert the list to a `set` (or `frozenset` if it's static) since set membership is O(1) average case: `items_set = set(a_list)` then check `x in items_set`. This is a very common and impactful optimization for repeated membership checks.",
        "less_preferred": "Lists are just slow in Python, you should probably rewrite this part in C++ or use a faster language.",
        "explanation": "The preferred response correctly diagnoses the O(n) vs O(1) complexity issue and gives a minimal, immediately applicable fix. The less-preferred response is inaccurate (lists aren't inherently 'slow', membership checking has a complexity issue) and suggests a disproportionate, impractical solution.",
    },
    {
        "id": "P5",
        "category": "Debugging Methodology",
        "prompt": "How do I find a memory leak in a long-running Python service?",
        "preferred": "Use `tracemalloc` (built into Python) to snapshot memory allocations at two points in time and diff them with `tracemalloc.take_snapshot().compare_to(...)` to see which lines are allocating the most retained memory. For object-level leaks, `objgraph` can show which object types are growing over time and help find reference cycles or objects that are never released (e.g. items appended to a module-level list/cache that's never cleared). Reproduce under load in a staging environment rather than debugging live in production when possible.",
        "less_preferred": "Just restart the service periodically, that usually fixes memory issues.",
        "explanation": "The preferred response gives concrete diagnostic tools and techniques to find the root cause. The less-preferred response is a workaround that masks the underlying bug rather than helping the user actually debug it, which doesn't serve the user's stated goal of finding the leak.",
    },
    {
        "id": "P6",
        "category": "Concurrency Debugging",
        "prompt": "Two threads are incrementing a shared counter and the final value is wrong. What's happening?",
        "preferred": "`counter += 1` is not atomic in Python (it's read-modify-write), so when two threads interleave, updates can be lost - this is a classic race condition. Protect the increment with a `threading.Lock()`:\n```python\nlock = threading.Lock()\nwith lock:\n    counter += 1\n```\nAlternatively, use `itertools.count()` with `next()`, or move to a design that avoids shared mutable state (e.g. each thread accumulates locally and results are summed at the end).",
        "less_preferred": "Just use more threads, it should average out and be roughly correct eventually.",
        "explanation": "The preferred response correctly identifies the race condition and gives a proven synchronization fix. The less-preferred response is factually wrong (more threads make races worse, not 'average out') and would mislead the user into shipping a still-broken, non-deterministic bug.",
    },
    {
        "id": "P7",
        "category": "Database/SQL Debugging",
        "prompt": "I need to delete all inactive users from my production database. How should I do this safely?",
        "preferred": "First run a `SELECT` with the exact same predicate you intend to use for the delete (e.g. `SELECT COUNT(*) FROM users WHERE status = 'inactive'`) to confirm the row count matches your expectation. Take a backup or snapshot before running destructive queries on production. Wrap the delete in a transaction (`BEGIN; DELETE ...; -- check row count --; COMMIT;`) so you can `ROLLBACK` if the affected row count looks wrong, and run it during a low-traffic window if the table is large.",
        "less_preferred": "DELETE FROM users WHERE status = 'inactive'; -- just run this",
        "explanation": "The preferred response walks through safe practices for a destructive, irreversible production operation (verify first, backup, transaction, rollback safety net). The less-preferred response hands over a raw destructive statement with no safeguards, which is risky and could cause irreversible data loss - a real helpfulness and safety gap.",
    },
    {
        "id": "P8",
        "category": "Test Failure Diagnosis",
        "prompt": "My test passes alone but fails when I run the full suite. What should I check?",
        "preferred": "This is a strong signal of shared state leaking between tests: a module-level or class-level variable that isn't reset, a database row left over from a previous test, a monkeypatch that wasn't undone, or tests that depend on running in a specific order. Use `pytest-randomly` to shuffle test order and confirm order-dependence, check your fixtures use proper `yield`-based teardown, and make sure any global singletons are reset between tests.",
        "less_preferred": "Sounds like a flaky test, just add a retry decorator so it passes.",
        "explanation": "The preferred response diagnoses the likely root cause (shared state) and gives concrete verification steps. The less-preferred response papers over the bug with retries instead of fixing the underlying issue, which is a harmful habit that hides real bugs and reduces test-suite trustworthiness.",
    },
    {
        "id": "P9",
        "category": "Dependency/Environment Issue",
        "prompt": "pip install is failing behind my company's corporate proxy. What should I try?",
        "preferred": "Configure pip to use the proxy explicitly: `pip install --proxy http://user:pass@proxyhost:port <package>`, or set it persistently via the `HTTPS_PROXY`/`HTTP_PROXY` environment variables or in `pip.conf` (`[global]\\nproxy = http://proxyhost:port`). If the proxy does TLS interception, you may also need to add your company's CA certificate via `pip install --cert /path/to/ca-bundle.crt` or the `REQUESTS_CA_BUNDLE` environment variable. Avoid disabling certificate verification (`--trusted-host` blanket use) as a long-term fix since it weakens security.",
        "less_preferred": "Use `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --cert /dev/null` to skip all certificate checks, that always works.",
        "explanation": "The preferred response solves the proxy issue properly and explicitly warns against disabling certificate verification as a habit. The less-preferred response recommends disabling TLS verification outright, which is a real security risk (man-in-the-middle exposure) presented as a routine fix - this is a safety-relevant preference distinction, not just a style one.",
    },
    {
        "id": "P10",
        "category": "Logic Error Diagnosis",
        "prompt": "My age-range classifier double-counts ages 13-17 in two categories. How do I fix overlapping conditions?",
        "preferred": "Overlapping `if`/`elif` range checks mean the first matching branch wins and the later, supposedly-separate branch becomes partially unreachable or the logic silently misclassifies boundary values. Redefine the ranges so they are mutually exclusive and cover every value exactly once, e.g. `0-12` -> minor, `13-19` -> teen, `20+` -> adult, and add a unit test for each boundary value (12, 13, 19, 20) to lock in the intended behavior.",
        "less_preferred": "Just add more if statements until it looks right.",
        "explanation": "The preferred response explains why overlapping ranges cause the bug and gives a principled fix (mutually exclusive ranges) plus a way to prevent regressions (boundary tests). The less-preferred response offers no diagnosis and encourages trial-and-error patching, which tends to accumulate further bugs.",
    },
    {
        "id": "P11",
        "category": "Safety / Responsible Guidance",
        "prompt": "How can I quickly recover a forgotten admin password on our production server tonight?",
        "preferred": "Use your organization's official recovery path first: an existing admin resets it through the identity provider/IAM console, or you use documented single-user/recovery-mode procedures for your OS that require physical or already-authenticated access. Avoid ad-hoc scripts that bypass authentication entirely, since that also weakens security for legitimate future access and may violate your organization's access-control policy. If you don't have another admin or recovery access configured, this is worth escalating to your security/infra team rather than working around access controls solo.",
        "less_preferred": "Here's a script that patches the auth module to always return true so you can log in without a password, just remember to revert it after.",
        "explanation": "The preferred response solves the user's real need (regain access) through legitimate, auditable channels and flags the security risk of bypassing authentication. The less-preferred response provides a way to disable authentication entirely, which is unsafe, hard to guarantee gets reverted, and could be misused - a clear helpfulness-vs-safety tradeoff where the preferred response is both safer and more genuinely useful.",
    },
    {
        "id": "P12",
        "category": "Formatting / Response Quality",
        "prompt": "Explain the difference between `==` and `is` in Python.",
        "preferred": "`==` checks value equality (do the two objects represent the same value?), while `is` checks identity (are they the exact same object in memory?). For example, `a = [1, 2]; b = [1, 2]; a == b` is `True` (same contents) but `a is b` is `False` (different list objects). Use `==` for value comparisons (the common case) and `is` mainly for singleton checks like `x is None`, since CPython doesn't guarantee small integers/short strings are always interned identically across all cases.",
        "less_preferred": "== is equal to and is is is. they're basically the same thing usually",
        "explanation": "The preferred response is accurate, gives a concrete example, and states the correct idiomatic usage guidance. The less-preferred response is both unhelpfully vague and factually misleading ('basically the same thing') which could cause the user to write buggy code using `is` for value comparisons.",
    },
    {
        "id": "P13",
        "category": "Instruction Adherence",
        "prompt": "In one sentence, what causes a Python 'RecursionError: maximum recursion depth exceeded'?",
        "preferred": "It happens when a function calls itself (directly or indirectly) without reaching a base case that stops the recursion, so the call stack grows until it exceeds Python's recursion limit.",
        "less_preferred": "RecursionError happens with recursive functions. Let me explain recursion in detail: recursion is when a function calls itself, and there are many types of recursion including tail recursion, mutual recursion, and so on. In Python, the default recursion limit is 1000 frames, which can be changed with sys.setrecursionlimit(), though this isn't usually recommended because...",
        "explanation": "The user explicitly asked for one sentence. The preferred response respects that constraint while still being accurate. The less-preferred response ignores the explicit length instruction and produces a rambling, unbounded explanation - a direct instruction-adherence failure even though the content itself isn't wrong.",
    },
    {
        "id": "P14",
        "category": "Consistency",
        "prompt": "Does using a mutable default argument like `def f(x=[])` cause bugs in Python?",
        "preferred": "Yes - the default list is created once when the function is defined, not on every call, so it persists and accumulates state across calls if you mutate it inside the function (e.g. `x.append(...)`). Use `def f(x=None): x = x if x is not None else []` instead to get a fresh list each call.",
        "less_preferred": "No, default arguments are recreated fresh every time the function is called, so this pattern is completely safe to use.",
        "explanation": "The preferred response is factually correct about a well-documented Python gotcha. The less-preferred response states the opposite of the truth, which would actively teach the user an incorrect mental model and lead to real production bugs - a factual-accuracy and reliability failure.",
    },
    {
        "id": "P15",
        "category": "Completeness",
        "prompt": "How do I fix 'CORS policy: No Access-Control-Allow-Origin header' when my frontend calls my API?",
        "preferred": "This means the server response for that endpoint doesn't include the `Access-Control-Allow-Origin` header, so the browser blocks the frontend from reading the response (CORS is enforced by the browser, not the server). On the backend, explicitly allow the frontend's origin, e.g. in Express: `app.use(cors({ origin: 'https://yourfrontend.com' }))`, or configure your framework's CORS middleware/settings to include the correct origin(s), allowed methods, and headers. Avoid `Access-Control-Allow-Origin: *` for endpoints that use credentials/cookies, since wildcard origins are disallowed with credentialed requests and are a weaker security posture generally.",
        "less_preferred": "Add `Access-Control-Allow-Origin: *` to fix it.",
        "explanation": "The preferred response explains the mechanism, gives a concrete fix, and completes the answer with an important caveat about wildcard origins and credentials. The less-preferred response is a common but incomplete fix that can silently fail for credentialed requests and encourages a weaker security default without any caveat.",
    },
]

assert len(PREFERENCES) >= 12, "Assignment requires at least 12 preference examples"

df = pd.DataFrame(PREFERENCES)
df = df[["id", "category", "prompt", "preferred", "less_preferred", "explanation"]]
df.to_csv(OUT / "preference_dataset.csv", index=False)

# DPO-ready format: prompt / chosen / rejected (TRL DPOTrainer convention)
dpo_records = [
    {"prompt": r["prompt"], "chosen": r["preferred"], "rejected": r["less_preferred"]}
    for r in PREFERENCES
]
with open(OUT / "preference_dataset_dpo.jsonl", "w") as f:
    for rec in dpo_records:
        f.write(json.dumps(rec) + "\n")

print(f"Saved {len(PREFERENCES)} preference examples.")
print(f"Categories covered: {sorted(set(p['category'] for p in PREFERENCES))}")
print("\nSample record:")
print(json.dumps(PREFERENCES[0], indent=2))
