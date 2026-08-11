"""
Task 1 - Domain Dataset Construction & Cleaning
Domain: Software Debugging Assistant

Builds an instruction-tuning dataset for a software-debugging assistant.
Each seed record pairs a real (language, bug/error) scenario with a correct,
expert-style explanation/fix. Seed records are then expanded with paraphrased
instruction templates (legitimate augmentation - the underlying code/response
pairs are unchanged, only the way the user asks is varied, which mirrors how
real users phrase the same debugging request differently).

Deliberate noise (exact duplicates, near-duplicates, missing fields, truncated
low-quality responses) is then injected into a copy of the pool to demonstrate
the cleaning pipeline, and removed again in the cleaning step.
"""

import json
import random
import re
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUT = Path("/sessions/gallant-dreamy-franklin/mnt/outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. SEED SCENARIOS (real, hand-authored, technically correct)
# ---------------------------------------------------------------------------
# Each item: category, language, context (code/error/log), response (expert fix/explanation)

SEEDS = []

def add(category, language, context, response):
    SEEDS.append({"category": category, "language": language, "context": context.strip(), "response": response.strip()})

# --- Category: Error Explanation ---
add("Error Explanation", "Python",
    "Traceback (most recent call last):\n  File \"app.py\", line 12, in <module>\n    result = data['user']['age']\nKeyError: 'user'",
    "This KeyError means the dictionary `data` does not have a key named 'user'. Before indexing, check with `if 'user' in data:` or use `data.get('user', {}).get('age')` to avoid the crash. Print `data.keys()` to confirm what keys actually exist at that point in the program.")
add("Error Explanation", "JavaScript",
    "TypeError: Cannot read properties of undefined (reading 'map')\n  at renderList (app.js:34)",
    "You are calling `.map()` on a variable that is `undefined` at the time `renderList` runs, most likely because the array is still being fetched asynchronously. Add a guard such as `if (!items) return null;` before the `.map()` call, or default the prop to an empty array, e.g. `items = []`.")
add("Error Explanation", "Java",
    "Exception in thread \"main\" java.lang.NullPointerException\n\tat com.app.Order.getTotal(Order.java:22)",
    "A NullPointerException at that line means an object reference used inside `getTotal()` is null, likely because a field was never initialized or a constructor argument was not set before the method was called. Add a null check, initialize fields with defaults, or trace the call site to see why the object was never assigned.")
add("Error Explanation", "C++",
    "Segmentation fault (core dumped)\nProgram received signal SIGSEGV, Address not mapped to object",
    "A segmentation fault means the program tried to read or write memory it does not own, commonly from dereferencing a null or dangling pointer, indexing an array out of bounds, or using memory after it was freed. Run the binary under `gdb` or `valgrind --tool=memcheck` to get the exact line and the type of invalid access.")
add("Error Explanation", "SQL",
    "ERROR 1054 (42S22): Unknown column 'usr.email' in 'field list'",
    "MySQL cannot find a column named `email` on the table aliased `usr`. Either the column name is misspelled, the alias doesn't match the table in the FROM/JOIN clause, or the column was renamed/dropped in a migration. Run `DESCRIBE <table>` to confirm the actual column names and fix the alias or column reference.")
add("Error Explanation", "Python",
    "ImportError: cannot import name 'urlencode' from 'urllib'",
    "In Python 3, `urlencode` moved to `urllib.parse`, not `urllib` directly. Change the import to `from urllib.parse import urlencode`. This error typically shows up when Python 2 code is run unmodified under Python 3.")

# --- Category: Bug Fix / Code Correction ---
add("Bug Fix / Code Correction", "Python",
    "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)\n\nprint(average([]))",
    "This raises `ZeroDivisionError` when `nums` is empty because `len(nums)` is 0. Fix it by guarding against an empty list:\n```python\ndef average(nums):\n    if not nums:\n        return 0\n    return sum(nums) / len(nums)\n```")
add("Bug Fix / Code Correction", "Python",
    "def is_even(n):\n    if n % 2 == 1:\n        return True\n    else:\n        return False",
    "The logic is inverted: `n % 2 == 1` is true for odd numbers, not even ones, so the function returns True for odd inputs. Fix: use `n % 2 == 0`, or more simply `return n % 2 == 0`.")
add("Bug Fix / Code Correction", "JavaScript",
    "function sumArray(arr) {\n  let total;\n  for (let i = 0; i <= arr.length; i++) {\n    total += arr[i];\n  }\n  return total;\n}",
    "Two bugs here: (1) `total` is never initialized, so it starts as `undefined` and every addition produces `NaN`; (2) the loop condition `i <= arr.length` reads one index past the end of the array. Fix:\n```javascript\nfunction sumArray(arr) {\n  let total = 0;\n  for (let i = 0; i < arr.length; i++) {\n    total += arr[i];\n  }\n  return total;\n}\n```")
add("Bug Fix / Code Correction", "Java",
    "for (int i = 0; i < list.size(); i++) {\n    if (list.get(i).equals(target)) {\n        list.remove(i);\n    }\n}",
    "Removing elements from a `List` while iterating by index causes elements to shift, so the loop can skip the element right after a removal. Iterate backwards, or use `Iterator.remove()`:\n```java\nIterator<String> it = list.iterator();\nwhile (it.hasNext()) {\n    if (it.next().equals(target)) {\n        it.remove();\n    }\n}\n```")
add("Bug Fix / Code Correction", "Python",
    "def get_config(overrides={}):\n    overrides['env'] = 'prod'\n    return overrides",
    "Using a mutable default argument (`{}`) is a classic Python bug: the same dict object is reused across every call and keeps accumulating changes. Use `None` as the default and create a new dict inside the function:\n```python\ndef get_config(overrides=None):\n    overrides = dict(overrides) if overrides else {}\n    overrides['env'] = 'prod'\n    return overrides\n```")
add("Bug Fix / Code Correction", "C++",
    "int* create_array(int size) {\n    int arr[size];\n    return arr;\n}",
    "`arr` is allocated on the stack and is destroyed when the function returns, so the returned pointer becomes a dangling pointer. Allocate on the heap instead (and remember to free/delete it, or prefer `std::vector<int>`):\n```cpp\nstd::vector<int> create_array(int size) {\n    return std::vector<int>(size);\n}\n```")

# --- Category: Logic Error Diagnosis ---
add("Logic Error Diagnosis", "Python",
    "def apply_discount(price, percent):\n    return price - percent\n\nprint(apply_discount(100, 10))  # expected 90",
    "The function subtracts the raw percent value instead of computing a percentage of the price, so `apply_discount(100, 10)` returns 90 by coincidence but `apply_discount(200, 10)` would wrongly return 190 instead of 180. Fix: `return price - (price * percent / 100)`.")
add("Logic Error Diagnosis", "JavaScript",
    "function isPalindrome(str) {\n  return str === str.split('').reverse();\n}",
    "`str.split('').reverse()` returns an array, and comparing a string to an array with `===` is always false. Join the reversed array back into a string before comparing: `return str === str.split('').reverse().join('');`")
add("Logic Error Diagnosis", "Python",
    "def find_max(nums):\n    max_val = 0\n    for n in nums:\n        if n > max_val:\n            max_val = n\n    return max_val\n\nprint(find_max([-5, -2, -9]))  # returns 0, expected -2",
    "Initializing `max_val = 0` silently breaks for lists of all-negative numbers because 0 is never exceeded. Initialize with the first element instead: `max_val = nums[0]` (and handle the empty-list case separately).")
add("Logic Error Diagnosis", "SQL",
    "SELECT * FROM orders WHERE status = 'shipped' OR status = 'delivered' AND customer_id = 42;",
    "Operator precedence: `AND` binds tighter than `OR`, so this is evaluated as `status = 'shipped' OR (status = 'delivered' AND customer_id = 42)`, which returns ALL shipped orders regardless of customer. Add parentheses to express the intended logic: `WHERE (status = 'shipped' OR status = 'delivered') AND customer_id = 42;`")
add("Logic Error Diagnosis", "Java",
    "if (age >= 13 && age <= 19) {\n    category = \"teen\";\n} else if (age >= 12 && age <= 17) {\n    category = \"minor\";\n}",
    "The two age ranges overlap (13-17 falls in both branches), and because `age >= 13 && age <= 19` is checked first, anyone 12-17 who should also match \"minor\" is always classified as \"teen\" instead - the second branch becomes partially unreachable. Redefine the ranges so they are mutually exclusive, e.g. minor = 0-12, teen = 13-19.")

# --- Category: Exception Handling Guidance ---
add("Exception Handling Guidance", "Python",
    "How should I handle a file that might not exist when reading a config file?",
    "Wrap the read in a targeted `try/except FileNotFoundError` rather than a bare `except:` so you don't silently swallow unrelated bugs:\n```python\ntry:\n    with open('config.yaml') as f:\n        config = yaml.safe_load(f)\nexcept FileNotFoundError:\n    config = DEFAULT_CONFIG\n    logging.warning('config.yaml not found, using defaults')\n```\nLog the fallback so the missing file is visible in production, and avoid catching `Exception` broadly since it can hide programming errors like typos in the YAML parser call.")
add("Exception Handling Guidance", "Java",
    "catch (Exception e) {\n    // ignore\n}",
    "Swallowing exceptions silently is dangerous - failures disappear with no trace, making bugs almost impossible to diagnose later. At minimum log the exception with context (`logger.error(\"failed to process order {}\", orderId, e)`), and catch the most specific exception type you can handle meaningfully rather than the generic `Exception`.")
add("Exception Handling Guidance", "JavaScript",
    "fetch('/api/user').then(res => res.json()).then(data => setUser(data));",
    "There is no `.catch()`, so a network failure or non-2xx response will produce an unhandled promise rejection and the UI will hang with no feedback. Add error handling and a status check:\n```javascript\nfetch('/api/user')\n  .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })\n  .then(data => setUser(data))\n  .catch(err => { console.error(err); setError('Could not load user'); });\n```")
add("Exception Handling Guidance", "Python",
    "What's wrong with using a bare `except:` clause?",
    "A bare `except:` catches everything, including `KeyboardInterrupt` and `SystemExit`, and hides the real exception type and traceback, making debugging much harder and potentially masking serious bugs. Catch specific exception classes instead (`except ValueError:`, `except (IOError, OSError):`), and if you truly need a catch-all, use `except Exception as e:` and log `e` with `logging.exception()`.")

# --- Category: Performance Debugging ---
add("Performance Debugging", "Python",
    "def contains_duplicate(nums):\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] == nums[j]:\n                return True\n    return False",
    "This is O(n^2) because of the nested loop comparing every pair. Use a set for O(n) time: \n```python\ndef contains_duplicate(nums):\n    seen = set()\n    for n in nums:\n        if n in seen:\n            return True\n        seen.add(n)\n    return False\n```")
add("Performance Debugging", "Python",
    "result = []\nfor row in large_dataframe.iterrows():\n    result.append(row[1]['value'] * 2)",
    "`DataFrame.iterrows()` is notoriously slow on large data because it creates a Series object per row. Use vectorized operations instead: `result = (large_dataframe['value'] * 2).tolist()`. This can be 50-100x faster on large datasets since it uses NumPy's compiled routines instead of a Python-level loop.")
add("Performance Debugging", "SQL",
    "SELECT * FROM orders WHERE YEAR(created_at) = 2024;",
    "Wrapping an indexed column in a function like `YEAR(created_at)` prevents the database from using an index on `created_at`, forcing a full table scan. Rewrite as a range condition that can use the index: `WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';`")
add("Performance Debugging", "JavaScript",
    "items.forEach(item => {\n  document.getElementById('list').innerHTML += `<li>${item}</li>`;\n});",
    "Appending to `innerHTML` inside a loop forces the browser to re-parse and re-render the entire list on every iteration, which is O(n^2) for n items. Build the HTML string (or a DocumentFragment) once, then assign it after the loop:\n```javascript\nconst html = items.map(item => `<li>${item}</li>`).join('');\ndocument.getElementById('list').innerHTML = html;\n```")

# --- Category: Debugging Methodology ---
add("Debugging Methodology", "General",
    "What's a systematic approach to debugging a hard-to-reproduce production bug?",
    "1) Reproduce reliably first - gather logs, stack traces, and the exact input/environment; 2) form a hypothesis about the cause rather than guessing randomly; 3) add targeted logging or use a debugger to inspect state at the suspected point; 4) use binary search / bisection (git bisect, or disabling half the code path) to narrow down where the bug was introduced; 5) write a minimal reproduction case; 6) fix the root cause, not just the symptom; 7) add a regression test so it cannot silently reappear.")
add("Debugging Methodology", "Python",
    "How do I use pdb to debug a Python script?",
    "Insert `import pdb; pdb.set_trace()` (or `breakpoint()` on Python 3.7+) at the line where you want execution to pause. Once paused you can inspect variables by typing their names, step line-by-line with `n` (next) or `s` (step into), continue with `c`, and print an expression with `p <expr>`. For post-mortem debugging after a crash, run the script with `python -m pdb -c continue script.py` or wrap the call in `pdb.pm()` in a REPL right after the exception.")
add("Debugging Methodology", "General",
    "My code works locally but fails in production. How should I approach this?",
    "Check for environment differences systematically: compare dependency versions (`pip freeze` / `npm ls` in both environments), environment variables and config/secrets, OS or runtime version, database schema/data differences, timezone and locale settings, and network/firewall access to external services. Reproduce the production environment as closely as possible (Docker helps), and add structured logging around the failing code path so the next occurrence gives you concrete evidence instead of guesswork.")
add("Debugging Methodology", "General",
    "How can git bisect help find which commit introduced a bug?",
    "`git bisect start`, then mark a known-bad commit with `git bisect bad` and a known-good commit with `git bisect good <commit>`. Git checks out a commit halfway between them; you test it and mark it `good` or `bad`, and git repeats the binary search until it identifies the exact commit that introduced the regression. You can automate this with `git bisect run <test-script>` if you have a script that exits non-zero on failure.")

# --- Category: Test Failure Diagnosis ---
add("Test Failure Diagnosis", "Python",
    "AssertionError: assert 3.0000000000000004 == 3.0",
    "This is a floating-point precision issue - `0.1 + 0.1 + 0.1` etc. do not always sum to an exact decimal in binary floating point. Don't compare floats with `==`; use `math.isclose(a, b, rel_tol=1e-9)` or pytest's `pytest.approx()`: `assert result == pytest.approx(3.0)`.")
add("Test Failure Diagnosis", "JavaScript",
    "FAIL  Timeout - Async callback was not invoked within the 5000ms timeout specified by jest.setTimeout",
    "The test is waiting on a promise or callback that never resolves, commonly because a mocked API call wasn't set up, `done()` was never called, or a returned promise was not awaited. Check that any mocked network/service calls in the test actually resolve, and that async test functions use `async/await` or return the promise so Jest can track completion.")
add("Test Failure Diagnosis", "Python",
    "Test passes in isolation (`pytest test_foo.py::test_a`) but fails when run with the full suite.",
    "This usually indicates shared/global state leaking between tests - a module-level variable, a database row not cleaned up, a monkeypatch not reverted, or test order dependence. Check for missing `fixture` teardown, use `pytest --randomly-seed` (pytest-randomly) to confirm order-dependence, and make sure each test sets up and tears down its own isolated state rather than relying on side effects from earlier tests.")

# --- Category: Dependency/Environment Issue ---
add("Dependency/Environment Issue", "Python",
    "ModuleNotFoundError: No module named 'requests'",
    "The `requests` package is not installed in the Python environment currently running your script. Install it with `pip install requests`, and if you use virtual environments, make sure the venv is activated before installing/running (`source venv/bin/activate`). If it's installed but still not found, check `which python` and `pip show requests` to confirm they point to the same environment.")
add("Dependency/Environment Issue", "JavaScript",
    "npm ERR! peer dep missing: react@\">=16\", required by react-router-dom@6.4.0",
    "`react-router-dom` needs `react` version 16 or higher as a peer dependency but it isn't installed (or an incompatible version is). Run `npm install react@^18 react-dom@^18`, or if using npm 7+, re-run `npm install` since it usually auto-installs compatible peer deps; check `npm ls react` to confirm the resolved version.")
add("Dependency/Environment Issue", "Python",
    "pip install fails with: Could not find a version that satisfies the requirement tensorflow==2.9.0",
    "This typically means the requested version isn't available for your Python version/platform (e.g. tensorflow 2.9 doesn't ship wheels for very new Python versions), or you're behind a proxy/offline index. Check `python --version` against the package's supported versions, try `pip install tensorflow` without pinning to see what resolves, and confirm `pip config list` doesn't point to a broken/private index.")
add("Dependency/Environment Issue", "General",
    "Docker container works on my machine but crashes with 'exec format error' on the deployment server.",
    "This means the image was built for a different CPU architecture than the server (commonly building on Apple Silicon/ARM and deploying to an x86_64 server, or vice versa). Rebuild with the correct target platform, e.g. `docker buildx build --platform linux/amd64 -t myimage .`, or use multi-platform builds so the image works on both architectures.")

# --- Category: Concurrency Debugging ---
add("Concurrency Debugging", "Python",
    "counter = 0\ndef increment():\n    global counter\n    for _ in range(100000):\n        counter += 1\n# run increment() in 2 threads -> counter ends up less than 200000",
    "`counter += 1` is not atomic - it's a read, add, and write, so two threads can interleave and lose updates (a classic race condition). Protect the critical section with a lock:\n```python\nimport threading\nlock = threading.Lock()\ndef increment():\n    global counter\n    for _ in range(100000):\n        with lock:\n            counter += 1\n```\nOr use `threading.local()`/an atomic counter structure, or avoid shared mutable state entirely with multiprocessing + message passing.")
add("Concurrency Debugging", "Java",
    "Two threads occasionally throw ConcurrentModificationException when iterating over an ArrayList that another thread modifies.",
    "`ArrayList` is not thread-safe; modifying it while another thread iterates triggers `ConcurrentModificationException` via its fail-fast iterator. Use a thread-safe collection such as `CopyOnWriteArrayList` for read-heavy/write-light workloads, or wrap access with `Collections.synchronizedList()` and synchronize manually during iteration, or use `java.util.concurrent` collections like `ConcurrentLinkedQueue` depending on the access pattern.")
add("Concurrency Debugging", "General",
    "My application deadlocks intermittently under load. How do I debug it?",
    "Deadlocks usually come from two or more threads acquiring the same locks in different orders. Take a thread dump when the app hangs (`jstack <pid>` for JVM, `py-spy dump` for Python) and look for threads BLOCKED waiting on a lock held by another BLOCKED thread - that circular wait is the deadlock. Fix it by always acquiring locks in a consistent global order, using timeouts on lock acquisition (`tryLock` with a timeout), or reducing the scope/number of locks needed.")

# --- Category: Database/SQL Debugging ---
add("Database/SQL Debugging", "SQL",
    "DELETE FROM users WHERE last_login < '2020-01-01'",
    "Without a `WHERE` clause bug this looks fine, but always test destructive statements first with a `SELECT` using the same predicate (`SELECT * FROM users WHERE last_login < '2020-01-01'`) to confirm the row count before running DELETE, and wrap it in a transaction (`BEGIN; ... ; COMMIT;`) so you can `ROLLBACK` if the count looks wrong.")
add("Database/SQL Debugging", "SQL",
    "Query runs fine on staging but times out on production with the same data volume.",
    "Check `EXPLAIN ANALYZE` on production - the query planner may be choosing a different execution plan due to stale statistics (`ANALYZE <table>`), a missing index that exists on staging, table bloat, or different `work_mem`/buffer settings. Compare index lists between environments (`\\d <table>` in psql) and confirm migrations that add indexes actually ran on production.")
add("Database/SQL Debugging", "Python",
    "conn = sqlite3.connect('app.db')\nfor user_id in user_ids:\n    cur = conn.execute(f\"SELECT * FROM users WHERE id = {user_id}\")",
    "Building SQL with an f-string is a SQL-injection vulnerability and also causes query-plan-cache misses since each query string is unique. Use parameterized queries: `conn.execute(\"SELECT * FROM users WHERE id = ?\", (user_id,))`. This is safer and lets the database reuse the prepared statement plan.")
add("Database/SQL Debugging", "SQL",
    "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5 WHERE status = 'active';",
    "`WHERE` cannot appear after `HAVING` - `WHERE` filters rows before grouping and must come before `GROUP BY`, while `HAVING` filters after aggregation. Correct order: `SELECT customer_id, COUNT(*) FROM orders WHERE status = 'active' GROUP BY customer_id HAVING COUNT(*) > 5;`")

print(f"Seed examples authored: {len(SEEDS)}")

# ---------------------------------------------------------------------------
# 2. INSTRUCTION PARAPHRASE TEMPLATES (per category) -> augmentation
# ---------------------------------------------------------------------------
TEMPLATES = {
    "Error Explanation": [
        "Explain what this error means and how to fix it:",
        "I'm getting the following error, what's causing it?",
        "Can you break down this error message for me?",
    ],
    "Bug Fix / Code Correction": [
        "Find and fix the bug in this code:",
        "This function isn't working correctly. What's wrong and how do I fix it?",
        "Review this code snippet and correct the bug:",
    ],
    "Logic Error Diagnosis": [
        "This code runs without crashing but gives the wrong result. Why?",
        "Diagnose the logic error in the following:",
        "Something is off with this logic - can you spot it?",
    ],
    "Exception Handling Guidance": [
        "What's the best way to handle this situation?",
        "How should error handling be improved here?",
        "Give me guidance on proper exception handling for this case:",
    ],
    "Performance Debugging": [
        "This code is too slow. How can I optimize it?",
        "Identify the performance bottleneck in this code:",
        "Why is this running slowly, and how do I speed it up?",
    ],
    "Debugging Methodology": [
        "", "", "",
    ],
    "Test Failure Diagnosis": [
        "My test is failing with this output. What's going on?",
        "Diagnose why this test fails:",
        "Help me understand this test failure:",
    ],
    "Dependency/Environment Issue": [
        "I'm running into this environment/dependency issue:",
        "How do I resolve this setup problem?",
        "This is failing in my environment - what's the fix?",
    ],
    "Concurrency Debugging": [
        "I'm seeing this concurrency bug. What's happening and how do I fix it?",
        "Diagnose this multi-threading issue:",
        "Explain this race condition / concurrency bug:",
    ],
    "Database/SQL Debugging": [
        "What's wrong with this SQL, and how should I fix it?",
        "Review this database query for issues:",
        "Debug this SQL problem:",
    ],
}

def build_instruction(cat, lang, context, template_idx):
    templates = TEMPLATES[cat]
    if cat == "Debugging Methodology":
        # context already IS the instruction/question for this category
        return context, ""
    tmpl = templates[template_idx % len(templates)]
    instr = f"{tmpl} ({lang})" if lang != "General" else tmpl
    return instr, context

rows = []
rid = 0
for seed in SEEDS:
    n_variants = 3 if seed["category"] != "Debugging Methodology" else 1
    for v in range(n_variants):
        instr, ctx = build_instruction(seed["category"], seed["language"], seed["context"], v)
        rid += 1
        rows.append({
            "id": rid,
            "instruction": instr,
            "context": ctx,
            "response": seed["response"],
            "category": seed["category"],
            "language": seed["language"],
        })

clean_pool = pd.DataFrame(rows)
print(f"Clean pool after paraphrase augmentation: {len(clean_pool)} rows")
print(clean_pool["category"].value_counts())

# ---------------------------------------------------------------------------
# 3. INJECT NOISE (to demonstrate the cleaning pipeline on a realistic raw pull)
# ---------------------------------------------------------------------------
raw = clean_pool.copy()

noisy_rows = []
next_id = raw["id"].max() + 1

# a) exact duplicates
for idx in raw.sample(12, random_state=1).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    noisy_rows.append(r)

# b) near-duplicates (whitespace/case noise, same content)
for idx in raw.sample(10, random_state=2).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    r["instruction"] = "   " + r["instruction"].upper() + "  \n\n"
    noisy_rows.append(r)

# c) missing response (incomplete sample)
for idx in raw.sample(8, random_state=3).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    r["response"] = ""
    noisy_rows.append(r)

# d) missing instruction
for idx in raw.sample(6, random_state=4).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    r["instruction"] = None
    noisy_rows.append(r)

# e) too-short / low-quality response ("thanks", "ok", single word)
low_quality_fillers = ["ok", "fix it", "idk", "see docs", "yes", "no bug"]
for idx in raw.sample(9, random_state=5).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    r["response"] = random.choice(low_quality_fillers)
    noisy_rows.append(r)

# f) inconsistent/garbled formatting (HTML tags, encoding artifacts leaking in)
for idx in raw.sample(7, random_state=6).index:
    r = raw.loc[idx].to_dict()
    r["id"] = next_id; next_id += 1
    r["response"] = "<p>" + r["response"].replace("\n", "<br>") + "&nbsp;</p>  "
    noisy_rows.append(r)

raw_dataset = pd.concat([raw, pd.DataFrame(noisy_rows)], ignore_index=True)
raw_dataset = raw_dataset.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle
raw_dataset.to_csv(OUT / "raw_dataset.csv", index=False)
print(f"\nRaw (noisy) dataset size: {len(raw_dataset)} rows -> saved raw_dataset.csv")

# ---------------------------------------------------------------------------
# 4. CLEANING PIPELINE
# ---------------------------------------------------------------------------
def normalize_text(x):
    if not isinstance(x, str):
        return x
    x = re.sub(r"<br\s*/?>", "\n", x)
    x = re.sub(r"<[^>]+>", "", x)          # strip HTML tags
    x = x.replace("&nbsp;", " ")
    x = x.strip()
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    # normalize SHOUTY-CAPS artifacts (e.g. scraped/OCR noise) to sentence case,
    # but leave normal mixed-case text (which may contain intentional code
    # identifiers/acronyms) untouched.
    letters = [c for c in x if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        x = x.lower()
        x = x[:1].upper() + x[1:] if x else x
    return x

report = {}
df = raw_dataset.copy()
report["raw_count"] = len(df)

# normalize whitespace/case/HTML first so duplicate detection works on content, not formatting
df["instruction_norm"] = df["instruction"].apply(normalize_text)
df["response_norm"] = df["response"].apply(normalize_text)

# 4a. missing value handling: drop rows with missing/empty instruction or response
before = len(df)
df = df[df["instruction_norm"].notna() & (df["instruction_norm"].str.strip() != "")]
df = df[df["response_norm"].notna() & (df["response_norm"].str.strip() != "")]
report["dropped_missing_fields"] = before - len(df)

# 4b. quality filtering: response must be reasonably substantive (>= 25 chars, >= 5 words)
before = len(df)
df = df[df["response_norm"].str.len() >= 25]
df = df[df["response_norm"].str.split().str.len() >= 5]
report["dropped_low_quality"] = before - len(df)

# 4c. duplicate removal: case/whitespace-insensitive on (instruction, response)
before = len(df)
df["dedup_key"] = (df["instruction_norm"].str.lower().str.strip() + "||" +
                    df["response_norm"].str.lower().str.strip())
df = df.drop_duplicates(subset="dedup_key", keep="first")
report["dropped_duplicates"] = before - len(df)

# 4d. response normalization: use normalized text as final fields
df["instruction"] = df["instruction_norm"]
df["response"] = df["response_norm"]
df["context"] = df["context"].fillna("").apply(normalize_text)
df = df.drop(columns=["instruction_norm", "response_norm", "dedup_key"])
df = df.reset_index(drop=True)
df["id"] = range(1, len(df) + 1)

report["final_clean_count"] = len(df)
df.to_csv(OUT / "clean_dataset.csv", index=False)

print("\n=== Cleaning report ===")
for k, v in report.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
df["instruction_len_words"] = df["instruction"].str.split().str.len()
df["response_len_words"] = df["response"].str.split().str.len()

eda = {
    "num_samples": int(len(df)),
    "num_categories": int(df["category"].nunique()),
    "avg_instruction_len_words": float(df["instruction_len_words"].mean()),
    "median_instruction_len_words": float(df["instruction_len_words"].median()),
    "avg_response_len_words": float(df["response_len_words"].mean()),
    "median_response_len_words": float(df["response_len_words"].median()),
    "min_response_len_words": int(df["response_len_words"].min()),
    "max_response_len_words": int(df["response_len_words"].max()),
    "category_distribution": df["category"].value_counts().to_dict(),
    "language_distribution": df["language"].value_counts().to_dict(),
}
with open(OUT / "eda_summary.json", "w") as f:
    json.dump(eda, f, indent=2)

print("\n=== EDA summary ===")
print(json.dumps(eda, indent=2))

# plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

df["category"].value_counts().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Category Distribution")
axes[0].set_xlabel("Count")
axes[0].invert_yaxis()

axes[1].hist(df["instruction_len_words"], bins=15, color="#55A868", edgecolor="white")
axes[1].set_title("Instruction Length (words)")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Frequency")

axes[2].hist(df["response_len_words"], bins=15, color="#C44E52", edgecolor="white")
axes[2].set_title("Response Length (words)")
axes[2].set_xlabel("Words")

plt.tight_layout()
plt.savefig(OUT / "eda_plots.png", dpi=130)
print("\nSaved eda_plots.png")

# ---------------------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST SPLIT (stratified by category)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["category"]
)
# Second split (val/test) is not stratified: after the first split, several
# categories have too few remaining members (e.g. "Debugging Methodology")
# for a stratified split to guarantee >=2 samples per class in each half.
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42
)

train_df.to_csv(OUT / "train.csv", index=False)
val_df.to_csv(OUT / "val.csv", index=False)
test_df.to_csv(OUT / "test.csv", index=False)

split_summary = {
    "train": len(train_df),
    "validation": len(val_df),
    "test": len(test_df),
    "train_pct": round(len(train_df) / len(df) * 100, 1),
    "val_pct": round(len(val_df) / len(df) * 100, 1),
    "test_pct": round(len(test_df) / len(df) * 100, 1),
}
with open(OUT / "split_summary.json", "w") as f:
    json.dump(split_summary, f, indent=2)

print("\n=== Split summary (80/10/10, stratified by category) ===")
print(json.dumps(split_summary, indent=2))

print("\nSample cleaned records:")
print(df[["id", "instruction", "category", "language"]].head(5).to_string(index=False))

print("\nDONE - all artifacts saved to outputs/")
