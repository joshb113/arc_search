# arc_search – Mapped Issues INDEX

Diagnosed issues whose **root cause lies outside arc_search** – the test harness, a
library default, Docker, PowerShell, or the OS – plus the confirmed workaround for each.

**Why this file exists:** these look exactly like app bugs and will be re-reported. Each entry
records the *evidence* that proved the cause was external, so a future session can recognise the
symptom immediately instead of re-running the whole investigation — or, worse, "fixing" working
code. ISS-003 and ISS-005 each cost most of an hour; ISS-001 produced a test that passed for
entirely the wrong reason.

**Not for app bugs.** Real defects belong in [[plans/INDEX]] / the relevant plan file.

| ID | Issue | Environment | Status | Workaround |
|----|-------|-------------|--------|------------|
| [[#ISS-001]] | A respx route registered without a port matches requests on **any** port | respx 0.23, all OS | ✅ Worked around (verified) | Register the ported route **first**; first match wins |
| [[#ISS-002]] | Docker Desktop: "Virtualization support not detected" despite firmware virtualization being enabled | Win10 22H2 19045, Docker Desktop 29.7 | ✅ Resolved | `wsl --install --no-distribution` as admin + reboot |
| [[#ISS-003]] | `psycopg.DatabaseError: lost synchronization with server` on a healthy query | psycopg 3.3, Postgres 16 in Docker on Windows | ✅ Worked around (verified) | `PostgresWriter._exec` reconnects once and retries |
| [[#ISS-004]] | Import errors invisible locally, fail in CI | Python 3.11–3.14, all OS | ✅ Resolved | Run bare `pytest`; `pythonpath = ["src", "tests"]` |
| [[#ISS-005]] | One shared `.env` cannot load: `extra_forbidden` on another group's variable | pydantic-settings 2.x | ✅ Resolved | `extra="ignore"` + `unknown_settings()` typo check |
| [[#ISS-006]] | A recorded PID never matches `tasklist`, so a live process reads as dead | Windows PowerShell 5.1 | ✅ Worked around | Strip the BOM, or check log mtime instead |

---

## ISS-001 – respx matches a portless pattern against any port

**Reported:** 2026-08-24 · **Status:** ✅ external, workaround verified

### Symptom
`test_robots_is_fetched_from_the_right_port` asserted that a host on `:8080` has its robots.txt
fetched from `:8080` and not `:80`. It failed — but with the *wrong* failure: `allowed()` returned
`True` for a disallowed path, as though robots.txt had never parsed.

### Root cause – NOT our code
`respx.get("http://h.test/robots.txt")` (no port) matches a request to
`http://h.test:8080/robots.txt`. Routes are matched in registration order and the first match wins,
so the portless 404 route **shadowed** the ported 200 route. Confirmed directly:

```
wrong(:80) called: 1   right(:8080) called: 0
```

Protego itself handles ports correctly — `can_fetch("http://h.test:8080/private/x")` returns
`False` — so the library under test was never at fault.

### Workaround
Register the **ported** route first. There is a comment in the test saying so, because the natural
instinct is to tidy the routes into "general then specific" order, which silently re-breaks it.

### Cost
The test would have passed for the wrong reason had the code been broken in a compensating way.
This is the failure mode where a green test is worse than no test.

---

## ISS-002 – Docker Desktop reports no virtualization support, wrongly

**Reported:** 2026-08-24 · **Status:** ✅ resolved

### Symptom
Docker Desktop refuses to start: *"Docker Desktop failed to start because virtualisation support
wasn't detected. Contact your IT admin to enable virtualization."* The obvious reading is a BIOS
setting.

### Root cause – NOT the firmware
`Get-ComputerInfo` reported **all four** Hyper-V requirements as `True`:

```
HyperVRequirementDataExecutionPreventionAvailable : True
HyperVRequirementSecondLevelAddressTranslation    : True
HyperVRequirementVirtualizationFirmwareEnabled    : True
HyperVRequirementVMMonitorModeExtensions          : True
```

The firmware was fine. What was missing was the **Windows side**: `wsl.exe` was the Windows 10
placeholder stub, supporting only `--install`/`--list`/`--status` — no `--version`, no
`--set-default-version`. The `VirtualMachinePlatform` and `Microsoft-Windows-Subsystem-Linux`
optional features had never been enabled.

### Workaround
Admin PowerShell: `wsl --install --no-distribution`, then reboot. Docker Desktop creates its own
`docker-desktop` distro, so no Linux distribution is needed.

### Recognising it
If the error says virtualization but `HyperVRequirementVirtualizationFirmwareEnabled` is `True`,
do not go into the BIOS. Check `wsl --version` — a usage screen means the stub.

---

## ISS-003 – Postgres wire desync on a query that is fine

**Reported:** 2026-08-24 · **Status:** ✅ external, worked around in code

### Symptom
The archive crawl died at startup:

```
psycopg.DatabaseError: insufficient data in "D" message
lost synchronization with server: got message type "f", length 1633837366
```

Raised from `SELECT id, sha1, pdq, face_count FROM image ORDER BY id`. The obvious suspect was the
`BIT(256)` `pdq` column having no psycopg loader.

### Evidence that closed the case
1. A **second crawler started in the same second** ran the identical query against the identical
   table and succeeded.
2. The query then ran **12/12 by hand** with zero failures.

So it is neither the schema nor the type: it is a corrupted TCP stream, transient, most likely
Docker Desktop's Windows port-forwarding proxy. `1633837366` decodes to ASCII-range bytes — the
client read payload text where a message length belonged, which is the signature of a desync
rather than a server error.

### Workaround
`PostgresWriter._exec` replaces the connection once and retries. It distinguishes the two failure
kinds by asking the connection to do something trivial: if `SELECT 1` works, the error was real
(constraint violation, bad SQL) and is re-raised untouched.

### Why this mattered more than the crash
Dying loudly was the good outcome. The same blip **mid-crawl** would have left every subsequent
write failing while the loop carried on fetching pages and counting them — hours of a five-hour run
into a database that had stopped listening, with the heartbeat reporting healthy throughput.

---

## ISS-004 – `python -m pytest` hides import errors that CI catches

**Reported:** 2026-08-24 · **Status:** ✅ resolved

### Symptom
`from tests.conftest import make_image` worked locally through several full green runs, then failed
in CI on the first push with `ModuleNotFoundError: No module named 'tests'`.

### Root cause – NOT our code
`python -m pytest` prepends the current working directory to `sys.path`; a bare `pytest` does not.
CI runs bare. Every local run had been silently repairing an import that could not work anywhere
else.

Compounding it: pytest loads `conftest.py` under the bare name `conftest`, not `tests.conftest`, so
that import path was never going to be right regardless.

### Resolution
Helpers moved to `tests/imagefixtures.py` (a plain module), and `pythonpath = ["src", "tests"]` in
`pyproject.toml` so the import resolves identically under both invocations. **Run bare `pytest`
locally** — noted in `pyproject.toml`, the README, and the `/wrap-up` command.

---

## ISS-005 – pydantic-settings refuses a shared `.env`

**Reported:** 2026-08-24 · **Status:** ✅ resolved

### Symptom
`FaceSettings()` raised `ValidationError: extra_forbidden` for `arc_crawl_user_agent` — a variable
that plainly belongs to a different settings group. A `.env` containing more than one prefix could
not load **at all**, so the first real run would have died before fetching anything.

### Root cause – NOT our code
All four settings groups read the same `.env`, and pydantic-settings defaults a dotenv source to
`extra="forbid"`. Each group therefore rejected the other three groups' variables.

Invisible to the test suite, because tests construct settings with explicit kwargs and never touch
the file.

### Resolution
`extra="ignore"` on all four groups. That trades a crash for silence — a typo'd
`ARC_CRAWL_PER_HOST_RPZ` would simply do nothing — so `config.unknown_settings()` compensates by
naming any `ARC_*` variable matching no field, with `_EXTERNAL_VARS` for the ones the app
deliberately does not own (`ARC_PG_PASSWORD`, `ARC_TEST_PG_DSN`).

---

## ISS-006 – A live process reads as dead because the PID file has a BOM

**Reported:** 2026-08-24 · **Status:** ✅ worked around

### Symptom
A crawler was reported finished — no report printed, images still pending — looking exactly like a
silent crash. It was running normally the entire time.

### Root cause – NOT our code
Windows PowerShell 5.1's `Out-File -Encoding utf8` writes a **BOM**. Reading the file back in Python
yields `'﻿10608'`, which never matches `tasklist` output, so every liveness check returned
false.

### Workaround
Strip `﻿` when reading, or skip PIDs entirely and check whether the log file is still growing:

```bash
a=$(stat -c %s run.log); sleep 60; b=$(stat -c %s run.log)
```

Better still, match on the command line via `Get-CimInstance Win32_Process` rather than tracking a
PID at all — `Start-Process -PassThru` does not always report the PID you end up with.

### Cost
Diagnosed a crash that had not happened, twice. The lesson generalises: before concluding a process
died, confirm with a signal that comes from the process itself.
