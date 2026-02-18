# Web GUI Rollout Plan and Execution

## Goal
Deliver a web app with CLI-equivalent functionality, including drag-and-drop ingest with duplicate detection.

## Stage 1: API foundation
- Build FastAPI app bound to existing service layer.
- Ensure project init and dependency wiring are centralized.
- Add health-safe JSON serialization for slot dataclasses.

Status: Complete.
Validation:
- API smoke tests via FastAPI TestClient.
- `pytest` passing.

## Stage 2: Full operations API parity with CLI
- Add endpoints for all CLI domains:
  - init/ingest/resources
  - refs/citations
  - extract
  - claims/claim sets
  - bind/validate
  - verify/run set
  - report
  - trace
  - doctor
  - ceapf
  - financial pipeline batch pass

Status: Complete.
Validation:
- Endpoint exercise in unit smoke test.
- Live server curl checks.

## Stage 3: Browser UI shell
- Add static web UI page with sectioned operations.
- Add API response panels with JSON feedback.

Status: Complete.
Validation:
- Page served successfully at `/`.

## Stage 4: Drag-and-drop ingest UX
- Implement drop zone with multipart upload to `/api/ingest/upload`.
- Return dedupe-aware result (`ingested` vs `duplicate`).

Status: Complete.
Validation:
- Upload endpoint tested in `tests/unit/test_web_app.py`.

## Stage 5: Operational controls and production readiness
- Add CLI `stemma web` command for hosting GUI.
- Add resumable financial batch pipeline command and state/log checkpoints.
- Add per-file extraction timeout option in pipeline command/API.

Status: Complete.
Validation:
- `stemma --help` includes `web` and `pipeline`.
- Financial pipeline run logs and checkpoint files updated under `.stemma/`.

## Stage 6: Full verification pass
- Run full test suite and compile checks.
- Run live web-server smoke against API routes.

Status: Complete.
Validation:
- `pytest`: passing.
- `python -m compileall src`: passing.
- live `curl` checks: passing.
