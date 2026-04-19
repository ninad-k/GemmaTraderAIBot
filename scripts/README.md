# scripts/

Standalone one-off utilities. These are **not** part of the trading pipeline
and **not** part of the automated test suite (`pytest.ini` limits pytest to
`tests/`).

Run from the repo root so local imports resolve:

```bash
python scripts/create_doc.py        # regenerate the .docx user guide
python scripts/test_gemma.py        # dev harness: send a fake alert to server.py
python scripts/test_lots.py         # MT5-only: list crypto symbols on your broker
```

- `create_doc.py` depends on `python-docx` (already in `requirements.txt`).
- `test_gemma.py` expects `server.py` running on `localhost:5000`.
- `test_lots.py` is Windows + MT5 terminal only.
