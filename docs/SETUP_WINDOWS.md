# Windows setup

The completed research measurements were made on Linux CPU. These commands provide a Windows setup route, but Windows execution has not been tested in this workspace.

Install Python 3.12, extract the project archive, open PowerShell in the extracted project folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/new_run.py --config configs/smoke.yaml --name smoke_local
.\.venv\Scripts\python.exe -m neuroedge prepare --config configs/smoke_local.yaml
.\.venv\Scripts\python.exe -m neuroedge run --config configs/smoke_local.yaml
.\.venv\Scripts\python.exe -m neuroedge report --config configs/smoke_local.yaml
.\.venv\Scripts\python.exe -m neuroedge verify --config configs/smoke_local.yaml
```

No environment activation or PowerShell execution-policy change is needed with these explicit Python paths. The source archive includes measured model and prediction artifacts, but excludes the raw EDF recordings and large processed array cache; `prepare` reacquires the official data subset. An internet connection to the documented dataset hosts and Python package indexes is required.

For the main study create `main_local.yaml` from `main.yaml` with the same helper and use that new configuration consistently. The included `runs/main` and `runs/smoke` folders are archived Linux evidence. Fresh local outputs avoid overwriting them or mixing platform-dependent CSV bytes and floating-point results with a completed study. See the main README for deployment and replay commands. Read the report and keep the frozen split protocol before trying any new model configuration.

