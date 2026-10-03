# Git workflow demonstration

Branches: `main` (production, tagged releases) and `develop` (integration). Work happens on `feature/*` branches.
Below is the real terminal transcript of the session that produced this repo's history
(commands: init, add, commit, status, diff, push, pull, fetch, merge, conflict resolution, tag).
The remote was a local bare repository; to publish to GitHub/GitLab see the end of this file.

```text
######## 1. Create remote, push main and develop
$ git init -q --bare -b main /home/claude/remote.git
$ git remote add origin /home/claude/remote.git
$ git push -q -u origin main develop
fatal: expected 'acknowledgments', received 'packfile'
warning: push negotiation failed; proceeding anyway with push
$ git branch -vv
* develop 2993967 [origin/develop] ci: GitHub Actions workflow (test -> build -> deploy per branch)
  main    3cfd981 [origin/main] Initial commit: ViT + LoRA + INT8 + FastAPI + Docker project
######## 2. Feature branch from develop: edit, status, diff, add, commit
$ git checkout -q -b feature/docs-monitoring develop
$ git status -sb
## feature/docs-monitoring
 M README.md
$ git diff
diff --git a/README.md b/README.md
index 215d767..9b65994 100644
--- a/README.md
+++ b/README.md
@@ -61,4 +61,4 @@ Example response:
 - Quantization is PyTorch **dynamic INT8** on all `nn.Linear` layers (~4x smaller, CPU only).
   Use the same `torch` version for quantizing and serving.
 - Merging LoRA before quantizing means the container needs no `peft` dependency.
-- Env vars: `MODEL_DIR`, `MAX_UPLOAD_MB`, `TORCH_THREADS`.
+- Env vars: `MODEL_DIR`, `MAX_UPLOAD_MB`, `TORCH_THREADS`, `MONITOR_LOG_DIR`, `BASELINE_PATH`.
$ git add README.md
$ git status -sb
## feature/docs-monitoring
M  README.md
$ git commit -q -m docs:\ document\ monitoring\ environment\ variables
$ git log --oneline -2
630933c docs: document monitoring environment variables
2993967 ci: GitHub Actions workflow (test -> build -> deploy per branch)
######## 3. Meanwhile on develop someone edits the SAME README line
$ git checkout -q develop
$ git commit -q -a -m docs:\ list\ defaults\ of\ environment\ variables
######## 4. Teammate pushes to origin/develop; we have a local commit too -> git pull creates a merge
$ git clone -q -b develop /home/claude/remote.git /home/claude/teammate
(teammate pushed CONTRIBUTING.md to origin/develop)
$ git fetch origin
From /home/claude/remote
   2993967..e0d0873  develop    -> origin/develop
$ git status -sb
## develop...origin/develop [ahead 1, behind 1]
$ git pull --no-edit origin develop
From /home/claude/remote
 * branch            develop    -> FETCH_HEAD
Merge made by the 'ort' strategy.
 CONTRIBUTING.md | 5 +++++
 1 file changed, 5 insertions(+)
 create mode 100644 CONTRIBUTING.md
$ git push -q origin develop
$ git log --oneline --graph --all -8
*   6f1de9b Merge branch 'develop' of /home/claude/remote into develop
|\  
| * e0d0873 docs: add CONTRIBUTING guide
* | 64b4f51 docs: list defaults of environment variables
|/  
| * 630933c docs: document monitoring environment variables
|/  
* 2993967 ci: GitHub Actions workflow (test -> build -> deploy per branch)
* f42a94b feat(infra): Prometheus, Grafana dashboard, cAdvisor/node-exporter and alert rules
* b412f13 feat(api): /metrics, /feedback, /monitoring/drift and prediction logging
* d2c7ce2 feat(monitoring): drift (PSI/KS), performance and calibration metrics + offline CLI
######## 5. Merge feature branch into develop -> CONFLICT
$ git merge --no-ff -m Merge\ feature/docs-monitoring\ into\ develop feature/docs-monitoring
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit the result.
$ git status
On branch develop
Your branch is up to date with 'origin/develop'.
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)
Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   README.md
no changes added to commit (use "git add" and/or "git commit -a")
$ git diff --name-only --diff-filter=U
README.md
$ git diff
diff --cc README.md
index b7130f7,9b65994..0000000
--- a/README.md
+++ b/README.md
@@@ -61,4 -61,4 +61,8 @@@ Example response
  - Quantization is PyTorch **dynamic INT8** on all `nn.Linear` layers (~4x smaller, CPU only).
    Use the same `torch` version for quantizing and serving.
  - Merging LoRA before quantizing means the container needs no `peft` dependency.
++<<<<<<< HEAD
 +- Env vars: `MODEL_DIR` (default `models/quantized`), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (all cores).
++=======
+ - Env vars: `MODEL_DIR`, `MAX_UPLOAD_MB`, `TORCH_THREADS`, `MONITOR_LOG_DIR`, `BASELINE_PATH`.
++>>>>>>> feature/docs-monitoring
######## 6. Resolve conflict: keep both sides' intent in one line
$ grep -c \<\<\<\<\<\<\<\\\|\>\>\>\>\>\>\> README.md
0
$ tail -2 README.md
- Merging LoRA before quantizing means the container needs no `peft` dependency.
- Env vars: `MODEL_DIR` (default `models/quantized`), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (all cores), `MONITOR_LOG_DIR` (`logs`), `BASELINE_PATH` (`$MODEL_DIR/reference_stats.json`).
$ git add README.md
$ git status -sb
## develop...origin/develop
M  README.md
$ git commit -q --no-edit
$ git log --oneline --graph -7
*   3a94cf7 Merge feature/docs-monitoring into develop
|\  
| * 630933c docs: document monitoring environment variables
* |   6f1de9b Merge branch 'develop' of /home/claude/remote into develop
|\ \  
| * | e0d0873 docs: add CONTRIBUTING guide
| |/  
* / 64b4f51 docs: list defaults of environment variables
|/  
* 2993967 ci: GitHub Actions workflow (test -> build -> deploy per branch)
* f42a94b feat(infra): Prometheus, Grafana dashboard, cAdvisor/node-exporter and alert rules
$ git branch -d feature/docs-monitoring
Deleted branch feature/docs-monitoring (was 630933c).
######## 7. Diff between the two branches (main vs develop)
$ git diff --stat main develop
 .github/workflows/ci.yml                           | 169 ++++++++++
 CONTRIBUTING.md                                    |   5 +
 Dockerfile                                         |   2 +-
 README.md                                          |   2 +-
 docker-compose.yml                                 |  37 +++
 monitoring/alerts.yml                              |  50 +++
 monitoring/grafana/dashboards/vit.json             | 348 +++++++++++++++++++++
 monitoring/grafana/provisioning/dashboards/d.yml   |   3 +
 .../grafana/provisioning/datasources/prom.yml      |   3 +
 monitoring/prometheus.yml                          |  12 +
 requirements-dev.txt                               |   3 +
 requirements-serve.txt                             |   1 +
 src/app.py                                         |  79 ++++-
 src/dev_model.py                                   |  30 ++
 src/metrics.py                                     |  15 +
 src/monitor_cli.py                                 | 100 ++++++
 src/monitoring.py                                  | 204 ++++++++++++
 tests/test_api.py                                  |  72 +++++
 tests/test_infra.py                                |  26 ++
 tests/test_monitoring.py                           |  71 +++++
 20 files changed, 1227 insertions(+), 5 deletions(-)
$ git diff main develop -- Dockerfile README.md
diff --git a/Dockerfile b/Dockerfile
index c80b9ef..9cc6679 100644
--- a/Dockerfile
+++ b/Dockerfile
@@ -15,7 +15,7 @@ RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch
 COPY src/ ./src/
 COPY models/quantized ./models/quantized
 
-RUN useradd -m appuser && chown -R appuser /app
+RUN useradd -m appuser && mkdir -p /app/logs && chown -R appuser /app
 USER appuser
 
 EXPOSE 8000
diff --git a/README.md b/README.md
index 215d767..5a69bcb 100644
--- a/README.md
+++ b/README.md
@@ -61,4 +61,4 @@ Example response:
 - Quantization is PyTorch **dynamic INT8** on all `nn.Linear` layers (~4x smaller, CPU only).
   Use the same `torch` version for quantizing and serving.
 - Merging LoRA before quantizing means the container needs no `peft` dependency.
-- Env vars: `MODEL_DIR`, `MAX_UPLOAD_MB`, `TORCH_THREADS`.
+- Env vars: `MODEL_DIR` (default `models/quantized`), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (all cores), `MONITOR_LOG_DIR` (`logs`), `BASELINE_PATH` (`$MODEL_DIR/reference_stats.json`).
$ git log --oneline main..develop
3a94cf7 Merge feature/docs-monitoring into develop
6f1de9b Merge branch 'develop' of /home/claude/remote into develop
64b4f51 docs: list defaults of environment variables
e0d0873 docs: add CONTRIBUTING guide
630933c docs: document monitoring environment variables
2993967 ci: GitHub Actions workflow (test -> build -> deploy per branch)
f42a94b feat(infra): Prometheus, Grafana dashboard, cAdvisor/node-exporter and alert rules
b412f13 feat(api): /metrics, /feedback, /monitoring/drift and prediction logging
d2c7ce2 feat(monitoring): drift (PSI/KS), performance and calibration metrics + offline CLI
$ git add README.md
$ git commit -q -m docs:\ README\ sections\ for\ tests\,\ CI/CD\ and\ observability
$ python3 -m pytest -q tests
../../../usr/local/lib/python3.13/dist-packages/starlette/testclient.py:53
tests/test_api.py::test_health_and_labels
tests/test_api.py::test_health_and_labels
######## 8. Release: merge develop into main, tag, push
$ git checkout -q main
$ git merge --no-ff -m Release\ v1.0.0:\ monitoring\ +\ CI/CD develop
Merge made by the 'ort' strategy.
 .github/workflows/ci.yml                           | 169 ++++++++++
 CONTRIBUTING.md                                    |   5 +
 Dockerfile                                         |   2 +-
 README.md                                          |  36 ++-
 docker-compose.yml                                 |  37 +++
 monitoring/alerts.yml                              |  50 +++
 monitoring/grafana/dashboards/vit.json             | 348 +++++++++++++++++++++
 monitoring/grafana/provisioning/dashboards/d.yml   |   3 +
 .../grafana/provisioning/datasources/prom.yml      |   3 +
 monitoring/prometheus.yml                          |  12 +
 requirements-dev.txt                               |   3 +
 requirements-serve.txt                             |   1 +
 src/app.py                                         |  79 ++++-
 src/dev_model.py                                   |  30 ++
 src/metrics.py                                     |  15 +
 src/monitor_cli.py                                 | 100 ++++++
 src/monitoring.py                                  | 204 ++++++++++++
 tests/test_api.py                                  |  72 +++++
 tests/test_infra.py                                |  26 ++
 tests/test_monitoring.py                           |  71 +++++
 20 files changed, 1261 insertions(+), 5 deletions(-)
 create mode 100644 .github/workflows/ci.yml
 create mode 100644 CONTRIBUTING.md
 create mode 100644 docker-compose.yml
 create mode 100644 monitoring/alerts.yml
 create mode 100644 monitoring/grafana/dashboards/vit.json
 create mode 100644 monitoring/grafana/provisioning/dashboards/d.yml
 create mode 100644 monitoring/grafana/provisioning/datasources/prom.yml
 create mode 100644 monitoring/prometheus.yml
 create mode 100644 requirements-dev.txt
 create mode 100644 src/dev_model.py
 create mode 100644 src/metrics.py
 create mode 100644 src/monitor_cli.py
 create mode 100644 src/monitoring.py
 create mode 100644 tests/test_api.py
 create mode 100644 tests/test_infra.py
 create mode 100644 tests/test_monitoring.py
$ git tag -a v1.0.0 -m v1.0.0
$ git push -q origin main develop --follow-tags
$ git branch -a -vv
  develop                38ddbed [origin/develop] docs: README sections for tests, CI/CD and observability
* main                   7c8f194 [origin/main] Release v1.0.0: monitoring + CI/CD
  remotes/origin/develop 38ddbed docs: README sections for tests, CI/CD and observability
  remotes/origin/main    7c8f194 Release v1.0.0: monitoring + CI/CD
$ git log --oneline --graph --all -12
*   7c8f194 Release v1.0.0: monitoring + CI/CD
|\  
| * 38ddbed docs: README sections for tests, CI/CD and observability
| *   3a94cf7 Merge feature/docs-monitoring into develop
| |\  
| | * 630933c docs: document monitoring environment variables
| * |   6f1de9b Merge branch 'develop' of /home/claude/remote into develop
| |\ \  
| | * | e0d0873 docs: add CONTRIBUTING guide
| | |/  
| * / 64b4f51 docs: list defaults of environment variables
| |/  
| * 2993967 ci: GitHub Actions workflow (test -> build -> deploy per branch)
| * f42a94b feat(infra): Prometheus, Grafana dashboard, cAdvisor/node-exporter and alert rules
| * b412f13 feat(api): /metrics, /feedback, /monitoring/drift and prediction logging
| * d2c7ce2 feat(monitoring): drift (PSI/KS), performance and calibration metrics + offline CLI
|/  
* 3cfd981 Initial commit: ViT + LoRA + INT8 + FastAPI + Docker project
```

## Publish to GitHub (or GitLab)
```bash
# create an EMPTY repo on github.com first, then:
git remote set-url origin https://github.com/<user>/<repo>.git   # or: git remote add origin ...
git push -u origin main develop --follow-tags
```
Then in GitHub: Settings -> Environments -> create `staging` and `production` (add *Required reviewers* to production),
and Settings -> Branches -> protect `main`/`develop` (require the `test` check). The workflow in
`.github/workflows/ci.yml` starts running on the first push.
