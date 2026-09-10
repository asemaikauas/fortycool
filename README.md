# FortyCool

> Evidence-first thermal decision support for data-center digital twins.

[![Tests](https://github.com/asemaikauas/fortycool/actions/workflows/tests.yml/badge.svg)](https://github.com/asemaikauas/fortycool/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-fortycool.vercel.app-111827)](https://fortycool.vercel.app/)

[Launch the dashboard](https://fortycool.vercel.app/) ·
[Check API health](https://fortycool-81024b6f2f51.herokuapp.com/health) ·
[Inspect the request example](examples/demo_request.json)

FortyCool combines thermal history, land-cover context, facility telemetry, forecasting, and
financial modeling in one traceable workflow. It estimates local thermal drift, screens a safe
12-hour operating envelope, translates the result into indicative energy and NPV ranges, and keeps
every conclusion connected to its assumptions and evidence.

> [!IMPORTANT]
> The hosted demo currently uses reproducible fixture providers. Fixture outputs are simulated,
> location-independent examples—not observations about the selected facility. FortyCool is an
> advisory screening tool and cannot control physical equipment.

## What FortyCool answers

| Question | Output |
| --- | --- |
| Is the site warming faster than its controls? | Site-versus-control difference-in-differences with a 95% confidence interval |
| When may cooling changes be considered? | Signed eligibility hours and a constrained 12-hour operating envelope |
| Is the suggested action inside the training and safety envelope? | Explicit safety gates, veto reasons, or <code>hold_current_settings</code> |
| What could the opportunity be worth? | Bounded energy, cost, and NPV estimates with disclosed assumptions |
| Can a reviewer reproduce the conclusion? | Evidence IDs, agent trace, warnings, data classes, and a downloadable PDF memo |

### Product highlights

- Five-page dashboard for site setup, command-center monitoring, thermal drift, optimization, and
  run history.
- Deterministic specialist agents for planning, temperature, asset modeling, decisions,
  investment, and audit.
- Live server-sent events while an analysis runs.
- Validated CSV telemetry uploads, with simulated BMS data used only when no upload is selected.
- FortyGuard thermal and satellite integrations plus Google Dynamic World land-cover history.
- A two-stage public data-center discovery workflow with an explicit
  <code>no_qualified_candidate</code> outcome.
- Persisted runs, immutable evidence endpoints, and evidence-linked investment memos.
- Optional GPT-4o copilot grounded only in a completed run.
- Typed Pydantic contracts, request limits, API-key protection, and CI across Python 3.11–3.13.

## How it works

~~~mermaid
flowchart LR
    A[Site, constraints, and analysis window] --> P[Thermal and land-cover providers]
    T[Optional facility telemetry CSV] --> O[Orchestrator]
    P --> O
    O --> S[Specialist agents and deterministic models]
    S --> G{Safety and evidence gates}
    G -->|pass| R[Recommendation and investment range]
    G -->|insufficient or unsafe| H[Hold current settings]
    R --> E[Dashboard, evidence, trace, PDF, and copilot]
    H --> E
~~~

The LLM boundary is deliberately narrow: a model may plan, choose tools, and explain results, while
calculations and safety decisions remain in deterministic code. The direct agent-tool routes force
the required specialist and safety stages so a planner cannot silently skip them.

## Quick start

### Requirements

- Python 3.11 or newer
- Git

Fixture mode is the default and needs no provider credentials.

~~~bash
git clone https://github.com/asemaikauas/fortycool.git
cd fortycool

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

uvicorn fortycool_agents.api:app --reload --port 8000
~~~

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API redirects to Site Setup, and the
dashboard is served by the same process.

Run the example from the CLI:

~~~bash
fortycool --request examples/demo_request.json
~~~

Or call the synchronous analysis endpoint:

~~~bash
curl -sS http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  --data-binary @examples/demo_request.json
~~~

Enable the local OpenAPI consoles only when needed:

~~~bash
FORTYCOOL_ENABLE_DOCS=1 uvicorn fortycool_agents.api:app --reload --port 8000
~~~

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Dashboard workflow

1. **Site Setup** — choose a facility, define the AOI and constraints, select an analysis window,
   or upload facility telemetry.
2. **Command Center** — submit the full workflow and follow named agent events as they arrive.
3. **Thermal Drift** — inspect annual site/control series, the confidence interval, land-cover
   context, and linked evidence.
4. **Optimization** — review the 12-hour operating envelope, safety verdicts, projected energy
   impact, and sensitivity bounds.
5. **History** — reopen persisted runs, verify evidence status, or download the investment memo.

The dashboard resolves metrics, charts, recommendations, assumptions, warnings, and evidence by
stable IDs from the response contract. It never infers verification from a client-side URL flag.

## Evidence contract

Every metric carries a <code>data_class</code> and an <code>evidence_grade</code>.

| Data class | Meaning |
| --- | --- |
| <code>observed</code> | Returned by a real source such as FortyGuard |
| <code>uploaded</code> | Supplied by a user and accepted by the telemetry validator |
| <code>inferred</code> | Calculated from other records or partially filled |
| <code>simulated</code> | Generated by the fixture provider or digital twin |
| <code>assumed</code> | A demo or financial input not confirmed by an operator |

| Grade | Meaning |
| --- | --- |
| A | Observed evidence |
| B | Derived evidence |
| C | Simulated or unbounded evidence |

Numeric <code>confidence</code> values are internal fit scores, not probabilities that a financial
estimate is correct.

### Verified demo gate

Site Setup can load a persisted verified ThermalDrift run only when the backend confirms all of the
following:

- At least three consecutive years of observed FortyGuard history.
- No calibrated thermal backcasts.
- Observed Google Dynamic World history with complete selected-control coverage.
- Accepted regional-control similarity, with those controls used in the thermal history.
- Drift and local-buildout metrics linked to the qualifying provider evidence.

Verification applies to those ThermalDrift inputs. The run may still label its BMS, forecast, or
optimization panels as simulated or inferred. The server exposes the result at
<code>GET /runs/{run_id}/verification</code>.

## Providers and run modes

| Capability | Mode | Configuration | What it provides |
| --- | --- | --- | --- |
| Thermal | <code>fixture</code> (default) | <code>FORTYCOOL_THERMAL_PROVIDER=fixture</code> | Fixed, reproducible synthetic history and forecast |
| Thermal | <code>live</code> | <code>FORTYGUARD_API_KEY</code> and <code>FORTYCOOL_THERMAL_PROVIDER=live</code> | FortyGuard historical temperature, forecast heatmaps, and activity evidence |
| Land cover | <code>fixture</code> (default) | <code>FORTYCOOL_URBAN_PROVIDER=fixture</code> | Reproducible synthetic land-cover series |
| Land cover | <code>live</code> | <code>FORTYCOOL_URBAN_PROVIDER=live</code> | FortyGuard Premium Satellite Segmentation |
| Land cover | <code>dynamic_world</code> | Earth Engine credentials, <code>EARTH_ENGINE_PROJECT</code>, and <code>FORTYCOOL_URBAN_PROVIDER=dynamic_world</code> | Annual Google Dynamic World history |

Live thermal mode is intentionally hybrid. FortyGuard supplies spatial temperatures and forecasts,
but BMS and environmental fields remain simulated unless the operator uploads facility telemetry.

Start a live thermal session:

~~~bash
export FORTYGUARD_API_KEY=replace-with-your-key
export FORTYCOOL_THERMAL_PROVIDER=live
uvicorn fortycool_agents.api:app --reload --port 8000
~~~

For Dynamic World, first authenticate Earth Engine and select the registered Cloud project:

~~~bash
earthengine authenticate
export EARTH_ENGINE_PROJECT=your-google-cloud-project
export FORTYCOOL_URBAN_PROVIDER=dynamic_world
~~~

Application Default Credentials and <code>GOOGLE_APPLICATION_CREDENTIALS</code> are also supported.
Never commit provider keys or service-account credentials.

## Upload facility telemetry

Send CSV as a raw request body:

~~~bash
curl -X POST http://127.0.0.1:8000/telemetry/uploads \
  -H 'Content-Type: text/csv' \
  --data-binary @facility.csv
~~~

Required columns:

~~~text
timestamp
it_load_kw
cooling_power_kw
server_inlet_temperature_c
~~~

Recommended control columns:

~~~text
supply_air_setpoint_c
chilled_water_supply_c
fan_speed_percent
economizer_state
~~~

The validator checks cadence and plausible power/temperature values, strips unknown columns, and
normalizes the input to the 336 hourly rows used by the model. Gaps of at most two hours may be
interpolated; filled data is reported in warnings and downgraded from <code>uploaded</code> to
<code>inferred</code>. If the history lacks enough control variation, the safety agent returns
<code>hold_current_settings</code>.

## API

### Analysis and agent tools

| Method | Route | Purpose |
| --- | --- | --- |
| POST | <code>/runs</code> | Run the complete workflow synchronously |
| POST | <code>/run-jobs</code> | Start a background run and return <code>202 Accepted</code> |
| GET | <code>/run-jobs/{run_id}</code> | Read job status |
| GET | <code>/run-jobs/{run_id}/stream</code> | Stream <code>trace</code> and <code>terminal</code> SSE events |
| POST | <code>/agent-tools/thermal-drift</code> | Run the thermal-drift workflow |
| POST | <code>/agent-tools/operations-12h</code> | Run the 12-hour operations workflow |
| POST | <code>/agent-tools/investment</code> | Run thermal drift plus investment analysis |
| POST | <code>/agent-tools/site-discovery</code> | Screen and deeply validate public or supplied candidates |

### Results and evidence

| Method | Route | Purpose |
| --- | --- | --- |
| GET | <code>/runs/{run_id}</code> | Read a persisted completed run |
| GET | <code>/runs/{run_id}/verification</code> | Evaluate that run against verification gates |
| GET | <code>/runs/{run_id}/events</code> | Read its agent trace |
| GET | <code>/runs/{run_id}/evidence/{evidence_id}</code> | Resolve one immutable evidence record |
| GET | <code>/runs/{run_id}/memo.pdf</code> | Download the evidence-linked investment memo |
| POST | <code>/runs/{run_id}/copilot</code> | Ask a grounded question about the completed run |
| GET | <code>/demo/verified-run</code> | Load the latest persisted run that passes verification |

### Contracts and support

| Method | Route | Purpose |
| --- | --- | --- |
| GET | <code>/health</code> | Provider, database, authentication, and copilot status |
| GET | <code>/tools</code> | Machine-readable agent-tool catalog |
| GET | <code>/schemas/{contract}</code> | Pydantic JSON schema for analysis, discovery, or copilot |
| GET | <code>/discovery/catalog</code> | Public default candidate catalog |
| POST | <code>/telemetry/uploads</code> | Validate and store a telemetry CSV |
| GET/DELETE | <code>/telemetry/uploads/{upload_id}</code> | Inspect or remove an upload |

Stable analysis response fields:

~~~text
run_id
status
confidence_tier
metrics
recommendations
charts
evidence
trace
assumptions
warnings
~~~

## Grounded copilot

Set <code>OPENAI_API_KEY</code> and optionally <code>FORTYCOOL_COPILOT_MODEL</code> (default:
<code>gpt-4o</code>). The server uses the Responses API with Structured Outputs and
<code>store: false</code>.

~~~bash
curl -sS -X POST "http://127.0.0.1:8000/runs/$RUN_ID/copilot" \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Explain the recommendation and its main risks.",
    "audience": "operator"
  }'
~~~

Supported audiences are <code>operator</code>, <code>investment_committee</code>, and
<code>technical_reviewer</code>. The model receives compact metrics, recommendations, warnings,
assumptions, chart summaries, trace events, and an evidence index—not raw GeoJSON. It cannot alter
deterministic calculations, safety verdicts, or equipment.

## Persistence

- Local runs use SQLite at <code>.fortycool-data/runs.sqlite3</code> by default.
- Set <code>FORTYCOOL_DB_PATH</code> to move the local database.
- A configured <code>DATABASE_URL</code> switches the service to PostgreSQL.
- <code>FORTYCOOL_MAX_RETAINED_RUNS</code> controls completed-run retention (default: 2,000).

Runs and normalized uploads are database-backed. In-progress job coordination remains best suited to
one application worker on the current deployment architecture.

## Configuration and security

Copy [.env.example](.env.example) as a reference. The service does not automatically require a
secret for local fixture use.

| Variable | Default | Purpose |
| --- | --- | --- |
| <code>FORTYCOOL_API_KEY</code> | unset | Require a matching <code>X-API-Key</code> on protected routes |
| <code>FORTYCOOL_CORS_ORIGINS</code> | unset | Comma-separated browser origins allowed to call the API |
| <code>FORTYCOOL_ENABLE_DOCS</code> | unset | Enable <code>/docs</code>, <code>/redoc</code>, and <code>/openapi.json</code> |
| <code>FORTYCOOL_MAX_REQUEST_BYTES</code> | 12 MiB | Reject oversized bodies before buffering |
| <code>FORTYCOOL_TRUSTED_PROXY_HOPS</code> | 0 | Number of trusted proxies used to resolve caller IPs |
| <code>FORTYCOOL_COPILOT_MAX_CALLS_PER_RUN</code> | 25 | Bound model calls for one run |

Per-caller and global rate limits are enabled even without authentication. The global ceilings are
the protection against callers rotating identity headers; set <code>FORTYCOOL_TRUSTED_PROXY_HOPS</code>
to the actual number of proxies rather than trusting <code>X-Forwarded-For</code> unconditionally.

<details>
<summary>Rate-limit defaults</summary>

| Traffic class | Per caller / minute | Global / minute |
| --- | ---: | ---: |
| Analysis | 20 | 60 |
| Copilot | 10 | 30 |
| Telemetry writes | 10 | 30 |
| Reads | 240 | 1,200 |

Change the window with <code>FORTYCOOL_RATE_WINDOW_SECONDS</code> and the limits with the
<code>FORTYCOOL_RATE_LIMIT_*</code> and <code>FORTYCOOL_GLOBAL_LIMIT_*</code> variables.

</details>

Every response carries a content-security policy, clickjacking protection, MIME sniffing protection,
a no-referrer policy, and a restrictive permissions policy.

> [!WARNING]
> Do not expose <code>FORTYGUARD_API_KEY</code> or <code>OPENAI_API_KEY</code> on a public service
> without also protecting the API. The current browser dashboard does not send
> <code>X-API-Key</code>, so enabling API-key authentication intentionally disables dashboard API
> calls until client-side authentication support is added.

Anonymous discovery is capped at four candidates and a shortlist of one because a full request may
fan out into many paid provider activities.

## Deployment

The public demo uses a split deployment:

~~~text
Vercel static dashboard
        |
        v
Heroku FastAPI service
        |
        v
Heroku Postgres
~~~

Vercel publishes <code>src/fortycool_agents/web</code>. Its
<code>api-config.js</code> points browser requests at the Heroku service, and the backend CORS
allowlist permits the Vercel origin. The checked-in [Procfile](Procfile), [vercel.json](vercel.json),
[render.yaml](render.yaml), and [Dockerfile](Dockerfile) cover the current Heroku/Vercel deployment,
Render, and container platforms.

<details>
<summary>Heroku setup</summary>

~~~bash
heroku login
heroku git:remote -a fortycool
heroku addons:create heroku-postgresql:essential-0 -a fortycool
heroku config:set \
  FORTYCOOL_CORS_ORIGINS=https://fortycool.vercel.app \
  FORTYCOOL_TRUSTED_PROXY_HOPS=1 \
  FORTYCOOL_THERMAL_PROVIDER=fixture \
  FORTYCOOL_URBAN_PROVIDER=fixture \
  -a fortycool
git push heroku main
heroku ps:resize web=basic -a fortycool
heroku ps:scale web=1 -a fortycool
~~~

</details>

Use one worker for the current low-cost deployment. On a host without <code>DATABASE_URL</code> or a
persistent volume, the SQLite store will reset when the instance is replaced.

## Modeling and interpretation limits

These limits are part of the product contract, not footnotes:

| Limitation | How FortyCool handles it |
| --- | --- |
| Fixture history is identical for every coordinate | Labels all outputs simulated and emits <code>fixture_series_location_independent</code> |
| A missing live year may require a calibrated fixture backcast | Labels the series inferred and withholds thermal drift and NPV whenever a non-simulated series contains a backcast |
| Historical eligibility uses matched seasonal screening | Reports annualized screening hours; does not call them utility-grade full-year accounting |
| A local outer control ring may sit inside the facility plume | Names it <code>local outer ring</code>, not a regional matched control |
| Controls are matched on latest land cover | Discloses possible post-treatment bias toward zero; baseline matching is not implemented |
| Satellite segmentation may return the same image year for historical requests | Withholds change when distinct dated imagery is unavailable |
| Discovery selects the largest result from several candidates | Treats the shortlist as investigation targets, not proof that the leader is warming |
| Control movement may be confounded by load and weather | Treats action confidence as model fit, not causal identification |
| Sparse or out-of-envelope telemetry cannot support a safe action | Returns <code>hold_current_settings</code> |

Land-cover correlation is published only with at least eight matched years of genuinely observed
annual history. It includes its sample size and Fisher 95% interval and is explicitly an attribution
hypothesis—not a causal claim.

## Development

Run the test suite:

~~~bash
python -m pytest -q
~~~

CI runs tests on Python 3.11, 3.12, and 3.13, audits dependencies, verifies vendored dashboard
assets, and prevents pages from drifting back to CDN dependencies.

For a hash-pinned install on the platform used to generate the lock:

~~~bash
python -m pip install --require-hashes -r requirements.lock
~~~

Regenerate the lock with <code>scripts/write_lock.py</code> after dependency changes. Because NumPy,
pandas, and ReportLab ship platform-specific wheels, regenerate the lock for a different OS,
architecture, or Python version.

### Repository map

~~~text
src/fortycool_agents/
├── api.py                 FastAPI application and routes
├── orchestrator.py        End-to-end workflow coordination
├── agents.py              Specialist agents and safety gates
├── models.py              Typed request and response contracts
├── modeling.py            Forecasting and evaluation
├── telemetry.py           CSV validation and normalization
├── discovery.py           Two-stage candidate screening
├── providers/             Fixture, FortyGuard, and Dynamic World adapters
├── copilot.py             Grounded completed-run copilot
├── memo.py                Deterministic PDF report generation
├── database.py            SQLite/PostgreSQL persistence
└── web/                   Static dashboard and vendored assets

tests/                     API, provider, storage, hardening, and workflow tests
examples/demo_request.json Reproducible fixture request
~~~

## Project status

FortyCool is currently version 0.1.0 and should be treated as a prototype for thermal, operational,
and investment screening. It is not an engineering design, a utility-grade measurement system, a
valuation opinion, or an autonomous facility controller.
