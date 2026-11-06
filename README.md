# Learned distractor suppression (HPL reversal)

PsychoPy additional-singleton visual search experiment for studying how learned distractor suppression updates when the high-probability distractor location (HPL) reverses.

Designed for eventual simultaneous EEG + eye tracking: accurate display timing, event markers, continuous gaze logging, and reproducible pre-generated trial schedules.

## Scientific design (do not change casually)

- 8 equidistant locations (indices 0–7; 0 = top).
- Target = diamond among circles; report internal line orientation (`F` = horizontal, `J` = vertical).
- Color singleton distractor on ~80% of trials; never overlaps the target.
- **Acquisition** concentrates distractors at one HPL (~70% of distractor-present trials); **reversal** (or stable control) follows.
- No informative pre-cue; fixation only before array onset (critical for −800–0 ms analyses).
- Participants are **not** told about HPL, statistical learning, or reversal.

### Methodological note: target–distractor exclusion vs 70% HPL

Target location is assigned first (balanced across positions). The distractor is then placed among remaining locations. Because the distractor cannot occupy the target, a naïve 70% HPL rule would yield ~61% effective HPL. This package uses **exact HPL quotas** among eligible trials (target ≠ HPL) so that realized `P(distractor at HPL | distractor present) ≈ configured 70%` while target locations stay uniform. See `experiment/trial_generator.py` (`conditional_hpl_probability`, quota filling).

## Setup

PsychoPy does **not** support Python 3.13/3.14. Use **Python 3.12** (already on this machine via Homebrew):

```bash
cd distractor-hpl-reversal
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then always activate the venv before running:

```bash
cd distractor-hpl-reversal
source .venv/bin/activate
python main.py --participant 99 --session trial
```

Headless simulation (no PsychoPy window; works on system Python too):

```bash
python main.py --simulate --participant 1 --session 1
```


## Tests

```bash
python3 -m pytest tests/ -q
```

## Outputs

Each session writes under `data/{participant}_{session}_{timestamp}/`:

| File | Contents |
|------|----------|
| `trials.csv` | One row per trial (RT, accuracy, HPL flags, `trial_relative_to_reversal`, …) |
| `events.csv` | Timestamped markers (aligned with EEG codes) |
| `gaze.csv` | Continuous samples (separate from trial rows) |
| `metadata.json` | IDs, versions, monitor, seed, full config |
| `schedules/block_XX.json` | Pre-generated block schedules |

### Event codes

| Code | Event |
|------|-------|
| S01 / S02 | Experiment start / end |
| S10 / S11 | Block start / end |
| S20 | Fixation onset |
| S30 / S31 / S32 | Array onset: absent / HPL / low-prob |
| S40 / S41 / S42 | Correct / incorrect / timeout |
| S50 / S51 | Acquisition / reversal phase |

## Offline QC

```bash
python3 analysis/behavioral_qc.py data/.../trials.csv
```

## Architecture

- `experiment/` — task loop, stimuli (deg), timing, trial generator, logging, event codes
- `hardware/` — `EyeTracker` + `EventMarker` ABCs; `MockTracker`, `MockMarkers`, optional `LSLMarkers`
- Real EyeLink/Tobii adapters and gaze-contingent controllers are intentionally **not** in v1

## Counterbalancing

Odd participant IDs (numeric): LEFT→RIGHT in reversal mode (LEFT throughout if stable).  
Even IDs: RIGHT→LEFT (RIGHT throughout if stable).  
Default indices: LEFT=`6`, RIGHT=`2` (`config.yaml` → `spatial`).

## License / citation

Research software — cite your protocol paper when published.
