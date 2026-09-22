# Data layout

This file documents the *expected shape* of the external data. It contains no
real data: no session IDs beyond the ones already discussed in project notes, no
durations, no transcript text.

Nothing described here lives inside the repository. All three roots are outside
it and are configured through the environment or a gitignored `.env`.

## Roots

| Variable | Contents | Written by |
| --- | --- | --- |
| `VC_DATA_ROOT` | Raw recordings. Treated as read-only. | nobody; supplied by the lab |
| `VC_WORK_ROOT` | Intermediates: extracted audio, diarization segments, transcripts, per-frame data, reference clips, downloaded models. | most stages |
| `VC_OUT_ROOT` | `inventory.csv`, previews, per-run logs, aggregated features, handoff bundles. | inventory, preview, aggregate, handoff |

## Raw layout

Five folders named by recording date, grouped into two recruitment waves. Folder
names contain spaces and are matched exactly as configured in
`config/default.yaml`.

```
$VC_DATA_ROOT/
├── December 21 2025/     # winter wave, session IDs 1-62
│   ├── 28.mp4
│   └── ...
├── January 17 2026/      # winter wave
├── January 31 2026/      # winter wave
├── July 4 2026/          # summer wave, session IDs 102-261
└── August 1 2026/        # summer wave
```

- 62 mp4 files in total, and no other files. Extra files are reported by
  `vc inventory` rather than ignored.
- Each file is named by its numeric session ID alone, e.g. `28.mp4`. A
  non-numeric filename is reported, never guessed at.
- Session IDs fall in two ranges: 1-62 (winter) and 102-261 (summer). A file
  whose ID belongs to one wave but sits in the other wave's folder is reported.
- Recordings are roughly 10-12 minutes. One is known to be unusually short and
  is listed in `dataset.known_short_sessions` so it reads as expected.
- There are **no** separate per-speaker audio files. Audio comes only from the
  mp4.

## Work layout

Created as stages run. Transcripts are the most sensitive artifact in the
project: they stay here, are never printed, and are never included in a handoff
bundle.

```
$VC_WORK_ROOT/
├── audio/               # mono 16 kHz WAV per session (one per audio stream)
├── segments/            # normalised diarization output, may include text
├── speech/              # VAD-refined speech spans
├── roles/               # speaker-to-role assignment plus QC evidence
├── face/                # per-frame landmark measures
├── reference/           # psychiatrist voice reference clips, made by hand
├── models/              # downloaded model assets, pinned in the manifest
└── tmp/                 # scratch; cleaned up as stages go
```

## Output layout

```
$VC_OUT_ROOT/
├── inventory.csv
├── previews/            # one contact sheet per session; contains faces
├── logs/                # one log file per run
└── handoff/
    └── <date>_<git short hash>/
        ├── features.csv
        ├── feature_dictionary.csv
        ├── qc_report.csv
        ├── manifest.json
        └── README.md
```

Previews contain participants' faces. They exist so the crop can be confirmed by
eye, they stay under `$VC_OUT_ROOT`, and they are never committed or included in
a bundle.

## Labels, on the other machine

The labels file never exists on the extraction machine. On the label holder's
machine it is a CSV:

```csv
session_id,K6,SRS2
28,7,61
...
```

Optionally, a participant map overriding the one-session-per-participant default
(see `docs/decisions/0007-session-as-participant-grouping.md`):

```csv
session_id,participant_id
28,P014
...
```

`vc model` validates both: missing IDs, duplicates and non-numeric values are
reported, and every dropped session is listed with the reason.
