# Setting up whisper-diarization separately

This project does **not** depend on
[whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization).
Its requirements are heavy and do not install cleanly on Apple silicon, so
pulling them into this lockfile would make every other stage harder to install
for no benefit. It is run as an external command in its own environment, and its
output is read by this project's `import` backend.

You probably do not need this. The preferred path is to obtain the **original**
diarization output from the lab: reusing the manuscript's own transcripts keeps
the comparison against its text features apples-to-apples, whereas re-running
the tool would confound "new modality" with "new transcripts". See
[ADR 2](../docs/decisions/0002-pluggable-diarization-backends.md).

## Install it in its own environment

```bash
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd whisper-diarization
uv venv --python 3.11 .venv-wd
source .venv-wd/bin/activate
pip install -r requirements.txt
```

Record the commit you used; it is not recoverable from the tool's output, and
the manifest cannot capture it automatically:

```bash
git rev-parse --short HEAD
```

## Point this project at it

In `config/default.yaml`, or an overlay:

```yaml
diarization:
  backend: "whisper_diarization"
  whisper_diarization:
    command: ["/path/to/whisper-diarization/.venv-wd/bin/python",
              "/path/to/whisper-diarization/diarize.py"]
    language: "ja"
    output_dir: "diarization"     # relative to $VC_WORK_ROOT
    timeout_seconds: 3600.0
```

`output_dir` is where the tool writes and where its output is read back from.
Keep it under `$VC_WORK_ROOT`: its output contains transcripts, which are the
most sensitive artifact in this project and must never enter the repository.

## How it runs

`vc diarize` invokes the command once per session, with the recording's path and
the configured language, in `output_dir`. Existing output for a session is
reused rather than regenerated, since a run takes minutes per session — delete a
file to force it to be redone.

Whatever the tool produces is then parsed by the same importer that reads output
produced elsewhere, so there is one parser rather than two. If your version
writes filenames this project does not recognise, add a pattern to
`diarization.import_patterns` rather than renaming files by hand; `vc diarize`
reports every file it could not match to a session.

## Verify the output before a full run

```bash
uv run vc --sessions 28 diarize
```

Check the reported speaker count is 2 and that coverage looks sane, then run the
rest.
