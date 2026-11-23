# Forecast demo, step by step

> **Using this from the GitHub repo instead of the zip?** First download the checkpoint into `demo/checkpoints/` and the CiK data into `data/cik_test.parquet` (commands in the folder's main `README.md`), then start from step 2 below, run from the `opentslm-cik` folder.


A small web page that runs your finetuned model. You pick a Context is Key example (or paste your own
series), type a business note, and get a forecast chart plus a short explanation.

The page runs on your own computer and opens in your browser at `http://127.0.0.1:7860`. Nothing is
public unless you ask for a share link (see "Sharing" below).

## What is in the zip

| Path | What it is |
| --- | --- |
| `demo/app.py` | The demo page. |
| `demo/demo.sh` | One script: `setup` (once), `run`, `replay`. |
| `demo/checkpoints/best_model_full.pt` | Your trained model from the Nebius run (only the parts training changed; the app downloads Llama 3.2 1B itself). |
| `demo/replay/test_predictions_full.jsonl` | The model's saved answers from the Nebius test run, for replay mode. |
| `cik_core.py`, `data/cik_test.parquet`, `VERSION` | The prompt format and the CiK data the app uses. |

## On your Mac (recommended for the showcase)

Everything below is typed in the **Terminal** app.

1. **Unzip** the file from the project (say it is in Downloads):

   ```bash
   cd ~/Downloads && unzip -o -q opentslm-cik-demo.zip && cd opentslm-cik-demo
   ```

2. **Set up, once (about 10 minutes):**

   ```bash
   bash demo/demo.sh setup
   ```

   If macOS asks to install "command line developer tools", click Install, then run the line again.
   When it asks for Hugging Face, sign in with the same account (or token) you used for training. It
   needs Llama 3.2 access, which you already have. It ends with `Setup done`.

3. **Start the demo:**

   ```bash
   bash demo/demo.sh run
   ```

   Wait until it prints `Running on local URL: http://127.0.0.1:7860`, then open that address in your
   browser. Loading takes about a minute the first time.

4. **Use it:**
   - **Benchmark examples** tab: pick an example (★ = ones where the note clearly helps), press
     **Forecast**. The green line is what really happened, so you can see how close the model got.
     Try editing the note and running it again.
   - **Your own series** tab: paste numbers (or `date,value` lines from a spreadsheet), set how many
     steps to forecast, type a note, press **Forecast my series**.
   - **Same-size comparison** tab: the scores against plain Llama 3.2 1B and Gemma 3 1B (not fine-tuned) on
     the same 105 tasks, and, for any test task, what each of the three models actually wrote. Good for the
     showcase video: it shows why fine-tuning matters.
   - Tick **Also forecast without the note** to draw a second line without the note. The gap between
     the blue and orange lines is what the note changed. This takes twice as long.

5. **Stop it:** press `Ctrl + C` in Terminal.

Expect roughly 10 to 60 seconds per forecast on the MacBook Air; long horizons (72+ steps) take longer.

## No time to wait, or the model won't load? Replay mode

```bash
bash demo/demo.sh replay
```

Same page, but it shows the answers the model already gave in the Nebius test run. It starts instantly and
needs no model (setup still has to have run once). The note box is locked, and "your own series" is off.

## On the Nebius GPU machine (faster)

Start the VM in the Nebius console, then on your Mac:

```bash
scp ~/Downloads/opentslm-cik-demo.zip <user>@<vm-ip>:~
ssh -L 7860:127.0.0.1:7860 <user>@<vm-ip>
```

The `-L 7860:...` part lets your Mac's browser reach the page running on the VM. Then on the VM:

```bash
unzip -o -q opentslm-cik-demo.zip && cd opentslm-cik-demo
bash demo/demo.sh setup     # once
bash demo/demo.sh run
```

Open `http://127.0.0.1:7860` in your Mac's browser. Stop the VM when you are done.

## Sharing (optional)

`bash demo/demo.sh run --share` also prints a public `https://....gradio.live` link that works for about
a week **while your computer (or VM) keeps the demo running**. Anyone with the link can use it, so share it
only where you mean to. For the showcase, a screen recording or screenshots are often simpler.

## Swapping in a newer checkpoint

When the extra-data training round finishes, copy its `best_model.pt` into `demo/checkpoints/` with a new
name, for example:

```bash
cp ~/Downloads/best_model.pt demo/checkpoints/best_model_round2.pt
```

and restart with `bash demo/demo.sh run`. The app loads the **newest** file in that folder, and a
**Model checkpoint** menu appears at the top so you can switch between them live. To use a specific file:
`bash demo/demo.sh run --ckpt path/to/file.pt`.

This works as long as the new model was trained from the same base (Llama 3.2 1B) with the same prompt
format (`VERSION` says `baseline-in-prompt v4.1`). If either changed, the app needs the matching `cik_core.py`.

## Reading the page honestly

- The **explanation is draft commentary**. The model writes fluent text but can invent numbers and dates.
  The page labels it that way; check it against the chart.
- On the test set the model is about as accurate as the simple baseline overall (average error 0.225 vs
  0.221), and clearly better on some task types (sensor maintenance, sensor spikes, fan speed). It misses
  others, such as the heat-wave spike. The ★ examples are the strong ones; showing one miss too is fair.
- Pasted series are a stretch: the model learned from CiK tasks, so business notes far from those
  (promotions, for example) may be ignored.

## If something goes wrong

- **`401` / `gated repo` during setup:** the Hugging Face account has no Llama 3.2 access. Sign in again:
  `source demo/.work/venv/bin/activate && hf auth login`.
- **An error mentioning `mps` or `bfloat16` when forecasting on the Mac:** run on the processor instead
  (slower): `bash demo/demo.sh run --device cpu`. Updating macOS also helps.
- **`No checkpoint found`:** put a `.pt` file in `demo/checkpoints/`.
- **`Address already in use`:** the demo is already running in another Terminal window. Close it, or use
  `bash demo/demo.sh run --port 7861`.
- **Anything else:** copy the last 30 lines of Terminal into the project thread.
