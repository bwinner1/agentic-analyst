# Running the CiK training on Nuvolos (VS Code app, Tesla T4)

Same steps as the Colab notebook, but typed in a terminal. One script, `run_nuvolos.sh`, does the work in four
commands. Credits are limited, so the runs are kept small, and you should **stop the app when you are not using it**.

## 0. Before you start

- In Nuvolos, open the VS Code app with a **GPU (T4) size**.
- Have your Hugging Face **read** token ready (huggingface.co/settings/tokens, the account with Llama access).
- Open a terminal in VS Code: menu **Terminal > New Terminal**.

## 1. Upload the code

1. In the VS Code file list on the left, right-click the folder you work in (usually `files`) and choose **Upload...**.
   Pick `opentslm-cik.zip`.
2. In the terminal, go to that folder and unzip (replace `/files` if your folder is different):

```bash
cd /files
rm -rf opentslm-cik                 # removes an older copy, if there is one
unzip -o -q opentslm-cik.zip
cat opentslm-cik/VERSION            # must say: 2026-11-21 extra-data v5
```

If the upload renamed the file (for example `opentslm-cik (1).zip`), use that name in the `unzip` line.

## 2. Set up (once, about 10 to 15 minutes)

```bash
bash opentslm-cik/run_nuvolos.sh setup
```

This checks the GPU, makes a Python environment (it works without admin rights: it tries a venv, then conda,
then a plain user install), downloads OpenTSLM and TimeNet (with your PR 788), removes
`torchao`, asks for your Hugging Face token (paste it and press Enter; nothing shows while you paste, that's
normal), and builds CiK into TimeNet. It should end with `355 forecasting tasks in TimeNet`.

Everything goes into a folder `cik_work` next to `opentslm-cik`, so it survives restarts of the app.

## 3. Smoke test (about 5 minutes)

```bash
bash opentslm-cik/run_nuvolos.sh smoke
```

Check two things:
- The first line says `Code version: 2026-11-21 extra-data v5`.
- At the end, each `wrote:` line starts with `Forecast: 1: ... 2: ...` (numbered lines). The numbers will be poor
  after only 8 examples; that's fine.

## 4. The real run (about 20 to 30 minutes)

So the run keeps going if your browser disconnects, start it in the background and watch the log:

```bash
nohup bash opentslm-cik/run_nuvolos.sh train > cik_work/train.log 2>&1 &
tail -f cik_work/train.log
```

Press **Ctrl+C** to stop watching (the run keeps going). Run the `tail -f` line again to look later.
It is finished when the log ends with `Training and scoring done`. Training stops after at most 4 epochs.

## 5. Download the results

```bash
bash opentslm-cik/run_nuvolos.sh pack
```

Then in the VS Code file list open `cik_work`, right-click `results.zip`, choose **Download**, and send it in the
project. Then **stop the Nuvolos app** so it stops using credits.

## If something goes wrong

- **`No GPU found`**: the app was started without a GPU. Stop it and start it again with a T4 size.
- **`No space left on device`**: run `df -h /files` to see what's free (need about 10 GB). Free space with
  `rm -rf cik_work/tmp cik_work/hf_cache`. If `/files` is too small, ask Nuvolos support or put the work folder elsewhere:
  `WORK=/path/with/space bash opentslm-cik/run_nuvolos.sh setup` (use the same `WORK=...` for every command).
- **An install or download fails with a connection, proxy or 403 error**: Nuvolos may block some sites. Copy the
  error into the project thread. For Hugging Face you can test with `curl -I https://huggingface.co`.
- **`401` or `gated repo` from Hugging Face**: the token is wrong or is from an account without Llama access.
  Run `huggingface-cli login` again.
- **The TimeNet build fails**: upload the CiK parquet (`test-00000-of-00001.parquet`) into `/files`, then add
  `--source hf --data /files/test-00000-of-00001.parquet` to the `python train_cik.py` and `python eval_cik.py`
  lines in `run_nuvolos.sh` (in the `smoke` and `train` sections).
- **`torchao >0.16 required`**: run `source cik_work/venv/bin/activate && pip uninstall -y torchao`, then retry.
- **Setup stopped halfway**: just run the same `setup` command again; it skips what is already done.
- **CUDA out of memory (an error, not just a warning)**: in `run_nuvolos.sh` change `--max_history 256` to `128`.
