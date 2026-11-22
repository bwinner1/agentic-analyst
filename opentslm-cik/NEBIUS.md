# Training on Nebius, step by step

You rent a cloud computer with a big GPU, copy our code onto it, run one script, copy the results back,
then **delete the computer so it stops costing money**. Nebius bills by the hour while the VM exists, even if
nothing is running. The whole job should take about 1.5 to 2 hours, so expect a few dollars of your $100.

All commands below are typed in the **Terminal** app on your Mac, unless it says "on the VM".

## 1. Make an SSH key (once, on your Mac)

An SSH key is how your Mac proves who it is to the VM. Skip this if `ls ~/.ssh/id_ed25519.pub` already shows a file.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub        # copy this whole line; you paste it into Nebius in the next step
```

## 2. Create the VM in the Nebius console

1. Go to **Compute > Virtual machines > Create virtual machine**.
2. **GPU / platform:** pick **1 GPU**. Cheapest good choice: **1x L40S** (48 GB) if it is offered in your region.
   Otherwise **1x H100** (80 GB). Both are plenty; the L40S is usually about half the hourly price. Check the price
   the console shows before you confirm.
3. **Image:** an **Ubuntu 22.04 (or 24.04) image with CUDA / NVIDIA drivers** preinstalled.
4. **Boot disk:** **100 GB**.
5. **Public IP:** turn it **on** (so your Mac can reach the VM).
6. **Access / SSH:** set a username (for example `analyst`) and paste the key line from step 1.
7. Create it, wait until it says **Running**, and copy its **public IP address**.

## 3. Copy the code to the VM and log in

On your Mac (replace `<user>` and `1.2.3.4` with your username and IP; `~/Downloads` with where the zip is):

```bash
scp ~/Downloads/opentslm-cik-nebius.zip <user>@1.2.3.4:~
ssh <user>@1.2.3.4
```

The first time, type `yes` when it asks about the fingerprint. You are now **on the VM**.

## 4. Set up (on the VM, about 10 minutes, once)

```bash
sudo apt-get update -q && sudo apt-get install -y -q unzip tmux python3-venv
unzip -o -q opentslm-cik-nebius.zip
cat opentslm-cik/VERSION                     # must say: 2026-11-21 extra-data v5
tmux new -s cik                              # a "safe" terminal: keeps running if your Mac disconnects
bash opentslm-cik/run_nuvolos.sh setup       # paste your Hugging Face READ token when asked
```

It should end with `355 forecasting tasks in TimeNet`.

**About tmux:** if your connection drops, log in again with `ssh <user>@1.2.3.4` and type `tmux attach -t cik`
to get back to where you were.

## 5. Quick check, then the real runs (on the VM)

```bash
bash opentslm-cik/run_nuvolos.sh smoke       # about 3 minutes: first line must say "Code version: ... v5"
bash opentslm-cik/run_nuvolos.sh full        # about 1 hour: main model + the "no context" comparison
```

`full` trains two models: one that reads the business context, and one with the context removed. Comparing
them shows how much the context helps. That comparison is the headline result for the showcase. At the end it
prints a small table like this (lower mae and crps are better):

```
               model mae  naive mae  model crps  naive crps  parsed
with context      ...        ...        ...         ...        ...
no context        ...        ...        ...         ...        ...
```

## 6. Copy the results to your Mac

On the VM:

```bash
bash opentslm-cik/run_nuvolos.sh pack
exit                                          # leaves tmux; type exit again to leave the VM
```

On your Mac:

```bash
scp <user>@1.2.3.4:~/cik_work/results.zip ~/Downloads/results-nebius.zip
```

Check the file is on your Mac (`ls -lh ~/Downloads/results-nebius.zip`) and send it in the project.

## 7. Delete the VM (important)

In the Nebius console, open the VM and choose **Delete** (also delete its boot disk if it asks). "Stop" alone
can still charge for the disk. Then check **Billing** a little later to see the cost stop growing.

## If something goes wrong

- **`Permission denied (publickey)`** when connecting: the username or key doesn't match what you entered in step 2.
- **`nvidia-smi: command not found`** during setup: the image has no GPU drivers. Delete the VM and create it again
  with a CUDA/NVIDIA image.
- **Hugging Face `401` / `gated repo`:** the token is wrong or from an account without Llama access. Run
  `source cik_work/venv/bin/activate && huggingface-cli login` and try again.
- **Out of GPU memory:** shouldn't happen on an L40S/H100. If it does, training skips that one example and
  continues; it prints which one.
- **Anything else:** copy the last 30 lines of the screen into the project thread.
