
# Aspect Dependency Graph Network: Inferring Latent Sentiment Beyond Mention for Aspect-Based Sentiment Analysis

___

## News

[26.05.05] Our code is released!

___

<details open>
<summary><h1>Environment</h1></summary>


### Hardware
- GPU: NVIDIA GeForce RTX 5090 (32 GB VRAM)

### Software
- OS: Windows 10 (WSL2 with Ubuntu 22.04)
- Docker Desktop with NVIDIA Container Toolkit
- Base Image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- CUDA: 12.8
- cuDNN: 9
- Python: 3.11
- PyTorch: 2.7.1

### Setup
This project uses Docker for environment management. See `Dockerfile` and `requirements.txt` for full dependency list.

```bash
# Build and run with Docker
docker build -t ADGN_image .
docker run -it --gpus all -v $(pwd):/workspace --name ADGN_con ADGN_image
```

Or open in VS Code with the Dev Containers extension for automatic setup.


<details open>
<summary><h1>Run</h1></summary>

Follow the steps below to train and test the model.

### Step 1. Move to the `src` directory

```bash
cd src
```

### Step 2. Run the script

```bash
bash run_ADGN.sh
```

> **Note**
>
> If the script does not run due to a `/bin/bash^M` error, convert the line endings to the Linux format.
>
> ```bash
> sed -i 's/\r$//' run_ADGN.sh
> ```
>
> Then run the script again.
>
> ```bash
> bash run_ADGN.sh
> ```

Modify `domain` to change the target domain, and modify `IS_PREPROCESS` to control whether preprocessing is performed.

```bash
domain='Rest-TA'
IS_PREPROCESS=1
```

It runs training first:

```bash
--is_preprocess 1
--is_train 1
```

Then it runs testing:

```bash
--is_preprocess 0
--is_train 0
```

Logs are saved to:

```bash
../logs/${domain}/
```

Results are saved to:

```bash
../results/${domain}/
```
</details>


## Citation

1st round revision
