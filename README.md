# FGPTM
>
> [**FGPTM: a focused graph attention network-based sequence–structure co-modeling framework for predicting protein post-translational modification sites**]

### Installation

Create and activate the conda environment:

```bash
conda create -n fgptm python=3.8 -y
conda activate fgptm
```

Install PyTorch and project dependencies:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 -c pytorch -y
pip install -r requirements.txt
```

Install the `torch-geometric` packages that match your CUDA/PyTorch build.

### Data Layout

Each PTM type is selected by name and stored under `./data/<ptm_type>/`:

```text
data/
  mkdssp
  <ptm_type>/
    train.txt
    test.txt
    graph_cache/
  pdb_structure_<ptm_type>/
    <uniprot_ID>.pdb
```

`train.txt` and `test.txt` use the existing four-column format:

```text
sequence label uniprot_ID modify_site
```

### Training

Single-node distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py \
  --ptm_type Musite_Methylation_K \
  --output_dir outputs/Musite_Methylation_K \
  --batch_size 32 \
  --width 128 \
  --epochs 12 \
  --lr 1e-4 \
  --warmup-lr 2e-4 \
  --min-lr 1e-5 \
  --num_heads 4
```

For another PTM type, add the matching files under `data/<ptm_type>/` and pass
that name to `--ptm_type`.

### Evaluation

Evaluate the test split with a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 eval.py \
  --ptm_type Musite_Methylation_K \
  --ckpt outputs/Musite_Methylation_K/checkpoint11.pth \
  --batch_size 64 \
  --width 128 \
  --num_heads 4
```
