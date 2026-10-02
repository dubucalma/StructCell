# StructCell: Injection of celltype context into protein structures

This README explains how to build the interactive platform for predicting and visualizing protein structures across multiple cell types.

*By Athena Schumacher, Alma Dubuc. With the contribution of Dana Dayan.*

![Workflow overview](images/Graphical_Abstract_LLM_Final_Project.png)
*Picture created with BioRender.*

## Requirements

- 1 GPU (CUDA 11.8 or 12.4)
- [ChimeraX 1.10](https://www.cgl.ucsf.edu/chimerax/) *(optional — for PNG thumbnails; 3Dmol viewer works without it)*
- 40GB RAM, 4 CPUs

**For Yale HPC users, here is our configuration**

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --partition=gpu_devel --gpus=1
```

`gpu_devel` is time-limited. For long jobs, you may need to switch to another GPU partition (for training purposes).

## Setup

### 1. Conda environment - ESM2, ESMFolf, Flask

This environment contains ESM2, ESMFold, flask and all their dependencies.

```bash
conda env create -f environment.yml
conda activate structcell_env
```

### 2. PINNACLE contextual embeddings

In addition to this environment, gitclone PINNACLE to obtain contextual embeddings.

```bash
git clone https://github.com/mims-harvard/PINNACLE
cd PINNACLE
# follow their README to download pinnacle_protein_embed.pth and pinnacle_protein_labels_dict.txt
```

Then update the paths in `app.py`:
```python
PINNACLE_EMBED_PATH  = "/path/to/pinnacle_protein_embed.pth"
PINNACLE_LABELS_PATH = "/path/to/pinnacle_protein_labels_dict.txt"
```


### 3. ChimeraX *(HPC only, optional)*

On the cluster:
```bash
module load ChimeraX
which chimerax
```
Then hard-code the path in `app.py`:
```python
CHIMERAX_BIN = "/path/to/chimerax"
```
Without ChimeraX, PNG thumbnails are disabled but the 3Dmol viewer works normally.


## Interactive Website

The interactive platform has been built using flask. The 2 scripts to start the platform are available in`flask_scripts/`. 

In order to launch the platform:

```bash
conda activate structcell_env
python flask_scripts/app.py
```

Then open `http://localhost:5000` in your browser.




## Cell Type Specific Predictions

`flask_scripts/app.py` extracts ESM2 sequence embeddings, aggregates them with cell type-specific PINNACLE embeddings, and reinjects the combined representation into the ESMFold decoder to generate cell type-aware protein structures.
