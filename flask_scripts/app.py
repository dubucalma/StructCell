"""
Cell-type-aware protein structure prediction web app.

A Flask server that predicts how a protein's 3D structure may differ across
human cell types. Per-residue ESM-2 embeddings of the input sequence are
combined with cell-type-specific PINNACLE embeddings for the given gene, and
ESMFold folds one structure per cell type alongside a context-free baseline.

Pipeline:
    1. Embed the sequence with ESM-2 (esm2_t36_3B_UR50D, layer 36).
    2. Look up PINNACLE embeddings for the gene in every cell type it appears in.
    3. Project each PINNACLE vector (128 -> 2560) and add it to the ESM-2
       embeddings (scaled by 0.1) to get cell-type-conditioned embeddings.
    4. Fold a baseline structure (ESM-2 only) and one structure per cell type
       with ESMFold.
    5. Render PNG thumbnails with headless ChimeraX (optional).
    6. Compare each cell-type structure to the baseline with US-align
       (TM-score, RMSD) (optional).
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os, threading, uuid, json, subprocess, traceback
from pathlib import Path

# ===================  Configuration & ChimeraX detection =====================
app = Flask(__name__)
OUTPUT_BASE = "/path/to/output_web"   # TODO: set to your output directory
CHIMERAX_BIN = "/apps/software/2024a/software/ChimeraX/1.10.1-1-gfbf-2024a-CUDA-12.8.0/usr/bin/chimerax"   

import shutil as _shutil
if os.path.isabs(CHIMERAX_BIN):
    CHIMERAX_AVAILABLE = os.path.isfile(CHIMERAX_BIN) and os.access(CHIMERAX_BIN, os.X_OK)
else:
    CHIMERAX_AVAILABLE = _shutil.which(CHIMERAX_BIN) is not None
    if CHIMERAX_AVAILABLE:
        CHIMERAX_BIN = _shutil.which(CHIMERAX_BIN)   # resolve to full path now

XVFB_AVAILABLE = _shutil.which("xvfb-run") is not None

if not CHIMERAX_AVAILABLE:
    print(f"WARNING: ChimeraX not found ({CHIMERAX_BIN!r}).\n"
          f"  → Run `module load ChimeraX && which chimerax` on the cluster and\n"
          f"    hard-code the absolute path as CHIMERAX_BIN in app.py.\n"
          f"  → PNG thumbnails disabled; 3Dmol viewer still works.")
else:
    print(f"INFO: ChimeraX found at {CHIMERAX_BIN!r}. "
          f"xvfb-run {'available' if XVFB_AVAILABLE else 'not found — using --offscreen only'}.")

# =================== US-align detection ====================================
# US-align (Zhang lab, Nature Methods 2022) is the modern successor to TM-align.
# It outputs TM-score and RMSD in the same format and handles proteins, RNA, DNA.
USALIGN_BIN       = "USalign"         
USALIGN_AVAILABLE = _shutil.which(USALIGN_BIN) is not None
if USALIGN_AVAILABLE:
    print(f"INFO: US-align found — structural metrics (TM-score, RMSD) will be computed.")
else:
    print(f"INFO: US-align not found — skipping structural metrics. "
          f"Install from https://zhanggroup.org/US-align/ and add to PATH.")

PINNACLE_EMBED_PATH  = "/path/to/pinnacle_protein_embed.pth"   # TODO: set to PINNACLE_EMBED_PATH
PINNACLE_LABELS_PATH = "/path/to/pinnacle_protein_labels_dict.txt" # TODO: set to PINNACLE_LABELS_PATH

jobs = {}   # in-memory job store

# =================== Cell-type colour palette ===================================
# Each cell type gets a stable colour used in both 3Dmol.js and ChimeraX renders.
CELL_TYPE_COLORS = [
    "#ef4444",   # red
    "#8b5cf6",   # violet
    "#3b82f6",   # blue
    "#10b981",   # emerald
    "#f59e0b",   # amber
    "#ec4899",   # pink
    "#06b6d4",   # cyan
    "#f97316",   # orange
    "#84cc16",   # lime
    "#6366f1",   # indigo
]

def _cell_color(index: int) -> str:
    return CELL_TYPE_COLORS[index % len(CELL_TYPE_COLORS)]

def _hex_to_cx(hex_color: str) -> str:
    """Convert #rrggbb to ChimeraX hex format (same, just ensure no #)."""
    return hex_color.lstrip("#")


# ===================  Routes ========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fold", methods=["POST"])
def fold():
    data          = request.json
    protein_id    = data.get("protein_id",    "").strip()
    sequence      = "".join(data.get("sequence", "").split())   # strip ALL whitespace/newlines
    gene          = data.get("gene",          "").strip()
    mutation_name = data.get("mutation_name", "").strip()

    if not protein_id or not sequence or not gene:
        return jsonify({"error": "protein_id, sequence and gene are required"}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status":             "running",
        "logs":               [],
        "structures":         [],
        "original_structure": None,
        "mutation_name":      mutation_name or protein_id,
        "metrics":            {},    # TM-score / RMSD table added after folding
    }

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, protein_id, sequence, gene),
        daemon=True
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/image/<path:relpath>")
def serve_image(relpath):
    directory = os.path.dirname(os.path.join(OUTPUT_BASE, relpath))
    filename  = os.path.basename(relpath)
    return send_from_directory(directory, filename)


@app.route("/api/pdb/<path:relpath>")
def serve_pdb(relpath):
    directory = os.path.dirname(os.path.join(OUTPUT_BASE, relpath))
    filename  = os.path.basename(relpath)
    return send_from_directory(directory, filename)


@app.route("/api/open_chimerax", methods=["POST"])
def open_chimerax():
    """Launch ChimeraX GUI with the specified PDB (non-blocking)."""
    data    = request.json
    relpath = data.get("pdb", "").strip()
    if not relpath:
        return jsonify({"error": "pdb path required"}), 400

    abs_path = os.path.join(OUTPUT_BASE, relpath)
    if not os.path.exists(abs_path):
        return jsonify({"error": f"PDB file not found: {abs_path}"}), 404

    try:
        subprocess.Popen(
            [CHIMERAX_BIN, abs_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return jsonify({"status": "launched"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================== Pipeline =========================================

def _log(job_id, msg):
    print(msg)
    jobs[job_id]["logs"].append(msg)


def _run_pipeline(job_id, protein_id, sequence, gene):
    try:
        import torch, torch.nn as nn, numpy as np, esm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(job_id, f"Device: {device}")

        # ── 1. ESM-2 embeddings ───────────────────────────────────────────
        _log(job_id, "Loading ESM-2 (t36_3B)…")
        esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        esm_model = esm_model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()

        data = [(protein_id, sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            out = esm_model(batch_tokens, repr_layers=[36], return_contacts=False)
            token_repr = out["representations"][36]

        L = len(sequence)
        embeddings = token_repr[0, 1:L+1].unsqueeze(0).float()   # (1, L, 2560)
        _log(job_id, f"ESM-2 embedding: {embeddings.shape}")

        del esm_model
        torch.cuda.empty_cache()

        # ── 2. PINNACLE embeddings ────────────────────────────────────────
        _log(job_id, "Loading PINNACLE embeddings…")
        embed = torch.load(PINNACLE_EMBED_PATH)
        with open(PINNACLE_LABELS_PATH) as f:
            protein_labels = json.loads(f.read().replace("'", '"'))

        cell_types = np.array(protein_labels["Cell Type"])
        genes      = np.array(protein_labels["Name"])

        ct_to_idx = {}
        for i, ct in enumerate(protein_labels["Cell Type"]):
            if ct.startswith("CCI_"):
                ct_to_idx[ct.replace("CCI_", "")] = i

        mask = genes == gene
        _log(job_id, f"Gene '{gene}' found in {mask.sum()} cell types")
        if mask.sum() == 0:
            raise ValueError(f"Gene '{gene}' not found in PINNACLE labels.")

        pinnacle_results = {}
        for ct in cell_types[mask]:
            ct_name = ct.replace("CCI_", "")
            ct_numerical_idx = ct_to_idx.get(ct_name)
            ct_mask  = cell_types == ct
            ct_genes = genes[ct_mask].tolist()
            local_idx = ct_genes.index(gene)
            pinnacle_results[ct_name] = embed[ct_numerical_idx][local_idx]  # (128,)

        # ── 3. Aggregate ESM2 + PINNACLE ─────────────────────────────────
        _log(job_id, "Aggregating ESM-2 + PINNACLE embeddings…")
        proj_pinnacle = nn.Linear(128, 2560).to(device)

        aggregated_projected = {}
        for ct_name, pinnacle_vec in pinnacle_results.items():
            pv                 = pinnacle_vec.to(device).unsqueeze(0).unsqueeze(0)
            pinnacle_projected = proj_pinnacle(pv)
            pinnacle_broadcast = pinnacle_projected.expand(-1, embeddings.shape[1], -1)
            combined           = embeddings + 0.1 * pinnacle_broadcast
            aggregated_projected[ct_name] = combined.squeeze(0)   # (L, 2560)

        # ── 4. Load ESMfold once ──────────────────────────────────────────
        _log(job_id, "Loading ESMfold…")
        esmfold = esm.pretrained.esmfold_v1().eval().to(device)

        # ── 5. Baseline structure (plain ESM-2, no PINNACLE) ─────────────
        _log(job_id, "Generating baseline ESMfold structure (no PINNACLE context)…")
        out_dir_orig = os.path.join(OUTPUT_BASE, protein_id, "original")
        os.makedirs(out_dir_orig, exist_ok=True)
        orig_mat = embeddings.squeeze(0)   # (L, 2560)

        pdb_path_orig, _ =  ## "Use ESMfold to obtain the predictions given input embeddings"
        _log(job_id, f"  ✓ Original PDB: {pdb_path_orig}")


        rel_pdb_orig = os.path.relpath(pdb_path_orig, OUTPUT_BASE)
        # Baseline rendered in neutral white/grey
        rel_img_orig = _chimerax_render(job_id, pdb_path_orig, "original",
                                        color_hex=None)   # None → default bychain
        jobs[job_id]["original_structure"] = {
            "label":  "Original ESMfold",
            "image":  rel_img_orig,
            "pdb":    rel_pdb_orig,
            "color":  "#ffffff",   # white in 3Dmol viewer
        }

        # ── 6. ESMfold per cell type ──────────────────────────────────────
        ct_pdb_paths = {}   # used for TM-align
        for idx, (ct_name, ct_embedding) in enumerate(aggregated_projected.items()):
            color = _cell_color(idx)
            _log(job_id, f"Folding: {ct_name} (color {color})…")

            safe_ct_name = ct_name.replace(" ", "_")
            out_dir = os.path.join(OUTPUT_BASE, protein_id, safe_ct_name)
            os.makedirs(out_dir, exist_ok=True)

            pdb_path, _ = ## "Use ESMfold to obtain the predictions given input embeddings"
            _log(job_id, f"  ✓ PDB: {pdb_path}")
     
            ct_pdb_paths[ct_name] = pdb_path

            rel_pdb = os.path.relpath(pdb_path, OUTPUT_BASE)
            rel_img = _chimerax_render(job_id, pdb_path, ct_name,
                                       color_hex=color)
            jobs[job_id]["structures"].append({
                "cell_type": ct_name,
                "image":     rel_img,
                "pdb":       rel_pdb,
                "color":     color,   # ← sent to frontend for 3Dmol colouring
            })

        jobs[job_id]["status"] = "done"
        _log(job_id, f"✓ All done — {len(aggregated_projected)} structures generated.")

        # ── 7. US-align: TM-score / RMSD ───────
        if USALIGN_AVAILABLE:
            _log(job_id, "Computing structural metrics via US-align (TM-score / RMSD)…")
            threading.Thread(
                target=_compute_metrics,
                args=(job_id, pdb_path_orig, ct_pdb_paths),
                daemon=True
            ).start()

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        _log(job_id, f"ERROR: {traceback.format_exc()}")


# ======================= ChimeraX rendering =====================================

def _chimerax_render(job_id, pdb_path, label, color_hex=None):
    """
    Render a PNG thumbnail via ChimeraX
    """
    if not CHIMERAX_AVAILABLE:
        _log(job_id, f"  ⚠ ChimeraX not available — skipping PNG for {label}")
        return None

    img_path    = pdb_path.replace(".pdb", ".png")
    script_path = pdb_path.replace(".pdb", ".cxc")

    with open(script_path, "w") as f:
        f.write(_chimerax_script(pdb_path, img_path, color_hex=color_hex))

    env_base = os.environ.copy()
    attempts = [
        ([CHIMERAX_BIN, "--nogui", "--offscreen", "--script", script_path],
         env_base),
        (["xvfb-run", "-a", CHIMERAX_BIN, "--nogui", "--offscreen", "--script", script_path],
         {**env_base, "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe"}),
    ]

    for cmd, env in attempts:
        try:
            result = subprocess.run(cmd, timeout=120, check=True,
                                    capture_output=True, text=True, env=env)
            _log(job_id, f"  ✓ ChimeraX image: {img_path}")
            return os.path.relpath(img_path, OUTPUT_BASE)
        except subprocess.CalledProcessError as e:
            _log(job_id, f"  ⚠ ChimeraX attempt failed (exit {e.returncode}): {' '.join(cmd[:3])}")
            for line in (e.stderr or "").strip().splitlines()[-3:]:
                _log(job_id, f"    {line}")
        except Exception as e:
            _log(job_id, f"  ⚠ ChimeraX error: {e}")

    _log(job_id, f"  ✗ All ChimeraX render attempts failed for {label}")
    return None


def _chimerax_script(pdb_path, img_path, color_hex=None):
    if color_hex:
        cx_hex    = _hex_to_cx(color_hex)
        color_cmd = f"color #{cx_hex}"
    else:
        color_cmd = "color bychain"   # baseline: rainbow by chain
    return f"""open "{pdb_path}"
preset publication 1
lighting soft
{color_cmd}
cartoon
view all
save "{img_path}" width 1400 height 1050 supersample 3
exit
"""


# ========================= US-align metrics =====================================

def _compute_metrics(job_id, reference_pdb, ct_pdb_paths):
    """
    Run US-align (Nature Methods 2022) between the baseline structure and each
    cell-type structure.  US-align is a drop-in successor to TM-align that
    also handles RNA/DNA/complexes and is more accurate on flexible regions.
    Output includes:
      TM-score (normalised to reference length) and RMSD of aligned residues.

    Results stored in jobs[job_id]["metrics"].
    """
    import re
    metrics = {}
    for ct_name, pdb_path in ct_pdb_paths.items():
        try:
            result = subprocess.run(
                [USALIGN_BIN, pdb_path, reference_pdb, "-mol", "prot"],
                capture_output=True, text=True, timeout=60
            )
            stdout = result.stdout

            # US-align output format (same as TM-align):
            #   TM-score= 0.9123 (normalized by length of Structure_2, ...)
            #   RMSD=   1.23, ...
            # We want TM-score normalised to Structure_2 (the reference).
            tm_match   = re.search(
                r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2", stdout)
            rmsd_match = re.search(r"RMSD=\s*([\d.]+),", stdout)

            tm_score = float(tm_match.group(1))   if tm_match   else None
            rmsd     = float(rmsd_match.group(1)) if rmsd_match else None

            metrics[ct_name] = {
                "tm_score":      tm_score,
                "rmsd_angstrom": rmsd,
                "tool":          "US-align (Nature Methods 2022)",
            }
            _log(job_id, f"  {ct_name}: TM-score={tm_score:.3f}, RMSD={rmsd:.2f}Å")

        except Exception as e:
            _log(job_id, f"  ⚠ US-align failed for {ct_name}: {e}")
            metrics[ct_name] = {"tm_score": None, "rmsd_angstrom": None, "tool": "US-align"}

    jobs[job_id]["metrics"] = metrics

    # Embed metrics directly into each structure dict so the frontend
    # can display TM-score / RMSD per card without a separate lookup.
    for s in jobs[job_id]["structures"]:
        m = metrics.get(s["cell_type"], {})
        s["tm_score"]      = m.get("tm_score")
        s["rmsd_angstrom"] = m.get("rmsd_angstrom")

    _log(job_id, "✓ Structural metrics computed (US-align).")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
