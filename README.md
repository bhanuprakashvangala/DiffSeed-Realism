# Visual Realism Does Not Predict Seed-Quality Utility in AI-Generated Synthetic Seeds

Code and run records for the paper by **Bhanu Prakash Vangala** and **Navya Vangala**.
Status: under review at *Artificial Intelligence in Agriculture*.

We trained eight generators (pixel and latent diffusion, GANs) on a five-class
soybean seed-quality corpus and scored the same fixed synthetic images for
realism, near-copying, real-versus-synthetic separability, usefulness on held-out
real seeds, sampler stability and cost. FID and real-seed usefulness are
essentially unrelated (Spearman rho = 0.071): FastGAN has the worst FID and the
largest gain, and a DDPM with broken class conditioning has the second-best FID.

## Layout

```
diffseed/           pipeline: data splits, generators (DDPM, SD-LoRA, GANs),
                    quality metrics, detector, classifiers, statistics, figures
scripts/            reproduce_paper.py (tables, figures, verification),
                    count_corpus.py
results/soybean/    saved run records: pooled and per-class image metrics,
                    finite-sample FID curves, LPIPS nearest neighbours, detector
                    AUCs, per-run test predictions, sampler diagnostic, cost
                    ledger, generator loss curves, provenance
outputs/            what reproduce.sh writes: tables/, figures/, verification.csv
data/               README and download script for the image corpus
reproduce.sh        single entry point
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, pandas, scipy, matplotlib
```

## Reproduce

**From the saved results (CPU, under a minute, no download):**

```bash
./reproduce.sh                     # same as: python scripts/reproduce_paper.py
```

This recomputes Table 2, Figures 2, 3, 6 and 7, and the numbers quoted in the
text from `results/`, and writes `outputs/verification.csv`, which compares each
recomputed value with the value printed in the manuscript.

**Full retraining (GPU):**

```bash
pip install -r requirements-full.txt     # see the torch index URL inside
./reproduce.sh full
```

This downloads the soybean corpus (Kaggle CLI), trains the eight generators at
the scarcity levels used, computes the image metrics, trains the real-versus-
synthetic detector and the downstream classifiers, runs the sampler diagnostic,
and then runs the same analysis on the new records. Generator training alone is
42.6 GPU-hours on one NVIDIA A10G (23 GB). Every stage is resumable. GPU results
are not bit-exact across hardware.

## Results

Table 2: pooled FID (1,600 generated vs 2,000 real images) against the change in
macro-F1 on held-out real seeds (ResNet-50, 100 real images per minority class,
3 training runs).

| Generator | FID | Delta macro-F1 (pp) |
|---|---|---|
| sd_lora_sdedit | 31.52 | 6.98 |
| ddpm_v1 (broken conditioning) | 50.67 | 4.05 |
| ddpm_ft | 54.25 | 4.55 |
| ddpm_fixed | 63.85 | 6.68 |
| sd_lora | 66.50 | 5.64 |
| sd_turbo_lora | 82.44 | 5.86 |
| dcgan | 122.80 | 3.18 |
| fastgan | 275.61 | 7.11 |

Spearman rho(FID, utility) = 0.071 over the eight generators.

Other numbers, all recomputed exactly from `results/`:

| | value |
|---|---|
| real-vs-real FID at N = 100 / 200 / 500 / 1000 | 69.05 / 48.76 / 28.01 / 16.70 |
| closest synthetic-to-real LPIPS; images below 0.15 | 0.264; none |
| detector AUC, repaired DDPM / SDEdit | 1.000 / 0.999 |
| DDPM 1000 steps: FID, output std | 107.7, 0.589 |
| DDPM 500 steps (production): FID, output std | 109.6, 0.576 |
| DDIM 100 steps: FID | 148.0 |
| DPM-Solver++ 100 steps: output std, saturated pixels, FID | 245.631, 99.7%, 352.1 |
| training GPU-minutes: FastGAN / repaired DDPM / SD LoRA | 3.7 / 397.4 / 57.6 |
| total generator training | 42.6 GPU-hours |

All 54 numbers checked by `scripts/reproduce_paper.py` match the manuscript.

## Notes

- Not regenerated from saved results: Figure 1 (schematic), Figure 4 (sample
  grid) and Figure 5 (closest synthetic-real image pairs). The last two show
  image pixels, so they come only from `./reproduce.sh full`. The distances
  printed on Figure 5 are recomputed from `results/soybean/quality/memorisation_n100.json`.
- Table 1 (per-class corpus counts) needs the images; after downloading, run
  `python scripts/count_corpus.py data/soybean_seeds`. The total (5,513) is
  checked against the inventory in `results/soybean/provenance.json`.
- The 42.6 GPU-hours is the sum of the generator-training entries in the cost
  ledger (`results/soybean/costs.json`). The ledger also records classifier and
  detector training; including those, the soybean run totals 56.3 GPU-hours.
- The DDPM costs are averaged over the four scarcity levels at which it was
  trained; FastGAN was trained once, at n = 100.
- Utility is measured with one classifier recipe and three training runs, and
  generators were trained once each, so image metrics carry no generator-seed
  interval. The rank correlation is over eight configurations.
- Trained generators, classifier checkpoints and the generated image sets
  (about 5 GB) are not included. They are available from the authors on request.

## Data

The soybean corpus (5,513 images, five quality classes) is public on Mendeley
Data (https://doi.org/10.17632/v6vzvfszj6.6); the pipeline fetches the Kaggle
mirror. See [data/README.md](data/README.md) for the corpus and for how the
synthetic images are generated.

## Citation

```bibtex
@article{vangala2026seedrealism,
  title   = {Visual Realism Does Not Predict Seed-Quality Utility in
             AI-Generated Synthetic Seeds},
  author  = {Vangala, Bhanu Prakash and Vangala, Navya},
  journal = {Artificial Intelligence in Agriculture},
  note    = {Under review},
  year    = {2026}
}
```

## License

Code: MIT ([LICENSE](LICENSE)). Run records and generated tables/figures under
`results/` and `outputs/`: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). The image
corpus is not redistributed and keeps its own terms.
