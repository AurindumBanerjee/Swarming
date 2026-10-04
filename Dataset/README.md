# PSO decap-placement datasets

Layout is **board-first**. Each board folder holds its Touchstone source plus every
`.mat` derived from it. Decap libraries and frequency grids are shared, so they live
in `Decaps/`.

```
Dataset/
  MPHY/          MPHY package  (5 supply rails)
  VDDQ_DDR3/     DDR3 board    (1 supply rail)
  Decaps/        capacitor libraries + frequency grids
  Reference/     the original 21-port benchmark PDN (y2.mat)
  Scripts/       converters and the MATLAB inspector
  Notebooks/     PSO notebooks
```

## Conventions for every `y_*.mat`

* variable `y`, shape `(N, N, F)` complex128 — multiport **admittance**, siemens
* **port 1 (MATLAB) / index 0 (Python) is the observation port**; the rest are decap sites
* sub-networks are **open-circuit** reductions: `Z = inv(Y_full)`, slice, `Y = inv(Z_sub)`.
  A raw `Y` submatrix would short every port left out — do not slice `Y` directly.
* the matching grid is `freq_sp.mat` (new SP extractions) or `freq78/81.mat` (legacy)

---

See `MPHY/README.md`, `VDDQ_DDR3/README.md` and `Decaps/README.md` for a file-by-file
breakdown of each folder, with port maps and measured metrics.

## MPHY/

### Current — `SP_Data_MPHY_68Caps.s73p` (Sep 2026 extraction)
73 ports: **1-5 = observation, one per rail**; 6-73 = 68 decap pads, contiguous per rail.
Grid `freq_sp.mat`: 2089 points, 100.9 Hz - 99.66 MHz.

| file | rail | N | pads | bare peak abs(Z11) |
|---|---|---|---|---|
| `y_sp_mer1.mat` | VDDE1V0_MER1 | 16 | 6-20 | 3.24 ohm |
| `y_sp_mer2.mat` | VDDE1V0_MER2 | 16 | 21-35 | 1.16 |
| `y_sp_pll1v0.mat` | VDD1V0_PLL | 15 | 36-49 | 3.52 |
| `y_sp_pll1v8.mat` | VDDA1V8_PLL | 16 | 50-64 | 24.3 |
| `y_sp_mphyvdd.mat` | MPHY_VDD | 10 | 65-73 | 3.15 |
| `y_sp_mphy_full.mat` | all five | 73 | - | port 1 = MER1 obs |

### Legacy — `MPHY_C28.s78p` (Aug 2025 extraction)
78 ports: 1-5 die, 6-10 observation, 11-78 pads. `y78.mat` is the full conversion;
`y_mer1/mer2/pll1v0/pll1v8/mphyvdd.mat` are the rail slices. Passive-only — no VRM
path, so bare impedances run to 1e12 ohm at anti-resonance.

## VDDQ_DDR3/

### Current — `SP_Data_VDDQ_DDR3_78Caps.s79p`
79 ports: **port 1 = observation**, ports 2-79 = 78 decap pads. Single rail.

| file | N | bare peak abs(Z11) |
|---|---|---|
| `y_sp_ddr_full.mat` | 79 | 0.502 ohm |
| `y_sp_ddr21.mat` | 21 (obs + first 20 pads) | 0.502 |

`y_sp_ddr21` exists for the `pure_python` baseline: Gauss-Jordan is O(n^3), so 79 ports
costs ~50x what 21 does per fitness evaluation.

### Legacy — `PCB_010316_..._pads9-5.S81P`
81 ports, 2 on an unrelated net. `y81.mat` full; `y_ddr.mat` (79) and `y_ddr21.mat` (21).

## Decaps/

| file | contents |
|---|---|
| `decaps_sp.mat` | **use this.** AVX library, 3347 shunt-mounted models, resampled to the 2089-point SP grid. Carries `names`. |
| `decaps_avx.mat` | source library: 6695 entries (3348 series + 3347 shunt), 301 points, 100 Hz - 100 MHz, with `names` and `fixture` |
| `decaps.mat` | legacy library, 3348 models on the 1391-point grid |
| `decaps_band.mat` / `freq_band.mat` | legacy library clipped to the 99-point legacy band |
| `freq_sp.mat` | 2089-point SP grid (identical for both boards) |

## Known issues before running PSO

1. **Port sampling in PythonBench is off-by-one.** `floor(u*(N_NODES-1))` yields port 0
   (the observation port) and never the last port. Use
   `min(1 + floor(u*(N_NODES-1)), N_NODES-1)`, and change the bound in
   `resolve_adjacent_ports` from `0 <= cand` to `1 <= cand`.
2. **`MAX_CAPS = 20` exceeds the site count** on four MPHY rails (9-15 pads).
   Set `MAX_CAPS = min(20, N_NODES - 1)`.
3. **The worst-case impedance sits at the 99.66 MHz band edge** on every dataset, and
   decaps cannot fix it there (ESL region). Stuffing every MPHY pad moves `sp_mer1`
   from 3.241 to only 3.211 ohm. Below 10 MHz the same configuration reaches 0.33 ohm.
   Either restrict the optimization band (<= 20-50 MHz) or extend the decap library
   above 100 MHz, otherwise PSO optimizes a point it cannot influence.
4. **Set targets per rail.** These rails differ by 20x in natural impedance
   (`sp_mer2` 1.16 ohm vs `sp_pll1v8` 24.3 ohm). One global threshold makes the
   high-impedance rails look like failures.


---

## Per-folder documentation

| folder | README covers |
|---|---|
| `MPHY/` | port map for both extractions, all 11 .mat files, per-rail metrics, the weak-capacitor-response issue |
| `VDDQ_DDR3/` | port map, all 8 .mat files, per-band impedance profile, cost of 79 vs 21 ports |
| `Decaps/` | the libraries, the series/shunt split, capacitance range, the band-edge problem |
