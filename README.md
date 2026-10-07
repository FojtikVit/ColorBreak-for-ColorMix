This is a tool for breaking up PrusaSlicer ColorMix surfaces and it is vibecoded, including the rest of this README, but I think it's fun. The goal of this is not to "improve" ColorMix per se - a ColorMix surface is meant to look like the target colour when viewed from afar. ColorBreak will typically reduce this effect, as the non-uniformity makes the colours blend together a bit less in the eye. Instead, the aim of ColorBreak is to use the ColorMix feature to create an interesting texture on the print, ideally less "artificial" looking, visible from any distance. It's not trying to hide that two or three different colours were used, but embrace that fact and make it look nice. It's like texturing in fancy Minecraft builds.

# ColorBreak

Break a surface painted with a PrusaSlicer **ColorMix** virtual extruder into small, random,
irregular patches of mix ratios, so the regular layer-by-layer alternation of
filaments no longer reads as stripes.

ColorMix prints a blend such as "15 % blue / 85 % white" by cycling the two (or three) filaments layer
by layer. The cycle is perfectly regular, so the layer lines become very visible, especially
for contrasting colours and uneven ratios. ColorBreak re-paints such a surface with many small patches that use slightly different blends (and the pure colours), chosen so that

* the area-weighted mix of all patches equals the original ratio (or any ratio you ask for),
* the patch shapes are irregular blobs rather than brush circles or Voronoi polygons,
* neighbouring patches can be made to print *out of phase* so their stripes are less likely to line up,
* every patch is large enough to print cleanly.

Two- and three-colour blends are supported; pieces may use fewer colours than the source.

Works on PrusaSlicer 3.0 `.3mf` project files (developed against 3.0.0-alpha12). Only the
painting (`Metadata/Slic3r_facets_annotation.json`) and, when new virtual extruders are
needed, the `virtual_extruders` list in `Metadata/PrusaSlicer3_project.json` are changed;
everything else in the archive is copied byte for byte.

## Installation

Python 3.10 or newer, numpy and scipy.

```sh
python -m venv .venv && . .venv/bin/activate      # or: uv venv .venv
pip install -e .                                   # or: uv pip install -e .
colorbreak --help
```

## Usage

```sh
# what is in the file: physical/virtual extruders, printed cycles, painted areas
colorbreak inspect model.3mf

# break the surface painted with virtual extruder 17 (a 2-colour blend)
colorbreak break model.3mf -o model_broken.3mf -e 17 --contrast 0.7 --preview preview.png

# break every painted virtual extruder (2- and 3-colour), creating blends on a 10 % grid
colorbreak break model.3mf -o model_broken.3mf -e all --levels step:0.1 --contrast 0.7
```

Then open the output in PrusaSlicer and slice as usual. `--dry-run` prints the report without
writing a file; `--preview` writes a PNG with three orthographic views; `--ply` writes the mesh
with per-face colours for MeshLab or Blender.

## How it works

1. **Target** – every triangle painted with extruder `-e`. With `-e all` every virtual
   extruder (id above the physical extruders) that has painted area is processed in turn,
   largest area first; pieces created by an earlier extruder are never broken again.
2. **Cells** – Poisson-disk seeds on the surface and a power diagram (Voronoi with random
   weights → varying sizes) computed in a *warped* space (smooth random displacement → wobbly
   borders) with the z axis shrunk (cells taller than wide → more layers per patch, fewer
   horizontal borders).
3. **Tones** – each cell gets `z = ρ·G + √(1−ρ²)·ε`: a smooth multi-octave Gaussian field
   `G` (large-scale mottling) mixed with per-cell blue noise `ε`. `--coherence` = ρ² is the
   correlation between neighbouring cells. Three colours use two such fields.
4. **Level distribution** – a Beta (2 colours) or Dirichlet (3 colours) prior with the target
   mean and the requested `--spread`, discretised onto the available levels and exponentially
   tilted so the mean is *exact*. Cells are cut at area-weighted quantiles of the tone
   (hierarchically along the two main colour directions for three colours), so the realised
   area fractions match.
5. **Contrast (optional)** – `--contrast` swaps levels between cells so that neighbouring
   patches print out of phase (below). Swaps keep the area fractions, hence mean and spread.
6. **Printability cleanup** – patches smaller than `--min-area` or thinner than `--min-width`
   are merged into a neighbour, choosing the neighbour that keeps the mean on target; then
   whole patches are moved to a nearby level until the mean is within `--tolerance`.
7. **Write** the 3mf and print a report per extruder.

### Nominal vs printed ("effective") ratio

PrusaSlicer prints a blend with the shortest layer cycle `n ≤ 64` whose counts `round(r_i·n)`
approximate the ratios within 0.03. Hence 15/85 prints as **1 layer in 6 (16.7 %)**, 10/90 as
1 in 8 (12.5 %), 5/95 as 1 in 13 (7.7 %), 35/15/50 as 2+1+3 in 6. The cycle is indexed by the
global layer number, so two regions with the *same* virtual extruder are perfectly in phase
(that is what makes the stripes so regular), while regions with different cycles are out of
phase. `colorbreak inspect` prints the cycle of every virtual extruder.

By default ColorBreak matches the **printed** mean of the source extruder (`--ratio-mode
effective`), i.e. the result is meant to look like the original ColorMix. `--ratio-mode
nominal` matches the nominal ratios instead. `--target-ratio` asks for a different mix: one
value (fraction of the first filament) for two colours, or `a,b,c` in the order of the
filaments (`--components` order, else ascending extruder id).

### Contrast: neighbours out of phase

The stripes of two adjacent patches line up when their layer cycles coincide: same level
(identical cycle), or cycles where one period divides the other. For every pair of levels
ColorBreak computes the fraction of layers in which both print the *same filament* (over the
common period), subtracts what uncorrelated cycles would give, and adds a small bonus for
equal levels (equal neighbours would also merge into one big patch). This is the
**alignment kernel**; the **aligned border fraction** (border length weighted by the kernel,
divided by total border length) is the figure reported before/after.

`--contrast c` runs simulated annealing over swaps of two similar-area cells: `c = 1` keeps
the fully annealed state, `c = 0.5` stops half-way between the random and the annealed value,
`0` is off. In `-e all` mode, patches of earlier extruders and the original paint around the
region are fixed neighbours that enter the energy. Coherence and contrast pull in opposite
directions: coherent tones mean similar neighbours. A good middle ground is `--coherence 0.3
--contrast 0.7`. If one level holds more than half of the area it cannot avoid touching
itself; raise `--spread` or add levels.

### Short patches cannot print long cycles

A patch `h` layers tall painted with a cycle of `n` layers receives ⌊h/n⌋ or ⌈h/n⌉ layers of a
colour depending on where the patch sits relative to the global cycle. The global mean is
still right (in expectation), but a 12-layer patch with a 1-in-13 level is either all second
colour or has one light layer. ColorBreak therefore drops levels whose cycle is longer than
the typical patch height (`--max-cycle auto`; patch height ≈ `patch-size / z-stretch /
layer-height`, 19 layers with the defaults) and reports the extra per-patch randomness this
phase effect adds. `--max-cycle none` keeps all levels; `--max-cycle 8` sets a limit.

Levels that print identically (e.g. 17.5 % and 20 % are both "1 layer in 5") are merged into
one before the distribution is fitted, so finer `--levels step:` values only help where the
cycle quantisation actually produces a new printed ratio.

### Levels for three colours

`--levels auto` uses the existing virtual extruders whose filaments are a subset of the
source's (3-colour blends, the 2-colour blends of each pair) plus the three pure colours. If
the project has few blends near the target, most of the area stays on the source level; use
`--levels step:0.1` to let ColorBreak create the missing blends on a 10 % grid. Existing
blends are kept, only the new blends that are actually used are added to the project, and
their display colours are fitted from the existing blends.

## Parameters (`colorbreak break --help`)

| group | option | default | meaning |
|---|---|---|---|
| target | `-e/--extruder` | | extruder to break up, or `all` |
| | `--components A,B[,C]` | from `-e` | physical extruders of the blend (needed only if `-e` is a physical extruder) |
| | `--target-ratio` | source ratio | nominal ratio to aim for (one value, or one per filament) |
| | `--ratio-mode` | `effective` | match printed (`effective`) or `nominal` ratios |
| look | `--spread` / `--preset` | 0.08 / `medium` | spread of patch ratios; `subtle`=0.05, `medium`=0.08, `bold`=0.12 |
| | `--patch-size` | 2.5 mm | typical patch width |
| | `--z-stretch` | 0.65 | patches are `1/z-stretch` × taller than wide |
| | `--irregularity` | 0.35 | wobbliness of patch borders (0 = Voronoi polygons) |
| | `--size-jitter` | 0.5 | variation of patch sizes |
| | `--coherence` | 0.5 | correlation of neighbouring patch tones (0 speckle … 1 smooth mottling) |
| | `--mottle-scale` | 4 × patch size | length scale of the large-scale tone variation |
| | `--contrast` | 0 | 0..1, make neighbouring patches print out of phase |
| | `--levels` | `auto` | `auto`, `step:X` (simplex grid) or explicit `a/b/c,...` / `r1,r2,...` |
| | `--no-pure` | | do not use the pure filaments as levels |
| | `--max-cycle` | `auto` | drop levels with longer layer cycles (`none` to keep all) |
| | `--seed` | 0 | random seed (same inputs + seed = same result) |
| printability | `--min-area` | 3 mm² | merge smaller patches |
| | `--min-width` | 1.1 mm | merge thinner patches (≈ 2 extrusion widths + margin) |
| | `--tolerance` | 1e-4 | allowed deviation of the mean ratio |
| output | `--preview file.png` | | three orthographic views of the whole model |
| | `--preview-scale` | 14 | preview pixels per mm |
| | `--ply file.ply` | | mesh with per-face colours (MeshLab, Blender) |
| | `--dry-run` | | report only |

## Reading the report

```
Result for extruder 5 (3057 mm^2, 296 patches), filaments (1, 2, 4):
  extruder            nominal             prints cycle  target%   area%
         5          35/15/50%  0.333/0.167/0.500     6    23.1%   23.4%
        14          45/10/45%  0.444/0.111/0.444     9    10.2%   10.0%
  ...
  area-weighted mean: nominal ..., printed 0.3333/0.1667/0.5000   (target effective ...)
  spread of patch ratios (printed): 0.0817
  extra per-patch randomness from layer phase (~19 layers per patch): 0.029
  aligned border fraction: 0.134 -> 0.035
  patch area  mm^2  5%/50%/95%: 4.6 / 8.9 / 27.1
  patch width mm    5%/50%/95%: 1.38 / 2.16 / 4.54
```

`target%` is the planned area share per level, `area%` the realised one. The spread is
`sqrt(Σ_i Var(r_i) / 2)`, which for two colours is simply the standard deviation of the
first filament's fraction across patches.

## Tuning hints

* Stripes still visible → more `--spread` (or `--preset bold`), smaller `--patch-size`, or
  `--contrast 0.7`.
* Looks noisy / salt-and-pepper → lower `--spread`, raise `--coherence` (≤ 0.7), lower
  `--contrast`, raise `--patch-size`.
* Most area stays on the source level (three colours) → `--levels step:0.1`.
* Print problems at patch borders → raise `--min-area` / `--min-width` / `--patch-size`.
* `colorbreak inspect` shows per-extruder areas and the printed cycle of every blend; it is
  also a quick way to check hand-painted files.

## Limitations

* Blends of up to three filaments (PrusaSlicer's limit).
* The extruder to break must be *painted*; a volume whose default extruder is a ColorMix
  without painted triangles is not handled.
* Previews are software-rendered point clouds, good enough to judge patch size and
  distribution, not a rendering.
* Patch size thresholds (3 mm², 1.1 mm) are reasonable guesses for a 0.4 mm nozzle; they
  have not been validated on many prints. Open the output in PrusaSlicer and check the
  slice before printing.


## Project layout

```
colorbreak/
  threemf.py     3mf reading/writing, mesh parsing, transforms, virtual extruder config
  paintcodes.py  PrusaSlicer per-triangle painting codec (bitstream ↔ hex string)
  levels.py      levels, printed-ratio quantisation (port of PrusaSlicer), colours, ids
  field.py       Gaussian random fields (random Fourier features)
  patches.py     Poisson-disk seeds, warped power diagram → cells
  assign.py      Beta/Dirichlet priors, tilting, quantile assignment
  contrast.py    layer-phase alignment kernel and simulated annealing
  cleanup.py     adjacency, patch statistics, merging of small patches, mean repair
  preview.py     PNG and PLY previews without extra dependencies
  cli.py         command line interface
```

## Acknowledgements

The painting codec and the blend quantisation replicate the behaviour of
[PrusaSlicer](https://github.com/prusa3d/PrusaSlicer) (`TriangleSelector::serialize`,
`FacetsAnnotation::get_triangle_as_string`, `quantise_component_counts`,
`build_canonical_cycle`). PrusaSlicer is not required to run ColorBreak, only to slice the
result.

## License

Public domain ([Unlicense](LICENSE)): do whatever you like with it, no attribution needed.
