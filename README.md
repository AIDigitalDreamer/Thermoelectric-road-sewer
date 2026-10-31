# Thermoelectric-road-sewer
Feasibility model and simulation code for thermoelectric energy harvesting from road/sewer waste heat — code accompanying a student research paper (mock/illustrative data).

# Thermoelectric Road-to-Sewer Energy Harvesting — Model Code

Code supporting the paper *"Thermoelectric Energy Harvesting from Urban Waste Heat"* by Kavish Shah.

- `teg_module.py` — the 1-D thermal-resistance model. Computes the capture fraction, the coupled road-to-sewer design's electrical output, and the module-area vs. sewer-flow-rate table (Tables 2-5 and every number in Sections 7-10 of the paper). No dependencies — run with:
  ```
  python3 teg_module.py
  ```

- `sim_2d.py` — a 2-D finite-difference check of the 1-D model, including a mesh-convergence study (Section 7.5). Needs `numpy` and `scipy`:
  ```
  pip install numpy scipy
  python3 sim_2d.py
  ```
  Takes about a minute to run.

In one line: heat moves from the hot road, through a thermoelectric module, down to the cooler sewage — soil resistance around the module limits how much of the available temperature difference actually reaches it, which is what most of this code is measuring.

