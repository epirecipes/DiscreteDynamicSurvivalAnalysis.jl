# Example data

The line lists analysed in the DDSA manuscript, included so that its analyses can be reproduced.
`example_data_path(name)` returns the path of a file. The examples read the files `using CSV, DataFrames`, then
bin them with `linelist`.

## `WSU.csv`: 2009 H1N1 influenza, Washington State University (`:wsu`)

- **Source:** `WSU.csv` in the SDS.Epidemic repository of KhudaBukhsh et al.,
  <https://github.com/cbskust/SDS.Epidemic>.
- **Contents:** 2,276 cases with columns `V1` (infection time, days) and `V2` (recovery time,
  days). The original data were daily counts of new cases. The individual infection times were
  reconstructed from those counts, and the recovery times were simulated from an exponential
  distribution with rate 1/5.5 per day.
- **Used in:** `docs/src/vignettes/h1n1_wsu.jl`:

  ```julia
  df = CSV.read(example_data_path(:wsu), DataFrame)
  data = linelist(df.V1, df.V2; nsteps = 104, censor_at = 104.0)
  ```

## `EVD_Wave1.csv`, `EVD_Wave2.csv`, `EVD_Wave3.csv`: Ebola virus disease, DRC 2018–2020 (`:ebola_wave1`–`3`)

- **Source:** the Zenodo record "wasiur/DSAofEbola: Dynamic Survival Analysis (DSA)", version
  v.2.0.0, doi [10.5281/zenodo.6104188](https://doi.org/10.5281/zenodo.6104188).
- **Contents:** 907, 1,104 and 1,069 cases from the three waves of the epidemic.
  The columns are:
  - `onset`: symptom onset, in days;
  - `hosp`: hospitalisation, in days;
  - `event1`: 1 if the onset is observed;
  - `event2`: 1 if the hospitalisation is observed.

  When a flag is 0, the corresponding time column repeats the other time.
- **Used in:** `docs/src/vignettes/ebola_drc.jl`, one line list per wave. The time of an
  unobserved event is set to `missing`:

  ```julia
  df = CSV.read(example_data_path(:ebola_wave1), DataFrame)   # :ebola_wave2, :ebola_wave3
  onset = ifelse.(df.event1 .== 1, df.onset, missing)
  hosp = ifelse.(df.event2 .== 1, df.hosp, missing)
  data = linelist(onset, hosp)
  ```

## `hagelloch.csv`: measles, Hagelloch 1861 (`:hagelloch`)

- **Source:** the R package `outbreaks`, data set `measles_hagelloch_1861`
  (<https://CRAN.R-project.org/package=outbreaks>). It was formatted from `hagelloch.df` in the R
  package `surveillance`
  (<https://surveillance.r-forge.r-project.org/pkgdown/reference/hagelloch.html>). The data were
  collected by Pfeilsticker (1863) and Oesterle (1992).
- **Contents:** 188 cases.
  - The 12 columns of `outbreaks`: `case_ID`, `infector`, `date_of_prodrome`, `date_of_rash`,
    `date_of_death`, `age`, `gender`, `family_ID`, `class`, `complications`, `x_loc`, `y_loc`.
  - Four columns added for the manuscript:
    - `t_infection`: prodrome onset, in days since 30 October 1861;
    - `t_rash`: rash onset, on the same scale;
    - `inf_period`: `t_rash − t_infection + 4`;
    - `t_recovery`: `t_infection + inf_period`.
  - `class` is 0 = not yet at school (90 cases), 1 = younger class (30), 2 = older class (68).
- **Used in:** `docs/src/vignettes/measles_hagelloch.jl`, one line list per class on a shared grid:

  ```julia
  df = CSV.read(example_data_path(:hagelloch), DataFrame)
  cls = parse.(Int, string.(df.class))        # the column may parse as Int or String
  classes = [df[cls .== c, :] for c in 0:2]
  nsteps = maximum(df.t_infection) + 2 * round(Int, mean(df.inf_period))   # 102 days
  groups = [linelist(d.t_infection, d.t_infection .+ d.inf_period; nsteps = nsteps) for d in classes]
  ```
