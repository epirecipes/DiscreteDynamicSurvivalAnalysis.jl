# Paths to the line lists distributed with the package (see data/README.md for provenance).

const EXAMPLE_DATA = Dict(
    :wsu         => "WSU.csv",
    :ebola_wave1 => "EVD_Wave1.csv",
    :ebola_wave2 => "EVD_Wave2.csv",
    :ebola_wave3 => "EVD_Wave3.csv",
    :hagelloch   => "hagelloch.csv",
)

"""
    example_data_path(name::Symbol) -> String

Path of a line list distributed with the package (only the path is returned; nothing is read).

- `:wsu`: 2009 H1N1 influenza at Washington State University (columns `V1` infection time and
  `V2` recovery time, in days; 2,276 cases).
- `:ebola_wave1`, `:ebola_wave2`, `:ebola_wave3`: the three waves of the 2018–2020 Ebola epidemic
  in the DRC analysed in the manuscript (columns `onset`, `hosp` in days, and the flags `event1`,
  `event2` marking an observed onset / hospitalisation; when a flag is 0 the corresponding column
  repeats the other time).
- `:hagelloch`: the 1861 Hagelloch measles outbreak (188 cases; `class`, `t_infection`,
  `inf_period`, … as in the R package `outbreaks`).

The sources and contents of the files are described in `data/README.md` of the package.
"""
function example_data_path(name::Symbol)
    haskey(EXAMPLE_DATA, name) ||
        throw(ArgumentError("unknown example data set :$name; available: $(sort(collect(keys(EXAMPLE_DATA))))"))
    return joinpath(pkgdir(@__MODULE__), "data", EXAMPLE_DATA[name])
end
