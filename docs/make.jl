using Documenter, Example, Cloudy

makedocs(
  sitename="Docs for cloudy",
  doctest = false,
  strict = false,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",),
  clean = false,
  pages = Any[
    "Home" => "index.md",
    "KernelTensors" => "KernelTensors.md",
    "ParticleDistributions" => "ParticleDistributions.md",
    "Sources" => "Sources.md"
  ]
)

deploydocs(
           repo = "github.com/CliMA/cloudy.git",
           target = "build",
          )

