using Documenter, Example, Cloudy

makedocs(
  sitename="Docs for cloudy",
  doctest = false,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",),
  clean = false,
  pages = Any[
    "Home" => "index.md",
    "KernelTensors" => "KernelTensors.md",
    "ParticleDistributions" => "ParticleDistributions.md",
    "Coalescence" => "Coalescence.md",
    "Sedimentation" => "Sedimentation.md"
  ]
)

deploydocs(
           repo = "github.com/CliMA/cloudy.git",
           target = "build",
          )

