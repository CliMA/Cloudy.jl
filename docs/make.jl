using Documenter, Example, Cloudy

Kernels = [
    "Kernel Functions" => "KernelFunctions.md",
    "Kernel Tensors" => "KernelTensors.md"
]

Coalescence = [
    "Coalescence" => "Coalescence.md",
    "EquationTypes" => "EquationTypes.md",
    "Kernels" => Kernels
]

Dynamics = [
    "Coalescence" => Coalescence,
    "Sedimentation" => "Sedimentation.md",
    "Condensation" => "Condensation.md"
]

mathengine = MathJax(
    Dict(
        :TeX => Dict(
            :equationNumbers => Dict(:autoNumber => "AMS"),
            :Macros => Dict(),
        ),
    ),
)

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
)

makedocs(
    sitename = "Cloudy.jl",
    format = format,
    clean = false,
    pages = Any[
        "Home" => "index.md",
        "API" => "API.md",
        "ParticleDistributions" => "ParticleDistributions.md",
        "Dynamics" => Dynamics,
        "Examples" => "Examples.md"
    ],
)

deploydocs(
    repo = "github.com/CliMA/cloudy.git", 
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
