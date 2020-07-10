using DifferentialEquations
using Plots

function f(du,u,p,t)
    du[1] = u[2]
    du[2] = -p
end

# implement callbacks to halt the integration
function condition(m,t,integrator) 
    println("Condition checked")
    t>=1e-9
end

function affect!(integrator) 
    terminate!(integrator)
end

cb=DiscreteCallback(condition, affect!)

u0 = [50.0,0.0]
tspan = (0.0,15.0)
p = 9.8
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=cb)
plot(sol)