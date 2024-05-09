using Cloudy
using Cloudy.KernelFunctions
using Cloudy.KernelTensors
using Plots

function compute_kernel_from_tensor(c, x, y)
    n, m = size(c)
    output = 0.0
    for i in 1:n
        for j in 1:m
            output += c[i, j] * x^(i - 1) * y^(j - 1)
        end
    end
    return output
end


# HydrodynamicKernelFunction
FT = Float64
limit = FT(1e-6)
order = 5

kernel_func = HydrodynamicKernelFunction(1e2 * π)
kernel_tensor = CoalescenceTensor(kernel_func, order, limit)

n = 100
x = range(0, limit, n)
y = range(0, limit, n)
z_kernel = zeros(n, n)
z_tensor = zeros(n, n)
for i in 1:n
    for j in 1:n
        z_kernel[i, j] = i < j ? NaN : kernel_func(x[i], y[j])
        z_tensor[i, j] = i < j ? NaN : compute_kernel_from_tensor(kernel_tensor.c, x[i], y[j])
    end
end

p1 = contourf(x, y, z_kernel, xaxis = "x", yaxis = "y", title = "Kernel Function", colorbar_exponentformat="power")
p2 = contourf(x, y, z_tensor, xaxis = "x", yaxis = "y", title = "Kernel Tensor", colorbar_exponentformat="power")

plot(p1, p2, layout = (1, 2), size = (1200, 400), bottom_margin = 10Plots.mm, left_margin = 2Plots.mm, right_margin = 10Plots.mm, yformatter = :scientific, xformatter = :scientific)
path = joinpath(pkgdir(Cloudy), "test/outputs/")
mkpath(path)
savefig(path * "HydrodynamicKernelFunction_Approximation.pdf")

# LongKernelFunction
limit = FT(1e-9)
order = 2

kernel_func =  LongKernelFunction(5.236e-10, 9.44e9, 5.78) # 5.236e-10 kg; 9.44e9 m^3/kg^2/s; 5.78 m^3/kg/s
kernel_tensor = CoalescenceTensor(kernel_func, order, limit)

n = 100
x = range(0, limit, n)
y = range(0, limit, n)
z_kernel = zeros(n, n)
z_tensor = zeros(n, n)
for i in 1:n
    for j in 1:n
        z_kernel[i, j] = i < j ? NaN : kernel_func(x[i], y[j])
        z_tensor[i, j] = i < j ? NaN : compute_kernel_from_tensor(kernel_tensor.c, x[i], y[j])
    end
end

p1 = contourf(x, y, z_kernel, xaxis = "x", yaxis = "y", title = "Kernel Function", colorbar_exponentformat="power")
p2 = contourf(x, y, z_tensor, xaxis = "x", yaxis = "y", title = "Kernel Tensor", colorbar_exponentformat="power")

plot(p1, p2, layout = (1, 2), size = (1200, 400), bottom_margin = 10Plots.mm, left_margin = 2Plots.mm, right_margin = 10Plots.mm, yformatter = :scientific, xformatter = :scientific)
path = joinpath(pkgdir(Cloudy), "test/outputs/")
mkpath(path)
savefig(path * "LongKernelFunction_Approximation.pdf")