"Testing correctness of KernelTensors module."

using Cloudy.KernelTensors


# constant coalescence tensor
ker = ConstantCoalescenceTensor(FT(π))
@test ker.r  == 0
@test ker.c == Array{FT}([π])

c = FT(-1)
@test_throws Exception ConstantCoalescenceTensor(c)

# linear coalescence tensor
c = Array{FT}([[1, 0] [0, 1]])
ker = LinearCoalescenceTensor(c)
@test ker.r  == 1
@test ker.c == Array{FT}([[1, 0] [0, 1]])

c = Array{FT}([[34, 67] [65, 54]])
@test_throws Exception LinearCoalescenceTensor(c)

c = Array{FT}([[-34, 67] [67, 54]])
@test_throws Exception LinearCoalescenceTensor(c)

c = Array{FT}([[-34, 67] [-34, 67]])
@test_throws Exception LinearCoalescenceTensor(c)
