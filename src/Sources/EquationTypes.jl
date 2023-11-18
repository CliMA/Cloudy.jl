"""
Types for dispatching on different equation sets for sources
"""

module EquationTypes

export AbstractStyle
export CoalescenceStyle
export AnalyticalCoalStyle
export NumericalCoalStyle
export HybridCoalStyle # TODO

abstract type AbstractStyle end
abstract type CoalescenceStyle <: AbstractStyle end
struct NumericalCoalStyle <: CoalescenceStyle end
struct AnalyticalCoalStyle <: CoalescenceStyle end
struct HybridCoalStyle <: CoalescenceStyle end # TODO: implement

end #module EquationTypes.jl