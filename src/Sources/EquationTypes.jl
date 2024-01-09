"""
Types for dispatching on different equation sets for sources
"""

module EquationTypes

export AbstractStyle
export CoalescenceStyle
export AnalyticalCoalStyle
export NumericalCoalStyle

abstract type AbstractStyle end
abstract type CoalescenceStyle <: AbstractStyle end
struct NumericalCoalStyle <: CoalescenceStyle end
struct AnalyticalCoalStyle <: CoalescenceStyle end

end #module EquationTypes.jl
