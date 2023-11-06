"""
Types for dispatching on different equation sets for sources
"""

module EquationTypes

export AbstractStyle
export CoalescenceStyle
export AnalyticalCoalStyle
export NumericalCoalStyle
export HybridCoalStyle # TODO
export OneModeCoalStyle # TODO: combine with TwoMode to become "Analytical"
export TwoModesCoalStyle # TODO: combine with OneMode to become "Analytical"

abstract type AbstractStyle end
abstract type CoalescenceStyle <: AbstractStyle end
abstract type AnalyticalCoalStyle <: CoalescenceStyle end # TODO: should become a struct
struct NumericalCoalStyle <: CoalescenceStyle end
struct HybridCoalStyle <: CoalescenceStyle end # TODO: implement
struct OneModeCoalStyle <: AnalyticalCoalStyle end # TODO: remove
struct TwoModesCoalStyle <: AnalyticalCoalStyle end # TODO: remove

end #module EquationTypes.jl