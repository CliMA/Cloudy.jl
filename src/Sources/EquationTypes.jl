"""
Types for dispatching on different equation sets for sources
"""

module EquationTypes

export AbstractStyle
export CoalescenceStyle
export OneModeCoalStyle
export TwoModesCoalStyle

abstract type AbstractStyle end
abstract type CoalescenceStyle <: AbstractStyle end
struct OneModeCoalStyle <: CoalescenceStyle end
struct TwoModesCoalStyle <: CoalescenceStyle end

end #module EquationTypes.jl