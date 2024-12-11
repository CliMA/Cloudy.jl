"""
Types for dispatching on different equation sets for sources
"""

module EquationTypes

export AbstractStyle
export CoalescenceStyle
export AnalyticalCoalStyle
export NumericalCoalStyle
export ThresholdStyle
export MovingThreshold
export FixedThreshold

abstract type AbstractStyle end
abstract type CoalescenceStyle <: AbstractStyle end
struct NumericalCoalStyle <: CoalescenceStyle end
struct AnalyticalCoalStyle <: CoalescenceStyle end

abstract type ThresholdStyle end
struct MovingThreshold <: ThresholdStyle end
struct FixedThreshold <: ThresholdStyle end

end #module EquationTypes.jl
