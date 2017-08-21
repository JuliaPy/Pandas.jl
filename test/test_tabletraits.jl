using Pandas
using NamedTuples
using Base.Test

@testset "TableTraits" begin

table_array = [@NT(a=1, b="John", c=3.2), @NT(a=2, b="Sally", c=5.8)]

df = DataFrame(table_array)

@test collect(columns(df)) == ["a", "b", "c"]
@test values(df[:a]) == [1,2]
@test values(df[:c]) == [3.2, 5.8]
@test [df[:b][i] for i in 1:2] == ["John", "Sally"]

@test TableTraits.isiterabletable(df) == true

it = TableTraits.getiterator(df)

@test eltype(it) == @NT(a::Int, b::String, c::Float64)

it_collected = collect(it)

@test eltype(it_collected) == @NT(a::Int, b::String, c::Float64)
@test length(it_collected) == 2
@test it_collected[1] == @NT(a=1, b="John", c=3.2)
@test it_collected[2] == @NT(a=2, b="Sally", c=5.8)

end
