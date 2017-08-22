using Pandas
using NamedTuples
using DataValues
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

table_array2 = [@NT(a=1, b=DataValue("John"), c=3.2), @NT(a=2, b=DataValue("Sally"), c=5.8)]

@test_throws ArgumentError DataFrame(table_array2)

table_array3 = [@NT(a=DataValue{Int}(), b="John", c=DataValue(3.2)), @NT(a=DataValue(2), b="Sally", c=DataValue{Float64}())]

df3 = DataFrame(table_array3)

it3_collected = collect(TableTraits.getiterator(df3))

@test length(it3_collected) == 2
@test isnan(it3_collected[1].a)
@test it3_collected[1].b == "John"
@test it3_collected[1].c == 3.2
@test it3_collected[2].a == 2
@test it3_collected[2].b == "Sally"
@test isnan(it3_collected[2].c)

end
