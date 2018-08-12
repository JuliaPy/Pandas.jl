using Pandas
using Base.Test

df = DataFrame(Dict(:name=>["a", "b"], :age=>[27, 30]))
age = values(df[:age])
age[2] = 31
@test loc(df)[1, "age"] == 31

query(df, :(age!=27))  # Issue #26

df = read_csv(joinpath(dirname(@__FILE__), "test.csv"))
typeof(df)
@test isa(df, Pandas.DataFrame)

include("test_tabletraits.jl")

@test !isempty(df)
empty!(df)
@test isempty(df)

x, y = [1, 2, 3], [2, 3, 4]
df = DataFrame(Dict(:x => x, :y => y, :z => y), index = ["a", "b", "c"])
@test values(loc(df)[:, "x"]) == values(iloc(df)[:, 1])
@test values(loc(df)[:, ["x", "y"]]) == values(iloc(df)[:, 1:2])
@test values(loc(df)["a", :]) == values(iloc(df)[1, :])
@test Array(loc(df)[["a", "b"], :]) == Array(iloc(df)[1:2, :])
@test values(loc(df)[:, :]) == values(iloc(df)[:, :])
for op in [:+, *, :/, :-, :(==), :!=, :>, :<, :>=, :<=, :&, :|]
    @eval @test values($op(df[:x], 2)) == broadcast($op, x, 2)
    @eval @test values($op(2, df[:x])) == broadcast($op, 2, x)
    @eval @test values($op(df[:x], df[:y])) == broadcast($op, x, y)
    if op in (:<, :>, :<=, :>=)
        @eval @test values($op(2, df[:x]) & $op(df[:x], 3)) == 
                    broadcast(z -> $op(2, z) & $op(z, 3), x)
    end
end
