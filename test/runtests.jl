using Pandas
using Test
import DataFrames

df = DataFrame(Dict(:name=>["a", "b"], :age=>[27, 30]))
age = values(df.age)
age[2] = 31
@test loc(df)[1, "age"] == 31

query(df, :(age!=27))  # Issue #26

text = repr(MIME("text/html"), df)
@test text isa String
@test occursin("<table", text)
@test occursin("age", text)

df = read_csv(joinpath(dirname(@__FILE__), "test.csv"))
typeof(df)
@test isa(df, Pandas.DataFrame)

include("test_tabletraits.jl")
include("test_tables.jl")

@test !isempty(df)
empty!(df)
@test isempty(df)

x = Series([3,5], index=[:a, :b])

@test x.a == 3
@test x[:a] == 3
@test loc(x)[:a] == 3
@test x.b == 5
@test iloc(x)[1] == 3
@test iloc(x)[2] == 5
@test length(x) == 2
@test values(x+1) == [4, 6]
@test sum(x) == 8
if !Sys.iswindows()
    @test eltype(x) == Int64
end
@test all(iloc(x)[1:2]==x)

# Rolling
roll = rolling(Series([1,2,3,4,5]), 3)
@test isequal(values(mean(roll)), [NaN, NaN, 2.0, 3.0, 4.0])

# Issue #71
julia_df = DataFrames.DataFrame(x=[1,2], y=[missing, missing])
py_df = Pandas.DataFrame(julia_df)
expected_df = Pandas.DataFrame(:x=>[1,2], :y=>[NaN, NaN])
@test Pandas.equals(py_df, expected_df)
