using Pandas
using Test
import DataFrames
using PyCall
using Dates

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
expected_df = Pandas.DataFrame(:x=>[1,2], :y=>[NaN, NaN])[["x", "y"]]
@test Pandas.equals(py_df, expected_df)

# Issue #68
py"""
import pandas as pd

def get_df():
    df = pd.DataFrame({
        "a":pd.to_datetime(["2021.01.15","2021.01.15","2020.04.06"])
    })
    return df
"""

py_df = py"get_df"()|>Pandas.DataFrame
julia_df = DataFrames.DataFrame(py_df)

@test julia_df.a == [DateTime(2021, 1, 15), DateTime(2021, 1, 15), DateTime(2020, 4, 6)]

# Issue #72
julia_df= DataFrames.DataFrame(C = 1:4, A = 5:8, B = 9:12)
py_df = Pandas.DataFrame(julia_df)
@test all(Pandas.columns(py_df) .== ["C","A","B"])

df1 = Pandas.Series(1:2)
df2 = Pandas.Series(1:2)
df3 = Pandas.Series(3:4)

@test all(df1 == df1)
@test all(df1 == df2)
@test df1 != [1, 2]

# Issue #93
df = DataFrame(:a=>[1,2], :b=>[4,5], :c=>["a","b"])
@test values(df) == [1 4 "a"; 2 5 "b"]
