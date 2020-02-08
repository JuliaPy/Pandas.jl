using Pandas
using Test

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
@test eltype(x) == Int64
@test all(iloc(x)[1:2]==x)

# Rolling
roll = rolling(Series([1,2,3,4,5]), 3)
@test isequal(values(mean(roll)), [NaN, NaN, 2.0, 3.0, 4.0])

# HDF
mktempdir() do dir
    store = HDFStore("$(dir)/store.h5")
    x = Series(["a", "b"])
    store["x"] = x
    x_fromstore = store["x"]
    @test values(x_fromstore) == ["a", "b"]
end
