using Pandas
using Base.Test

df = DataFrame(Dict(:name=>["a", "b"], :age=>[27, 30]))
age = values(df[:age])
age[2] = 31
@test loc(df)[1, "age"] == 31
