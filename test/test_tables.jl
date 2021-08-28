using Pandas
using Test
using CSV
using Tables

@testset "tables" begin

file = IOBuffer("""Temp;Val;Gr
       20;7863;1
       100;7834;1
       200;7803;1""")
csv = CSV.File(file, types=[Float64, Float64, Int])
df = DataFrame(csv)
expected_df = DataFrame(:Val=>[7863.0, 7834.0, 7803.0], :Temp=>[20.0, 100.0, 200.0], :Gr=>[1,1,1])
@test equals(df, expected_df)

@test Tables.istable(df)
df_cols = Tables.columns(df)
@test Tables.getcolumn(df_cols, :Gr) == [1, 1, 1]
@test Tables.getcolumn(df_cols, :Val) == [7863.0, 7834.0, 7803.0]
@test Tables.getcolumn(df_cols, :Temp) == [20.0, 100.0, 200.0]

@test Tables.rowaccess(df)
@test Tables.columnaccess(df)
@test Tables.istable(df)

end
