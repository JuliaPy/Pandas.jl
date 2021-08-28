using Tables

function _construct_pandas_from_tables(source)
    if(!Tables.istable(source))
        return nothing
    end
    source_columns = Tables.columns(source)
    source_as_dict = Dict(column => Tables.getcolumn(source_columns, column) for column in Tables.columnnames(source_columns))
    return invoke(DataFrame, Tuple{Vararg{Any}}, source_as_dict)
end

Tables.columnaccess(::DataFrame) = true
Tables.rowaccess(::DataFrame) = true
Tables.istable(::DataFrame) = true
