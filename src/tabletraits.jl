using TableTraitsUtils

TableTraits.isiterable(x::DataFrame) = true
TableTraits.isiterabletable(x::DataFrame) = true

function TableTraits.getiterator(df::DataFrame)
    col_names = [Symbol(i) for i in Pandas.columns(df)]

    column_data = [eltype(df[i])==String ? [df[i][j] for j=1:length(df)] : values(df[i]) for i in col_names]

    return create_tableiterator(column_data, col_names)
end

function _construct_pandas_from_iterabletable(source)
    columns, column_names = create_columns_from_iterabletable(source)
    cols = Dict(i[1]=>i[2] for i in zip(column_names, columns))
    return DataFrame(cols)
end
