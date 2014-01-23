using PyCall
using PyPlot

import Base.getindex, Base.setindex!, Base.length, Base.size, Base.mean, Base.std, Base.show, Base.merge, Base.convert, Base.hist, Base.join
import PyPlot.plot

@pyimport pandas
np = pyimport("numpy")

type_map = Dict()

abstract PyWrapped

convert(::Type{PyObject}, x::PyWrapped) = x.pyo
PyCall.PyObject(x::PyWrapped) = x.pyo

macro pytype(name, class)
    quote
        immutable $(name) <: PyWrapped
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            function $(esc(name))(args...; kwargs...)
                pandas_method = pandas.$name
                new(pandas_method(args...; kwargs...))
            end
        end
        type_map[$class] = $name
    end
end

quot(x) = Expr(:quote, x)

function Base.Array(x::PyWrapped)
    c = np[:asarray](x)
    if typeof(c).parameters[1] == PyObject
        out = cell(size(x))
        for row=1:size(x,1)
            for col=1:size(x,2)
                out[row, col] = convert(PyAny, c[row, col])
            end
        end
        out
    else
        c
    end
end


function pandas_wrap(pyo::PyObject)
    for pyt in keys(type_map)
        if pyisinstance(pyo, pyt)
            return type_map[pyt](pyo)
        end
    end
    return convert(PyAny, pyo)
end

pandas_wrap(pyo) = pyo

fix_arg(x::Range1) = pyeval(@sprintf "slice(%d, %d)" x.start x.start+x.len)
fix_arg(x::Range) = pyeval(@sprintf "slice(%d, %d, %d)" x.start x.start+x.len*x.step x.step)
fix_arg(x) = x

function pyattr(class, method, orig_method)
    if orig_method == :nothing
        m_quote = quot(method)
    else
        m_quote = quot(orig_method)
    end
    quote
        function $(esc(method))(pyt::$class, args...; kwargs...)
            pyo = pyt.pyo[string($m_quote)]
            if pytype_query(pyo) == Function
                new_args = [fix_arg(arg) for arg in args]
                pyo = pyt.pyo[$m_quote](new_args...; kwargs...)
            else
                pyo = pyt.pyo[string($m_quote)]
            end

            wrapped = pandas_wrap(pyo)
        end
    end
end


function delegate(new_func, orig_func, escape=false)
    @eval begin
        function $(escape ? esc(new_func) : new_func)(args...; kwargs...)
            f = $(orig_func)
            pyo = f(args...; kwargs...)
            pandas_wrap(pyo)
        end
    end
end

macro delegate(new_func, orig_func)
    delegate(new_func, orig_func, true)
end

macro pyattr(class, method, orig_method)
    pyattr(class, method, orig_method)
end

macro df_pyattrs(methods...)
    classes = [:DataFrame, :Series]
    m_exprs = Expr[]
    for method in methods
        exprs = Array(Expr, length(classes))
        for (i, class) in enumerate(classes)
            exprs[i] = pyattr(class, method, :nothing)
        end
        push!(m_exprs, Expr(:block, exprs...))
    end
    Expr(:block, m_exprs...)
end

macro gb_pyattrs(methods...)
    classes = [:GroupBy]
    m_exprs = Expr[]
    for method in methods
        exprs = Array(Expr, length(classes))
        for (i, class) in enumerate(classes)
            exprs[i] = pyattr(class, method, :nothing)
        end
        push!(m_exprs, Expr(:block, exprs...))
    end
    Expr(:block, m_exprs...)
end

macro pyasvec(class, offset)
    offset = eval(offset)
    if offset
        index_expr = quote 
            function $(esc(:getindex))(pyt::$class, args...)
                new_args = tuple([fix_arg(arg-1) for arg in args]...)
                pyo = pyt.pyo[:__getitem__](length(new_args)==1 ? new_args[1] : new_args)
                pandas_wrap(pyo)
            end

            function $(esc(:setindex!))(pyt::$class, value, idx)
                new_idx = fig_arg(idx-1)
                pyt.pyo[:__setitem__](tuple(new_idx...), value)
            end
        end
    else
        index_expr = quote  
            @pyattr $class getindex __getitem__ 
            function $(esc(:setindex!))(pyt::$class, value, idx)
                pyt.pyo[:__setitem__](fix_arg(idx), value)
            end            
        end
    end
    quote
        $index_expr
        @pyattr $class length __len__ 
    end
end


@pytype DataFrame pandas.core[:frame]["DataFrame"]
@pytype Iloc pandas.core[:indexing]["_iLocIndexer"]
@pytype Loc pandas.core[:indexing]["_LocIndexer"]
@pytype Ix pandas.core[:indexing]["_IXIndexer"]
@pytype Series pandas.core[:series]["Series"]
@pytype Index pandas.core[:index]["Index"]
@pytype GroupBy pandas.core[:groupby]["DataFrameGroupBy"]

@pyattr DataFrame groupby nothing
@pyattr DataFrame columns nothing
@pyattr DataFrame query nothing
@pyattr GroupBy app apply 

@gb_pyattrs mean std agg aggregate median var ohlc transform groups indices get_group

@df_pyattrs iloc loc reset_index index head xs to_csv to_pickle plot hist join align drop drop_duplicates duplicated filter first idxmax idxmin last reindex reindex_axis reindex_like rename tail set_index select take truncate abs any clip clip_lower clip_upper corr corrwith count cov cummax cummin cumprod cumsum describe diff mean median min mode pct_change rank quantile sum skew var std dropna fillna replace delevel pivot reodrer_levels sort sort_index sortlevel swaplevel stack unstack T boxplot

Base.size(df::PyWrapped, i::Integer) = size(df)[i]
Base.size(df::PyWrapped) = df.pyo[:shape]

@pyasvec Series true
@pyasvec Loc false
@pyasvec Ix false
@pyasvec Iloc true
@pyasvec DataFrame false
@pyasvec Index true

Base.ndims(df::Union(DataFrame, Series)) = length(size(df))


for m in [:read_csv, :read_html, :read_json, :save, :stats,  :melt, :rolling_count, :rolling_sum, :rolling_window, :rolling_quantile, :ewma]
    delegate(m, quote pandas.$m end)
end

function show(io::IO, df::DataFrame)
    if length(df)>10
        show(io, head(df))
        return
    end
    idx = index(df)
    cols = columns(df)
    loc_ = loc(df)
    print(io, "\t")
    n_cols = length(cols)
    n_rows = length(idx)
    for i=1:n_cols
        print(io, cols[i], "\t")
    end
    println(io)
    for i=1:n_rows
        print(io, idx[i], "\t")
        for j=1:n_cols
            print(io, loc_[(idx[i], cols[j])], "\t")
        end
        println(io)
    end
end

function show(io::IO, series::Series)
    if length(series)>10
        show(io, head(series))
        return
    end
    idx = index(series)
    N = length(idx)
    loc_ = loc(series)
    for n=1:N
        print(io, idx[n], "\t")
        print(io, loc_[idx[n]])
        println(io)
    end
end
