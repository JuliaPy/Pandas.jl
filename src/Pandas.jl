module Pandas

using PyCall
using PyPlot

import Base.getindex, Base.setindex!, Base.length, Base.size, Base.mean, Base.std, Base.show, Base.merge, Base.convert, Base.hist, Base.join, Base.replace, Base.endof
import PyPlot.plot

export DataFrame, Iloc, Loc, Ix, Series, MultiIndex, Index, GroupBy
export mean, std, agg, aggregate, median, var, ohlc, transform, groups, indices, get_group
export iloc,loc,reset_index,index,head,xs,plot,hist,join,align,drop,drop_duplicates,duplicated,filter,first,idxmax,idxmin,last,reindex,reindex_axis,reindex_like,rename,tail,set_index,select,take,truncate,abs,any,clip,clip_lower,clip_upper,corr,corrwith,count,cov,cummax,cummin,cumprod,cumsum,describe,diff,mean,median,min,mode,pct_change,rank,quantile,sum,skew,var,std,dropna,fillna,replace,delevel,pivot,reodrer_levels,sort,sort_index,sortlevel,swaplevel,stack,unstack,T,boxplot
export to_clipboard,to_csv,to_dense,to_dict,to_excel,to_gbq,to_hdf,to_html,to_json,to_latex,to_msgpack,to_panel,to_pickle,to_records,to_sparse,to_sql,to_string,query, groupby, columns, app, values, from_arrays, from_tuples
export read_csv, read_html, read_json, read_excel, read_table, save, stats,  melt, rolling_count, rolling_sum, rolling_window, rolling_quantile, ewma, set_columns, concat


np = pyimport("numpy")
pandas_raw = pyimport("pandas")
pandas_mod = pywrap(pandas_raw)
type_map = {}

abstract PandasWrapped

convert(::Type{PyObject}, x::PandasWrapped) = x.pyo
PyCall.PyObject(x::PandasWrapped) = x.pyo

macro pytype(name, class)
    quote
        immutable $(name) <: PandasWrapped
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            function $(esc(name))(args...; kwargs...)
                pandas_method = pandas_mod.$name
                new(pandas_method(args...; kwargs...))
            end
        end
        push!(type_map, ($class, $name))
    end
end

quot(x) = Expr(:quote, x)

function Base.Array(x::PandasWrapped)
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

values(x::PandasWrapped) = convert(PyArray, x.pyo["values"])

function pandas_wrap(pyo::PyObject)
    for (pyt, pyv) in type_map
        if pyisinstance(pyo, pyt)
            return pyv(pyo)
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

macro df_pyattrs_eval(methods...)
    m_exprs = Expr[]
    for method in methods
        push!(m_exprs, quote 
            function $(esc(method))(df::PandasWrapped, arg)
                res = pyeval(@sprintf("df.%s('%s')", $method, arg), df=df)
                pandas_wrap(res)
            end
        end)
    end
    Expr(:block, m_exprs...)
end

macro gb_pyattrs(methods...)
    classes = [:GroupBy, :SeriesGroupBy]
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

            function $(esc(:setindex!))(pyt::$class, value, idxs...)
                new_idx = [fix_arg(idx-1) for idx in idxs]
                pyt.pyo[:__setitem__](tuple(new_idx...), value)
            end
        end
    else
        index_expr = quote  
            function $(esc(:getindex))(pyt::$class, args...)
                new_args = tuple([fix_arg(arg) for arg in args]...)
                pyo = pyt.pyo[:__getitem__](length(new_args)==1 ? new_args[1] : new_args)
                pandas_wrap(pyo)
            end

            function $(esc(:setindex!))(pyt::$class, value, idxs...)
                new_idx = [fix_arg(idx) for idx in idxs]
                if length(new_idx) > 1
                    pyt.pyo[:__setitem__](tuple(new_idx...), value)
                else
                    pyt.pyo[:__setitem__](new_idx[1], value)
                end
            end
        end
    end
    quote
        $index_expr
        @pyattr $class length __len__ 
        @pyattr $class endof __len__
    end
end


@pytype DataFrame pandas_mod.core[:frame]["DataFrame"]
@pytype Iloc pandas_mod.core[:indexing]["_iLocIndexer"]
@pytype Loc pandas_mod.core[:indexing]["_LocIndexer"]
@pytype Ix pandas_mod.core[:indexing]["_IXIndexer"]
@pytype Series pandas_mod.core[:series]["Series"]
@pytype MultiIndex pandas_mod.core[:index]["MultiIndex"]
@pytype Index pandas_mod.core[:index]["Index"]
@pytype GroupBy pandas_mod.core[:groupby]["DataFrameGroupBy"]
@pytype SeriesGroupBy pandas_mod.core[:groupby]["SeriesGroupBy"]


@pyattr DataFrame groupby nothing
@pyattr DataFrame columns nothing
@pyattr GroupBy app apply 


@gb_pyattrs mean std agg aggregate median var ohlc transform groups indices get_group hist plot

@df_pyattrs iloc loc reset_index index head xs plot hist join align drop drop_duplicates duplicated filter first idxmax idxmin last reindex reindex_axis reindex_like rename tail set_index select take truncate abs any clip clip_lower clip_upper corr corrwith count cov cummax cummin cumprod cumsum describe diff mean median min mode pct_change rank quantile sum skew var std dropna fillna replace delevel pivot reodrer_levels sort sort_index sortlevel swaplevel stack unstack T boxplot

@df_pyattrs_eval to_clipboard to_csv to_dense to_dict to_excel to_gbq to_hdf to_html to_json to_latex to_msgpack to_panel to_pickle to_records to_sparse to_sql to_string query

Base.size(df::PandasWrapped, i::Integer) = size(df)[i]
Base.size(df::PandasWrapped) = df.pyo[:shape]

@pyasvec Series false
@pyasvec Loc false
@pyasvec Ix false
@pyasvec Iloc true
@pyasvec DataFrame false
@pyasvec Index true
@pyasvec GroupBy false

Base.ndims(df::Union(DataFrame, Series)) = length(size(df))


for m in [:read_csv, :read_html, :read_json, :read_excel, :read_table, :save, :stats,  :melt, :rolling_count, :rolling_sum, :rolling_window, :rolling_quantile, :ewma, :concat, :merge]
    delegate(m, quote pandas_mod.$m end)
end

function show(io::IO, df::PandasWrapped)
    s = df.pyo[:__str__]()
    println(io, s)
end

function query(df::DataFrame, e::Expr) # This whole method is a terrible hack
    s = string(e)
    s = s[3:end-1] 
    s = replace(s, "&&", "&")
    s = replace(s, "||", "|")
    query(df, s)
end

for m in [:from_arrays, :from_tuples]
    @eval function $m(args...; kwargs...)
        f = pandas_raw["MultiIndex"][string($(quot(m)))]
        res = pycall(f, PyObject, args...; kwargs...)
        pandas_wrap(res)
    end
end

for op in [(:+, :__add__), (:*, :__mul__), (:/, :__div__)]
    @eval begin
        function $(op[1])(x::PandasWrapped, y::PandasWrapped)
            f = $(quot(op[2]))
            py_f = pyeval(string("x.", f), x=x.pyo)
            res = py_f(y)
            return pandas_wrap(res)
        end

        function $(op[1])(x::PandasWrapped, y)
            f = $(quot(op[2]))
            py_f = pyeval(string("x.", f), x=x.pyo)
            res = py_f(y)
            return pandas_wrap(res)
        end

        $(op[1])(y, x::PandasWrapped) = $(op[1])(x, y)
    end
end

function set_columns(df::PandasWrapped, new_columns)
    df.pyo[:__setattr__]("columns", new_columns)
end

end