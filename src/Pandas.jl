__precompile__(true)
module Pandas

using PyCall
using PyPlot
using Lazy

import Base.getindex, Base.setindex!, Base.length, Base.size, Base.mean, Base.std, Base.show, Base.merge, Base.convert, Base.hist, Base.join, Base.replace, Base.endof, Base.start, Base.next, Base.done, Base.sum, Base.var

import Base: abs, any, count, cov, cummax, cummin, cumprod, cumsum, diff, drop, filter, first, indices, last, median, min, quantile, rank, select, sort, take, truncate

import Base: +, -, *, /, !

import PyPlot.plot

export DataFrame, Iloc, Loc, Ix, Series, MultiIndex, Index, GroupBy
export mean, std, agg, aggregate, median, var, ohlc, transform, groups, indices, get_group
export iloc,loc,reset_index,index,head,xs,plot,hist,join,align,drop,drop_duplicates,duplicated,filter,first,idxmax,idxmin,last,reindex,reindex_axis,reindex_like,rename,tail,set_index,select,take,truncate,abs,any,clip,clip_lower,clip_upper,corr,corrwith,count,cov,cummax,cummin,cumprod,cumsum,describe,diff,mean,median,min,mode,pct_change,rank,quantile,sum,skew,var,std,dropna,fillna,replace,delevel,pivot,reodrer_levels,sort,sort_index,sortlevel,swaplevel,stack,unstack,T, to_numeric, isin
export to_clipboard,to_csv,to_dense,to_dict,to_excel,to_gbq,to_hdf,to_html,to_json,to_latex,to_msgpack,to_panel,to_pickle,to_records,to_sparse,to_sql,to_string,query, groupby, columns, app, values, from_arrays, from_tuples
export read_csv, read_html, read_json, read_excel, read_table, save, stats,  melt, rolling_count, rolling_sum, rolling_window, rolling_quantile, ewma, setcolumns!, concat, read_pickle
export pivot_table, crosstab, cut, qcut, get_dummies, deletecolumn!, siz, name, setname!
export argsort,order,asfreq,asof,shift,first_valid_index,last_valid_index,weekday,resample,tz_conert,tz_localize
export resample,date_range,to_datetime,to_timedelta,bdate_range,period_range,ewma,ewmstd,ewmvar,ewmcorr,ewmcov
export rolling_count, expanding_count, rolling_sum, expanding_sum, rolling_mean, expanding_mean, rolling_median, expanding_median, rolling_var, expanding_var, rolling_std, expanding_std, rolling_min, expanding_min, rolling_max, expanding_max, rolling_corr, expanding_corr, rolling_corr_pairwise, expanding_corr_pairwise, rolling_cov, expanding_cov, rolling_skew, expanding_skew, rolling_kurt, expanding_kurt, rolling_apply, expanding_apply, rolling_quantile, expanding_quantile, index!, sample
export @>, @query

const np = PyNULL()
const pandas_raw = PyNULL()

function __init__()
    copy!(np, pyimport_conda("numpy", "numpy"))
    copy!(pandas_raw, pyimport_conda("pandas", "pandas"))
    for (pandas_expr, julia_type) in pre_type_map
        push!(type_map, (pandas_expr(), julia_type))
    end
end

const pre_type_map = []
const type_map = []

abstract PandasWrapped

convert(::Type{PyObject}, x::PandasWrapped) = x.pyo
PyCall.PyObject(x::PandasWrapped) = x.pyo

macro pytype(name, class)
    quote
        immutable $(name) <: PandasWrapped
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            function $(esc(name))(args...; kwargs...)
                pandas_method = ($class)()
                new(pandas_method(args...; kwargs...))
            end
        end
        $(esc(:start))(x::$name) = start(x.pyo)
        function $(esc(:next))(x::$name, state)
            new_val, new_state = next(x.pyo, state)
            return pandas_wrap(new_val), new_state
        end
        $(esc(:done))(x::$name, state) = done(x.pyo, state)
        push!(pre_type_map, ($class, $name))
    end
end

quot(x) = Expr(:quote, x)

function Base.Array(x::PandasWrapped)
    c = np[:asarray](x)
    if typeof(c).parameters[1] == PyObject
        out = Array{Any}(size(x))
        for idx in eachindex(out)
            out[idx] = convert(PyAny, c[idx])
        end
        out
    else
        c
    end
end

function Base.values(x::PandasWrapped)
    pyarray = convert(PyArray, x.pyo["values"])
    unsafe_wrap(Array, pyarray.data, size(pyarray))
end

function pandas_wrap(pyo::PyObject)
    for (pyt, pyv) in type_map
        if pyisinstance(pyo, pyt)
            return pyv(pyo)
        end
    end
    return convert(PyAny, pyo)
end

pandas_wrap(x::Union{AbstractArray, Tuple}) = [pandas_wrap(_) for _ in x]

pandas_wrap(pyo) = pyo

fix_arg(x::StepRange) = pyeval(@sprintf "slice(%d, %d, %d)" x.start x.start+length(x)*x.step x.step)
fix_arg(x::UnitRange) = fix_arg(StepRange(x.start, 1, x.stop))
fix_arg(x) = x

function fix_arg(x, offset)
    if offset
        fix_arg(x-1)
    else
        fix_arg(x)
    end
end

function pyattr(class, method, orig_method)
    if orig_method == :nothing
        m_quote = quot(method)
    else
        m_quote = quot(orig_method)
    end
    quote
        function $(esc(method))(pyt::$class, args...; kwargs...)
            pyo = pyt.pyo[string($m_quote)]
            if true
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

macro pyasvec(class)

    index_expr = quote
        function $(esc(:getindex))(pyt::$class, args...)
            offset = should_offset(pyt, args...)
            new_args = tuple([fix_arg(arg, offset) for arg in args]...)
            pyo = pyt.pyo[:__getitem__](length(new_args)==1 ? new_args[1] : new_args)
            pandas_wrap(pyo)
        end

        function $(esc(:setindex!))(pyt::$class, value, idxs...)
            offset = should_offset(pyt, args...)
            new_idx = [fix_arg(idx, offset) for idx in idxs]
            if length(new_idx) > 1
                pyt.pyo[:__setitem__](tuple(new_idx...), value)
            else
                pyt.pyo[:__setitem__](new_idx[1], value)
            end
        end
    end

    if class in [:Iloc, :Loc, :Ix]
        length_expr = quote
            function $(esc(:length))(x::$class)
                x.pyo[:obj][:__len__]()
            end
        end
    else
        length_expr = quote
            @pyattr $class length __len__
        end
    end
    quote
        $index_expr
        $length_expr
        function $(esc(:endof))(x::$class)
            length(x)
        end
    end
end


@pytype DataFrame ()->pandas_raw[:core][:frame]["DataFrame"]
@pytype Iloc ()->pandas_raw[:core][:indexing]["_iLocIndexer"]
@pytype Loc ()->pandas_raw[:core][:indexing]["_LocIndexer"]
@pytype Ix ()->pandas_raw[:core][:indexing]["_IXIndexer"]
@pytype Series ()->pandas_raw[:core][:series]["Series"]
@pytype MultiIndex ()->pandas_raw[:core][:index]["MultiIndex"]
@pytype Index ()->pandas_raw[:core][:index]["Index"]
@pytype GroupBy ()->pandas_raw[:core][:groupby]["DataFrameGroupBy"]
@pytype SeriesGroupBy ()->pandas_raw[:core][:groupby]["SeriesGroupBy"]


@pyattr DataFrame groupby nothing
@pyattr DataFrame columns nothing
@pyattr GroupBy app apply


@gb_pyattrs mean std agg aggregate median var ohlc transform groups indices get_group hist plot count

siz(gb::GroupBy) = gb.pyo[:size]()

function index(df::PandasWrapped)
    pandas_wrap(df.pyo[:index])
end

@df_pyattrs iloc loc reset_index  head xs plot hist join align drop drop_duplicates duplicated filter first idxmax idxmin last reindex reindex_axis reindex_like rename tail set_index select take truncate abs any clip clip_lower clip_upper corr corrwith count cov cummax cummin cumprod cumsum describe diff mean median min mode pct_change rank quantile sum skew var std dropna fillna replace delevel pivot reodrer_levels sort sort_index sortlevel swaplevel stack unstack T boxplot argsort order asfreq asof shift first_valid_index last_valid_index weekday resample tz_conert tz_localize isin sample to_clipboard to_csv to_dense to_dict to_excel to_gbq to_hdf to_html to_json to_latex to_msgpack to_panel to_pickle to_records to_sparse to_sql to_string query


Base.size(x::Union{Loc, Iloc, Ix}) = x.pyo[:obj][:shape]
Base.size(df::PandasWrapped, i::Integer) = size(df)[i]
Base.size(df::PandasWrapped) = df.pyo[:shape]

should_offset(::Any, args...) = false
should_offset(::Union{Iloc, Index}, args...) = true

function should_offset(s::Series, arg)
    if eltype(arg) == Int64
        if eltype(index(s)) ≠ Int64
            return true
        end
    end
    false
end

@pyasvec Series
@pyasvec Loc
@pyasvec Ix
@pyasvec Iloc
@pyasvec DataFrame
@pyasvec Index
@pyasvec GroupBy

Base.ndims(df::Union{DataFrame, Series}) = length(size(df))


for m in [:read_pickle, :read_csv, :read_html, :read_json, :read_excel, :read_table, :save, :stats,  :melt, :ewma, :concat, :merge, :pivot_table, :crosstab, :cut, :qcut, :get_dummies, :resample, :date_range, :to_datetime, :to_timedelta, :bdate_range, :period_range, :ewmstd, :ewmvar, :ewmcorr, :ewmcov, :rolling_count, :expanding_count, :rolling_sum, :expanding_sum, :rolling_mean, :expanding_mean, :rolling_median, :expanding_median, :rolling_var, :expanding_var, :rolling_std, :expanding_std, :rolling_min, :expanding_min, :rolling_max, :expanding_max, :rolling_corr, :expanding_corr, :rolling_corr_pairwise, :expanding_corr_pairwise, :rolling_cov, :expanding_cov, :rolling_skew, :expanding_skew, :rolling_kurt, :expanding_kurt, :rolling_apply, :expanding_apply, :rolling_quantile, :expanding_quantile, :rolling_window, :to_numeric]
    delegate(m, quote pandas_raw[$(quot(m))] end)
end

function show(io::IO, df::PandasWrapped)
    s = df.pyo[:__str__]()
    println(io, s)
end

function query(df::DataFrame, s::AbstractString)
    pandas_wrap(pyeval("df.query(s)", df=df, s=s))
end

function query(df::DataFrame, e::Expr) # This whole method is a terrible hack
    s = string(e)
    s = replace(s, "&&", "&")
    s = replace(s, "||", "|")
    s = replace(s, "∈", "==")
    s = replace(s, "!", "~")
    query(df, s)
end

macro query(df, e)
    quote
        query($(esc(df)), $(QuoteNode(e)))
    end
end

for m in [:from_arrays, :from_tuples]
    @eval function $m(args...; kwargs...)
        f = pandas_raw["MultiIndex"][string($(quot(m)))]
        res = pycall(f, PyObject, args...; kwargs...)
        pandas_wrap(res)
    end
end

for op in [(:+, :__add__), (:*, :__mul__), (:/, :__div__), (:-, :__sub__)]
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

for op in [(:-, :__neg__)]
    @eval begin
        $(op[1])(x::PandasWrapped) = pandas_wrap(x.pyo[$(quot(op[2]))]())
    end
end

function setcolumns!(df::PandasWrapped, new_columns)
    df.pyo[:__setattr__]("columns", new_columns)
end

function deletecolumn!(df::DataFrame, column)
    df.pyo[:__delitem__](column)
end

name(s::Series) = s.pyo[:name]
setname!(s::Series, name) = s.pyo[:name] = name

import Base: .==, .>, .<, .>=, .<=, .!=

for (op, pyop) in [(:.==, :__eq__), (:.>, :__gt__), (:.<, :__lt__), (:.>=, :__ge__), (:.<=, :__le__), (:.!=, :__ne__)]
    @eval function $op(s::PandasWrapped, x)
        pandas_wrap(s.pyo[$(QuoteNode(pyop))](x))
    end
end

function DataFrame(pairs::Pair...)
    DataFrame(Dict(pairs...))
end

function index!(df::PandasWrapped, new_index)
    df.pyo[:index] = new_index
    df
end

function Base.eltype(s::Series)
    dtype_map = Dict(
        np[:dtype]("int64") => Int64,
        np[:dtype]("float64") => Float64,
    )
    get(dtype_map, s.pyo[:dtype], Any)
end

function Base.map(f::Function, s::Series)
    if eltype(s) ∈ (Int64, Float64)
        Series([f(_) for _ in values(s)])
    else
        Series([f(_) for _ in s])
    end
end

function Base.map(x, s::Series; na_action=nothing)
    pandas_wrap(s.pyo[:map](x, na_action))
end

function Base.get(df::PandasWrapped, key, default)
    pandas_wrap(df.pyo[:get](key, default=default))
end

function Base.getindex(s::Series, c::CartesianIndex{1})
    s[c[1]]
end

function Base.copy(df::PandasWrapped)
    pandas_wrap(df.pyo[:copy]())
end

function !(df::PandasWrapped)
    pandas_wrap(df.pyo[:__neg__]())
end


end
