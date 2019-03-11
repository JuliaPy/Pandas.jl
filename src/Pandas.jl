__precompile__(true)
module Pandas

using PyCall
using Lazy
using Compat
using TableTraits
using Statistics

import Base: getindex, setindex!, length, size, show, merge, convert,
 join, replace, lastindex, sum, abs, any, count,
 cumprod, cumsum, diff, filter, first, last,
 min, sort, truncate, +, -, *, /, !,
 ==, >, <, >=, <=, !=, &, |
import Statistics: mean, std, var, cov, median, quantile


include("exports.jl")

const np = PyNULL()
const pandas_raw = PyNULL()

function __init__()
    copy!(np, pyimport_conda("numpy", "numpy"))
    copy!(pandas_raw, pyimport_conda("pandas", "pandas"))
    for (pandas_expr, julia_type) in pre_type_map
        type_map[pandas_expr()] = julia_type
    end
end

const pre_type_map = []

# Maps a python object corresponding to a Pandas class to a Julia type which
# wraps that class.
const type_map = Dict{PyObject, Type}()

abstract type PandasWrapped end

PyCall.PyObject(x::PandasWrapped) = x.pyo

macro pytype(name, class)
    quote
        struct $(name) <: PandasWrapped
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            function $(esc(name))(args...; kwargs...)
                pandas_method = ($class)()
                new(pycall(pandas_method, PyObject, args...; kwargs...))
            end
        end

        # This won't work until PyCall is updated to support
        # the Julia 1.0 iteration protocol.
        function Base.iterate(x::$name, state...)
            res = Base.iterate(x.pyo, state...)
            if res === nothing
                return nothing
            else
                value, state = res
                return pandas_wrap(value), state
            end
        end

        push!(pre_type_map, ($class, $name))
    end
end

quot(x) = Expr(:quote, x)

function Base.Array(x::PandasWrapped)
    c = np.asarray(x.pyo)
    if typeof(c).parameters[1] == PyObject
        out = Array{Any}(undef, size(x))
        for idx in eachindex(out)
            out[idx] = convert(PyAny, c[idx])
        end
        out
    else
        c
    end
end


function Base.values(x::PandasWrapped)
    # Zero-copy conversion to a Julia native type is possible
    x_kind = x.pyo.dtype.kind
    if x_kind in ["i", "u", "f", "b"]
        pyarray = convert(PyArray, x.pyo."values")
        unsafe_wrap(Array, pyarray.data, size(pyarray))
    else  # Convert element by element otherwise
        collect(x)
    end
end

"""
    pandas_wrap(pyo::PyObject)

Wrap an instance of a Pandas python class in the Julia type which corresponds
to that class.
"""
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

fix_arg(x::StepRange) = py"slice($(x.start), $(x.start+length(x)*x.step), $(x.step))"
fix_arg(x::UnitRange) = fix_arg(StepRange(x.start, 1, x.stop))
fix_arg(x::Colon) = pybuiltin("slice")(nothing, nothing, nothing)
fix_arg(x) = x

function fix_arg(x, offset)
    if offset
        fix_arg(x .- 1)
    else
        fix_arg(x)
    end
end

fix_arg(x::Colon, offset) = pybuiltin("slice")(nothing, nothing, nothing)

pyattr(class, method) = pyattr(class, method, method)

function pyattr(class, jl_method, py_method)
    quote
        function $(esc(jl_method))(pyt::$class, args...; kwargs...)
            new_args = fix_arg.(args)
            method = pyt.pyo.$(string(py_method))
            pyo = pycall(method, PyObject, new_args...; kwargs...)
            wrapped = pandas_wrap(pyo)
        end
    end
end

macro pyattr(class, method)
    pyattr(class, method)
end

macro pyattr(class, method, orig_method)
    pyattr(class, method, orig_method)
end

"""
    pyattr_set(types, methods...)

For each Julia type `T<:PandasWrapped` in `types` and each method `m` in `methods`,
define a new function `m(t::T, args...)` that delegates to the underlying
pyobject wrapped by `t`.
"""
function pyattr_set(classes, methods...)
    for class in classes
        for method in methods
            @eval @pyattr($class, $method)
        end
    end
end

macro pyasvec(class)

    index_expr = quote
        function $(esc(:getindex))(pyt::$class, args...)
            offset = should_offset(pyt, args...)
            new_args = tuple([fix_arg(arg, offset) for arg in args]...)
            new_args = (length(new_args)==1 ? new_args[1] : new_args)
            pyo = pycall(pyt.pyo.__getitem__, PyObject, new_args)
            pandas_wrap(pyo)
        end

        function $(esc(:setindex!))(pyt::$class, value, idxs...)
            offset = should_offset(pyt, idxs...)
            new_idx = [fix_arg(idx, offset) for idx in idxs]
            if length(new_idx) > 1
                pandas_wrap(pycall(pyt.pyo.__setitem__, PyObject, tuple(new_idx...), value))
            else
                pandas_wrap(pycall(pyt.pyo.__setitem__, PyObject, new_idx[1], value))
            end
        end
    end

    if class in [:Iloc, :Loc, :Ix]
        length_expr = quote
            function $(esc(:length))(x::$class)
                x.pyo.obj.__len__()
            end
        end
    else
        length_expr = quote
            function $(esc(:length))(x::$class)
                x.pyo.__len__()
            end
        end
    end

    quote

        $index_expr
        $length_expr
        function $(esc(:lastindex))(x::$class)
            length(x)
        end
    end
end


@pytype DataFrame ()->pandas_raw.core.frame."DataFrame"
@pytype Iloc ()->pandas_raw.core.indexing."_iLocIndexer"
@pytype Loc ()->pandas_raw.core.indexing."_LocIndexer"
@pytype Ix ()->pandas_raw.core.indexing."_IXIndexer"
@pytype Series ()->pandas_raw.core.series."Series"
@pytype MultiIndex ()->pandas_raw.core.index."MultiIndex"
@pytype Index ()->pandas_raw.core.index."Index"
@pytype GroupBy ()->pandas_raw.core.groupby."DataFrameGroupBy"
@pytype SeriesGroupBy ()->pandas_raw.core.groupby."SeriesGroupBy"
@pytype Rolling () -> pandas_raw.core.window."Rolling"

@pyattr GroupBy app apply
@pyattr Rolling app apply

pyattr_set([GroupBy, SeriesGroupBy], :mean, :std, :agg, :aggregate, :median,
:var, :ohlc, :transform, :groups, :indices, :get_group, :hist,  :plot, :count)

pyattr_set([Rolling], :agg, :aggregate, :apply, :corr, :count, :cov, :kurt, :max, :mean, :median, :min, :ndim, :quantile, :skew, :std, :sum, :validate, :var)

@pyattr GroupBy siz size

pyattr_set([DataFrame, Series], :T, :abs, :align, :any, :argsort, :asfreq, :asof,
:boxplot, :clip, :clip_lower, :clip_upper, :corr, :corrwith, :count, :cov,
:cummax, :cummin, :cumprod, :cumsum, :delevel, :describe, :diff, :drop,
:drop_duplicates, :dropna, :duplicated, :fillna, :filter, :first, :first_valid_index,
:head, :hist, :idxmax, :idxmin, :iloc, :isin, :join, :last, :last_valid_index,
:loc, :mean, :median, :min, :mode, :order, :pct_change, :pivot, :plot, :quantile,
:rank, :reindex, :reindex_axis, :reindex_like, :rename, :reorder_levels,
:replace, :resample, :reset_index, :sample, :select, :set_index, :shift, :skew,
:sort, :sort_index, :sortlevel, :stack, :std, :sum, :swaplevel, :tail, :take,
:to_clipboard, :to_csv, :to_dense, :to_dict, :to_excel, :to_gbq, :to_hdf, :to_html,
:to_json, :to_latex, :to_msgpack, :to_panel, :to_pickle, :to_records, :to_sparse,
:to_sql, :to_string, :truncate, :tz_conert, :tz_localize, :unstack, :var, :weekday,
:xs, :merge)
pyattr_set([DataFrame], :groupby)
pyattr_set([Series, DataFrame], :rolling)

Base.size(x::Union{Loc, Iloc, Ix}) = x.pyo.obj.shape
Base.size(df::PandasWrapped, i::Integer) = size(df)[i]
Base.size(df::PandasWrapped) = df.pyo.shape

Base.isempty(df::PandasWrapped) = df.pyo.empty
Base.empty!(df::PandasWrapped) = df.pyo.drop(df.pyo.index, inplace=true)

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

for attr in [:index, :columns]
    @eval function $attr(x::PandasWrapped)
        pandas_wrap(x.pyo.$(string(attr)))
    end
end

@pyasvec Series
@pyasvec Loc
@pyasvec Ix
@pyasvec Iloc
@pyasvec DataFrame
@pyasvec Index
@pyasvec GroupBy
@pyasvec Rolling

Base.ndims(df::Union{DataFrame, Series}) = length(size(df))

for m in [:read_pickle, :read_csv, :read_html, :read_json, :read_excel, :read_table,
    :save, :stats,  :melt, :ewma, :concat, :pivot_table, :crosstab, :cut,
    :qcut, :get_dummies, :resample, :date_range, :to_datetime, :to_timedelta,
    :bdate_range, :period_range, :ewmstd, :ewmvar, :ewmcorr, :ewmcov, :rolling_count,
    :expanding_count, :rolling_sum, :expanding_sum, :rolling_mean, :expanding_mean,
    :rolling_median, :expanding_median, :rolling_var, :expanding_var, :rolling_std,
    :expanding_std, :rolling_min, :expanding_min, :rolling_max, :expanding_max,
    :rolling_corr, :expanding_corr, :rolling_corr_pairwise, :expanding_corr_pairwise,
    :rolling_cov, :expanding_cov, :rolling_skew, :expanding_skew, :rolling_kurt,
    :expanding_kurt, :rolling_apply, :expanding_apply, :rolling_quantile,
    :expanding_quantile, :rolling_window, :to_numeric, :read_sql, :read_sql_table,
    :read_sql_query]
    @eval begin
        function $m(args...; kwargs...)
            method = pandas_raw.$(string(m))
            result = pycall(method, PyObject, args...; kwargs...)
            pandas_wrap(result)
        end
    end
end

function show(io::IO, df::PandasWrapped)
    s = df.pyo.__str__()
    println(io, s)
end

function query(df::DataFrame, s::AbstractString)
    pandas_wrap(py"$(df.pyo).query($s)"o)
end

function query(df::DataFrame, e::Expr) # This whole method is a terrible hack
    s = string(e)
    for (target, repl) in [("&&", "&"), ("||", "|"), ("∈", "=="), (r"!(?!=)", "~")]
        s = replace(s, target=>repl)
    end
    query(df, s)
end

macro query(df, e)
    quote
        query($(esc(df)), $(QuoteNode(e)))
    end
end

for m in [:from_arrays, :from_tuples]
    @eval function $m(args...; kwargs...)
        f = pandas_raw."MultiIndex"[string($(quot(m)))]
        res = pycall(f, PyObject, args...; kwargs...)
        pandas_wrap(res)
    end
end

for (jl_op, py_op, py_opᵒ) in [(:+, :__add__, :__add__), (:*, :__mul__, :__mul__),
                               (:/, :__div__, :__rdiv__), (:-, :__sub__, :__rsub__),
                               (:(==), :__eq__, :__eq__), (:!=, :__ne__, :__ne__),
                               (:>, :__gt__, :__lt__), (:<, :__lt__, :__gt__),
                               (:>=, :__ge__, :__le__), (:<=, :__le__, :__ge__),
                               (:&, :__and__, :__and__), (:|, :__or__, :__or__)]
    @eval begin
        function $(jl_op)(x::PandasWrapped, y)
            res = x.pyo.$(string(py_op))(y)
            pandas_wrap(res)
        end

        function $(jl_op)(x::PandasWrapped, y::PandasWrapped)
            invoke($(jl_op), Tuple{PandasWrapped, Any}, x, y)
        end

        function $(jl_op)(y, x::PandasWrapped)
            res = x.pyo.$(string(py_opᵒ))(y)
            pandas_wrap(res)
        end
    end
end

for op in [(:-, :__neg__)]
    @eval begin
        $(op[1])(x::PandasWrapped) = pandas_wrap(x.pyo.$(quot(op[2]))())
    end
end

function setcolumns!(df::PandasWrapped, new_columns)
    df.pyo.__setattr__("columns", new_columns)
end

function deletecolumn!(df::DataFrame, column)
    df.pyo.__delitem__(column)
end

name(s::Series) = s.pyo.name
name!(s::Series, name) = s.pyo.name = name

include("operators_v6.jl")

function DataFrame(pairs::Pair...)
    DataFrame(Dict(pairs...))
end

function index!(df::PandasWrapped, new_index)
    df.pyo.index = new_index
    df
end

function Base.eltype(s::Series)
    dtype_map = Dict(
        np.dtype("int64") => Int64,
        np.dtype("float64") => Float64,
        np.dtype("object") => String,
    )
    get(dtype_map, s.pyo.dtype, Any)
end

function Base.eltype(df::DataFrame)
    types = []
    for column in columns(df)
        push!(types, eltype(df[column]))
    end
    Tuple{types...}
end

function Base.map(f::Function, s::Series)
    if eltype(s) ∈ (Int64, Float64)
        Series([f(_) for _ in values(s)])
    else
        Series([f(_) for _ in s])
    end
end

function Base.map(x, s::Series; na_action=nothing)
    pandas_wrap(s.pyo.map(x, na_action))
end

function Base.get(df::PandasWrapped, key, default)
    pandas_wrap(df.pyo.get(key, default=default))
end

function Base.getindex(s::Series, c::CartesianIndex{1})
    s[c[1]]
end

function Base.copy(df::PandasWrapped)
    pandas_wrap(df.pyo.copy())
end

function !(df::PandasWrapped)
    pandas_wrap(df.pyo.__neg__())
end

include("tabletraits.jl")

function DataFrame(obj)
    y = _construct_pandas_from_iterabletable(obj)
    if y===nothing
        return invoke(DataFrame, Tuple{Vararg{Any}}, obj)
    else
        return y
    end
end

function has_named_attr(x::Index, s)
    return x.pyo.__contains__(Symbol(s))
end

named_index(x::DataFrame) = columns(x)
named_index(x::Series) = index(x)

function Base.getproperty(x::Union{DataFrame, Series}, s::Symbol)
    if s == :pyo
        return getfield(x, s)
    end
    if has_named_attr(named_index(x), s)
        return x[s]
    else
        return getfield(x, s)
    end
end

end
