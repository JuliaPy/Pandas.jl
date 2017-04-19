import Base: ==, >, <, >=, <=, !=

for (op, pyop) in [(:(==), :__eq__), (:>, :__gt__), (:<, :__lt__), (:>=, :__ge__), (:<=, :__le__), (:!=, :__ne__)]
    @eval function Base.broadcast(::typeof($op), s::PandasWrapped, x)
        pandas_wrap(s.pyo[$(QuoteNode(pyop))](x))
    end
end
