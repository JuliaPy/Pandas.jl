import Base: .==, .>, .<, .>=, .<=, .!=

for (op, pyop) in [(:.==, :__eq__), (:.>, :__gt__), (:.<, :__lt__), (:.>=, :__ge__), (:.<=, :__le__), (:.!=, :__ne__)]
    @eval function $op(s::PandasWrapped, x)
        method = s.pyo[$(QuoteNode(pyop))]
        pandas_wrap(pycall(method, PyObject, x))
    end
end
