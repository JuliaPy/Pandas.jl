function gen_window_names()
    s_set = {}
    for m in ["count", "sum", "mean", "median", "var", "std", "min", "max", "corr", "corr_pairwise", "cov", "skew", "kurt", "apply", "quantile"]
        for f in ["rolling", "expanding"]
            s = string(":", f, "_", m)
            push!(s_set, s)
        end
    end
    return join(s_set, ", ")
end