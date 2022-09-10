def static_function_vars(**kwargs):
    def dec(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return dec
