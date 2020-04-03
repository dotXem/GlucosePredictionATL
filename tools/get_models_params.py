from pydoc import locate


def locate_model(model_name):
    return locate("processing.models." + model_name + "." + model_name.upper())


    # getattr(__import__('processing.models.' + model_name, fromlist=[model_name.upper()]))


def locate_params(params_name):
    return locate ("processing.models.params." + params_name + ".parameters")


# def my_import(name):
#     components = name.split('.')
#     mod = __import__(components[0])
#     for comp in components[1:]:
#         mod = getattr(mod, comp)
#     return mod