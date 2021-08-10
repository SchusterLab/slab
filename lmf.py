__author__ = 'Andrew Oriani'
import numpy as np
import scipy
import inspect
import lmfit
from lmfit import *
import matplotlib
import warnings
import re
from matplotlib import pyplot as plt


def get_model_params(fit_func):
    if inspect.isclass(fit_func):
        module_name = getattr(fit_func, '__module__', None)
        if module_name == lmfit.models.__name__:
            print('Model is: %s' % str(getattr(fit_func, '__name__')))
            model = fit_func()
            param_dict = {}
            for indep_vals in model.independent_vars:
                print('Independent Variable: %s' % indep_vals)
            for I, keys in enumerate(model.param_names):
                print('Parameter # %d: %s' % (I + 1, keys))
                param_dict[keys] = None
            print('p0 Param dictionary')
            print('p0=%s' % str(param_dict))
        else:
            pass
    elif type(fit_func) == lmfit.model.CompositeModel or type(fit_func) == lmfit.model.Model:
        module_name = getattr(fit_func, '__module__', None)
        if module_name == lmfit.model.__name__:
            print('Model is: %s' % str(mod).split(': ')[1].rsplit('>')[0])
            param_dict = {}
            for indep_vals in fit_func.independent_vars:
                print('Independent Variable: %s' % indep_vals)
            for I, keys in enumerate(fit_func.param_names):
                print('Parameter # %d: %s' % (I + 1, keys))
                param_dict[keys] = None
            print('p0 param dictionary')
            print('p0=%s' % str(param_dict))
        else:
            pass
    elif inspect.isfunction(fit_func):
        print('Model is: %s' % fit_func.__name__)
        model = Model(fit_func)
        param_dict = {}
        for indep_vals in model.independent_vars:
            print('Independent Variable: %s' % indep_vals)
        for I, keys in enumerate(model.param_names):
            print('Parameter # %d: %s' % (I + 1, keys))
            param_dict[keys] = None
        print('p0 Param dictionary')
        print('p0=%s' % str(param_dict))
    else:
        raise Exception('ERROR: fit function is %s, needs to be lm_fit model class or function.' %
                        str(type(a)).split(' ')[1].rsplit('>')[0])


def get_lm_models():
    models = inspect.getmembers(lmfit.models, inspect.isclass)
    print('Available lmfit model functions:')
    print('----------------------------------')
    for model in models:
        print(model[0])
    print('----------------------------------')
    print(
        'To create an empty parameter dict or learn about the variables of the different models use lmft.get_model_params(fcn)')
    print(
        'For more information on the above functional forms go to: https://lmfit.github.io/lmfit-py/builtin_models.html')


def asteval_convert(fcn, handle=None):
    if inspect.isfunction(fcn):
        aeval = asteval.Interpreter()
        if handle != None:
            fcn_name = handle
        else:
            fcn_name = fcn.__name__
        fcn_vars = inspect.getfullargspec(fcn)[0]
        aeval.symtable[fcn_name] = fcn
        return fcn_name, fcn_vars
    else:
        raise Exception('ERROR: input function is type %s' % str(type(fcn)).split(' ')[1].rsplit('>')[0])


def lm_curve_fit(fit_func, x_data, y_data, p0, p_over=None, param_domain=None, p_exprs=None, fit_domain=None,
                 verbose=False, plot_fit=False, full_output=False):
    '''
    :param fit_func: Can be a user defined function, composite model, or a built in lm_fit fit function. For list of built-in models
    use get_lm_models().
    :param x_data: x-axis data (array like (n,))
    :param y_data: y_axis data (array like (n,))
    :param p0: Initial fit parameter values, list or dict object of form p0=[a1,a2,a3..an] or p0={a1:value, a2:value...an:value}
    if p0 is of list type than p_over, param_domain, and p_exprs must also be entered as list the length of number of
    fit model input parameters. If dict is used p0 can be any length as long as parameter keys match those of input function.
    To determine parameter keys and return an empty dict object for use in fit use get_model_params(fcn).
    :param p_over: Parameter override, allows the ability to fix parameter at a set value, list or dict of form of p0. If list
    must be length of p0, if dict only needs the desired parameter for function input parameter
    :param param_domain: Bounded domain in which fit parameter is allowed, list or dict of form of p0.
    :param fit_domain: domain over which data is actually fitted, given as a tuple (min,max)
    :param p_exprs: Parameter constraint equations, given as string, can have passable user defined functions and variables.
    For dict object of form: p_exprs={'a1':'fcn1', 'a2':'fcn2'...}, with user defined variables input into dict as 'var1':[val, min, max]
    and user defined functions given as 'fcn_handle':user_fcn. For list object, each constrain must be given as a dict
    {'fcn':'fcn1','user_var':[val,min,max],'fcn_handle':user_fcn} where 'user_var' and 'fcn_handle' are optional.
    :param verbose: Formatted print statement that gives complete fit parameter information and fit statistics from lm_fit
    :param plot_fit: Plot of data with fit along with residuals plot from lm_fit
    :param full_output: Returns fit_params, fit_params_error, and returns lm_fit full fit result object

    :return: fit_params, fit_error as list or dict depending on input type if full_output=False, if fill_output=True then returns
             full fit output from lm_fit along with fit_params, fit_error
    '''

    # create fit model, determine function variables

    if inspect.isclass(fit_func):
        module_name = getattr(fit_func, '__module__', None)
        print(module_name, lmfit.models.__name__)
        if module_name == lmfit.models.__name__:
            fit_model = fit_func()
            fcn_vars = fit_model.param_names
    elif type(fit_func) == lmfit.model.CompositeModel or type(fit_func) == lmfit.model.Model:
        module_name = getattr(fit_func, '__module__', None)
        if module_name == lmfit.model.__name__:
            fit_model = fit_func
            fcn_vars = fit_model.param_names
    elif inspect.isfunction(fit_func):
        fit_model = Model(fit_func)
        fcn_vars = fit_model.param_names
    else:
        raise Exception('ERROR: fit function is %s, needs to be lm_fit model class or function.' %
                        str(type(a)).split(' ')[1].rsplit('>')[0])

    # set data fit domain
    if fit_domain != None:
        if type(fit_domain) == list:
            if len(fit_domain) == 2:
                ind = np.searchsorted(x_data, fit_domain)
                x_data = x_data[ind[0]:ind[1]]
                y_data = y_data[ind[0]:ind[1]]
            else:
                raise Exception('ERROR: fit_domain must be list of len=2, len=%d' % len(fit_domain))
        else:
            raise Exception(
                'ERROR: fit_domain myst be list type, %s type provided' % str(type(p_over)).split(' ')[1].rsplit('>')[
                    0])
    else:
        pass
    # for dict inputs
    if type(p0) is dict:

        p_guess = {}
        for keys in fcn_vars:
            if keys in p0:
                if p0[keys] == None:
                    p_guess[keys] = 0
                else:
                    p_guess[keys] = p0[keys]
            else:
                # if initial guess not provided set to zero
                p_guess[keys] = 0

        unused_keys = []
        for keys in iter(p0.keys()):
            if keys in p_guess:
                pass
            else:
                unused_keys.append(keys)

        # print a warning for any unused keys
        if len(unused_keys) != 0:
            warnings.warn('WARNING: Unused Keys: %s. Valid Parameters: %s' % (
            ', '.join(map(str, unused_keys)), ', '.join(map(str, fcn_vars))))
        else:
            pass

        # set parameters to vary
        vary_param = {}
        for key in iter(p_guess.keys()):
            vary_param[key] = True

        if p_over != None:
            if type(p_over) is dict:
                vary_len = 0
                for I, key in enumerate(iter(p_guess.keys())):
                    if key in p_over:
                        if p_over[key] == None:
                            pass
                        else:
                            p_guess[key] = p_over[key]
                            vary_param[key] = False
                            vary_len += 1
                        if vary_len == len(p_guess):
                            raise Exception('ERROR: Not enough parameters to fit')
                        else:
                            pass
                    else:
                        pass
            else:
                raise Exception(
                    'ERROR: p_over is %s, must be dict object' % str(type(p_over)).split(' ')[1].rsplit('>')[0])
        elif p_over == None:
            pass

        # set parameter domains, default, [-inf, inf]
        param_domain_vals = {}
        for key in iter(p_guess.keys()):
            param_domain_vals[key] = [-np.inf, np.inf]

        if param_domain != None:
            if type(param_domain) == dict:
                for key in iter(param_domain.keys()):
                    if param_domain[key] == None:
                        pass
                    else:
                        param_domain_vals[key] = param_domain[key]
            else:
                raise Exception('ERROR: param_domain is %s, must be dict object' %
                                str(type(param_domain)).split(' ')[1].rsplit('>')[0])
        elif param_domain == None:
            pass

        params = Parameters()
        # create parameters
        for I, key in enumerate(iter(p_guess.keys())):
            params.add(key, value=p_guess[key], vary=vary_param[key], min=param_domain_vals[key][0],
                       max=param_domain_vals[key][1])

        # set constraining expressions
        const_eqn = {}
        for key in iter(p_guess.keys()):
            const_eqn[key] = None

        if p_exprs != None:
            if type(p_exprs) == dict:
                unused_dummy_keys = []
                for key in iter(p_exprs.keys()):
                    if key in p_guess:
                        const_eqn[key] = p_exprs[key]
                    else:
                        # checking for dummy variables
                        if p_exprs[key] != None:
                            count = 0
                            for key_expr in iter(p_guess.keys()):
                                if key_expr in p_exprs:
                                    count += sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(key), p_exprs[key_expr]))
                                else:
                                    pass
                            # checking if number of parameters is correct
                            if count == 0:
                                unused_dummy_keys.append(key)
                            else:
                                if inspect.isfunction(p_exprs[key]):
                                    params._asteval.symtable[key] = p_exprs[key]
                                elif type(p_exprs[key]) == list:
                                    if len(p_exprs[key]) == 3:
                                        params.add(key, value=p_exprs[key][0], min=p_exprs[key][1], max=p_exprs[key][2])
                                    else:
                                        raise Exception(
                                            '%s needs list len=3, [val, min, max], provided list of len=%d' % (
                                            p_exprs[key], len(p_exprs[key])))
                                elif type(float(p_exprs[key])) == float:
                                    params.add(key, value=p_exprs[key], vary=False)
                                else:
                                    raise Exception('Variable input must be function or list, input is %s for %s' % (
                                    str(type(p_exprs[key])).split(' ')[1].rsplit('>')[0], key))
                        else:
                            pass
                if len(unused_dummy_keys) != 0:
                    warnings.warn('WARNING: Unused dummy variables: %s' % (', '.join(map(str, unused_dummy_keys))))
                else:
                    pass
            else:
                raise Exception(
                    'ERROR: p_exprs is %s, must be dict object' % str(type(p_exprs)).split(' ')[1].rsplit('>')[0])
        elif p_exprs == None:
            pass

        # set constraining equations for appropriate variables
        for key in iter(const_eqn.keys()):
            params[key].set(expr=const_eqn[key])
        print(y_data, params, x_data)
        result = fit_model.fit(y_data, params, x=x_data)

        fit_params = {}
        fit_err = {}
        for key in iter(result.params.keys()):
            fit_params[key] = result.params[key].value
            fit_err[key] = result.params[key].stderr

        if verbose == True:
            print(result.fit_report())
        if plot_fit == True:
            fig = plt.figure(figsize=(10, 8))
            result.plot(fig=fig)
        if full_output == True:
            return fit_params, fit_err, result
        elif full_output == False:
            return fit_params, fit_err

    # same as above except for list arguments
    elif type(p0) is list:

        if len(p0) != len(fcn_vars):
            raise Exception('ERROR: Initial guess incorrect length, %d entered, %d required' % (len(p0), len(fcn_vars)))

        vary_param = [True] * len(p0)

        if p_over != None:
            if type(p_over) == list:
                if len(p_over) != len(fcn_vars):
                    raise Exception('Parameter override must be len=%d, instead len=%d' % (len(fcn_vars), len(p_over)))
                else:
                    vary_len = 0
                    for I, val in enumerate(p_over):
                        if p_over[I] == None:
                            pass
                        else:
                            p0[I] = p_over[I]
                            vary_param[I] = False
                            vary_len = 0
                        if vary_len == len(p0):
                            raise Exception('ERROR: Not enough parameters to fit')
                            return
                    else:
                        pass
            else:
                raise Exception(
                    'ERROR: p_over is %s, must be list object' % str(type(p_over)).split(' ')[1].rsplit('>')[0])
        elif p_over == None:
            pass

        for I, vals in enumerate(p0):
            if vals == None:
                p0[I] = 0.0

        param_domain_vals = [[-np.inf, np.inf]] * len(p0)

        if param_domain != None:
            if len(param_domain) != len(fcn_vars):
                raise Exception('Parameter domain must be len=%d, instead len=%d' % (len(fcn_vars), len(param_domain)))
            else:
                pass
            if type(param_domain) == list:
                for I, param in enumerate(param_domain):
                    if param_domain[I] == None:
                        pass
                    else:
                        param_domain_vals[I] = param_domain[I]
            else:
                raise Exception('ERROR: param_domain is %s, must be list object' %
                                str(type(param_domain)).split(' ')[1].rsplit('>')[0])
        elif param_domain == None:
            pass

        params = Parameters()
        for I, (p_guess, vary, domain) in enumerate(zip(p0, vary_param, param_domain_vals)):
            params.add(fcn_vars[I], value=p_guess, vary=vary, min=domain[0], max=domain[1])

        # set constraint equations
        const_eqn = [None] * len(p0)
        if p_exprs != None:
            if type(p_exprs) == list:
                if len(p_exprs) != len(fcn_vars):
                    raise Exception('Parameter domain must be len=%d, instead len=%d' % (len(fcn_vars), len(p_exprs)))
                else:
                    pass
                for I, val in enumerate(p_exprs):
                    if val != None:
                        if type(val) != dict:
                            raise Exception(
                                'Parameter expression must be type=dict of form {\'fcn\':expr,\'opt_var\':[value, min, max]}, instead %d' %
                                str(type(val)).split(' ')[1].rsplit('>')[0])
                        else:
                            pass
                        if 'fcn' in val:
                            for key in iter(val.keys()):
                                if key == 'fcn':
                                    const_eqn[I] = val[key]
                                else:
                                    if val[key] != None:
                                        count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(key), val['fcn']))
                                        if count == 0:
                                            warnings.warn('WARNING: Unused dummy variable: %s' % key)
                                        else:
                                            if inspect.isfunction(val[key]):
                                                params._asteval.symtable[key] = val[key]
                                            elif type(val[key]) == list:
                                                if len(val[key]) == 3:
                                                    params.add(key, value=val[key][0], min=val[key][1], max=val[key][2])
                                                else:
                                                    raise Exception(
                                                        '%s needs list len=3, [val, min, max], or len=1 for vary=False, provided list of len=$d' % (
                                                        val[key], len(val[key])))
                                            elif type(float(val[key])) == float:
                                                params.add(key, value=val[key], vary=False)
                                            else:
                                                raise Exception(
                                                    'Variable input must be function, float, or list, type of input is %s' %
                                                    str(type(p_exprs[key])).split(' ')[1].rsplit('>')[0])
                                    else:
                                        pass
                        else:
                            raise Exception('Missing \'fcn\' key in input dict object')
                    else:
                        pass
            else:
                raise Exception(
                    'Parameter expression must be type=list, made of type=dict of form {\'fcn\':expr,\'opt_var\':[value, min, max]}, instead %s' %
                    str(type(val[key])).split(' ')[1].rsplit('>')[0])

        # set constraining expressions
        for exprs, var in zip(const_eqn, fcn_vars):
            params[var].set(expr=exprs)
        print(y_data, params, x_data)
        result = fit_model.fit(y_data, params, x=x_data)

        fit_params = []
        fit_err = []
        for key in iter(result.params.keys()):
            fit_params.append(result.params[key].value)
            fit_err.append(result.params[key].stderr)
        if verbose == True:
            print(result.fit_report())
        if plot_fit == True:
            fig = plt.figure(figsize=(10, 8))
            result.plot(fig=fig)
        if full_output == True:
            return fit_params, fit_err, result
        elif full_output == False:
            return fit_params, fit_err
    else:
        raise Exception(
            'ERROR: Unsupported data type input %s, must be dict or list' % str(type(p0)).split(' ')[1].rsplit('>')[0])


