""" 
    helps to automatically assign arguments and keywords to a class
    works as a decorator. 
    More argument please refer to
    http://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
    
    first discovered by Phil

    created by Ge 2012-08-02    
    """

import inspect
import functools
def autoargs(*include,**kwargs):   
    def _autoargs(func):
        attrs,varargs,varkw,defaults=inspect.getargspec(func)
        def sieve(attr):
            if kwargs and attr in kwargs['exclude']: return False
            if not include or attr in include: return True
            else: return False            
        @functools.wraps(func)
        def wrapper(self,*args,**kwargs):
            # handle default values
            for attr,val in zip(reversed(attrs),reversed(defaults)):
                if sieve(attr): setattr(self, attr, val)
            # handle positional arguments
            positional_attrs=attrs[1:]            
            for attr,val in zip(positional_attrs,args):
                if sieve(attr): setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args=args[len(positional_attrs):]
                if sieve(varargs): setattr(self, varargs, remaining_args)                
            # handle varkw
            if kwargs:
                for attr,val in kwargs.iteritems():
                    if sieve(attr): setattr(self,attr,val)            
            return func(self,*args,**kwargs)
        return wrapper
    return _autoargs
    
    
### Unit test

import unittest
import utils_method as um

class Test(unittest.TestCase):
    def test_autoargs(self):
        class A(object):
            @um.autoargs()
            def __init__(self,foo,path,debug=False):
                pass
        a=A('rhubarb','pie',debug=True)
        self.assertTrue(a.foo=='rhubarb')
        self.assertTrue(a.path=='pie')
        self.assertTrue(a.debug==True)

        class B(object):
            @um.autoargs()
            def __init__(self,foo,path,debug=False,*args):
                pass
        a=B('rhubarb','pie',True, 100, 101)
        self.assertTrue(a.foo=='rhubarb')
        self.assertTrue(a.path=='pie')
        self.assertTrue(a.debug==True)
        self.assertTrue(a.args==(100,101))        

        class C(object):
            @um.autoargs()
            def __init__(self,foo,path,debug=False,*args,**kw):
                pass
        a=C('rhubarb','pie',True, 100, 101,verbose=True)
        self.assertTrue(a.foo=='rhubarb')
        self.assertTrue(a.path=='pie')
        self.assertTrue(a.debug==True)
        self.assertTrue(a.verbose==True)        
        self.assertTrue(a.args==(100,101))        

    def test_autoargs_names(self):
        class C(object):
            @um.autoargs('bar','baz','verbose')
            def __init__(self,foo,bar,baz,verbose=False):
                pass
        a=C('rhubarb','pie',1)
        self.assertTrue(a.bar=='pie')
        self.assertTrue(a.baz==1)
        self.assertTrue(a.verbose==False)
        self.assertRaises(AttributeError,getattr,a,'foo')

    def test_autoargs_exclude(self):
        class C(object):
            @um.autoargs(exclude=('bar','baz','verbose'))
            def __init__(self,foo,bar,baz,verbose=False):
                pass
        a=C('rhubarb','pie',1)
        self.assertTrue(a.foo=='rhubarb')
        self.assertRaises(AttributeError,getattr,a,'bar')


if __name__ == '__main__':
    unittest.main(argv = unittest.sys.argv + ['--verbose'])