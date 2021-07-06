# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:30:31 2018

@author: Josie Meyer
"""

from distutils.core import setup, Extension

module_name = "datahandlerhelper"

module = Extension(module_name,
                   sources = ['DataHandlerHelper.c'],
                   include_dirs = [r"C:\_Lib\SD1\Libraries\include\c",
                                   r"C:\_Lib\C\pthreads-win32_2",
                                   r"C:\Users\slab\Anaconda3\pkgs\numpy-base-1.14.3-py36h555522e_1\Lib\site-packages\numpy\core\include\numpy"],
                    libraries = ["keysightSD1", "pthreadVC2"], 
                    library_dirs = [r"C:\_Lib\SD1\Libraries\libx64\c",
                                r"C:\_Lib\C\pthreads-win32_2",
                                r"C::\_Lib\SD1\shared"])                                 

setup(name = module_name,
      author = "Josephine Meyer",
      author_email = "jcmeyer@stanford.edu",
      version = "1.0",
      ext_modules = [module])