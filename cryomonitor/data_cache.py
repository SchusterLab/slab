# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 22:13:59 2012

@author: Ge
"""


import numpy as np
from slab.datamanagement import SlabFile
from slab.dataanalysis import get_current_filename, get_next_filename
import matplotlib.pyplot as plt

# from liveplot import LivePlotClient

from os import path
import textwrap2
import re

class dataCacheProxy():
    def __init__(self, expInst=None, newFile=False, stack_prefix='stack_', filepath=None):
        """ filepath is for reading out files ONLY. To create new data files, please use the expInst
        interface """
        if filepath == None and expInst != None:
            #print 'now load the file from instance'
            self.exp = expInst
            self.data_directory = expInst.expt_path
            #print "self.data_directory", self.data_directory
            self.prefix = expInst.prefix
            #print "self.prefix", expInst.prefix
            self.set_data_file_path(newFile)
        else:
            self.path = filepath

        self.current_stack = ''
        self.stack_prefix = stack_prefix
        # self.currentStack = lambda:None;
        # self.currentStack.note = self.note
        # self.currentStack.set = self.set
        # self.currentStack.post = self.post

    def set_data_file_path(self, newFile=False):
        print('create new file: ', newFile)
        try:
            if newFile == True:
                #print 'creating new file'
                self.filename = get_next_filename(self.data_directory, self.prefix, suffix='.h5')
                #print "the new file that's created", self.filename
            else:
                #print 'load the current file'
                self.filename = get_current_filename(self.data_directory, self.prefix, suffix='.h5')
        except AttributeError:
            self.filename = self.exp.filename
        self.path = path.join(self.data_directory, self.filename)

    def add(self, keyString, data):
        def add_data(f, group, keyList, data):
            if len(keyList) == 1:
                f.add_data(group, keyList[0], data)
            else:
                try:
                    group.create_group(keyList[0])
                except ValueError:
                    pass
                return add_data(f, group[keyList[0]], keyList[1:], data)

        keyList = keyString.split('.')
        with SlabFile(self.path, 'a') as f:
            add_data(f, f, keyList, data)

    def append(self, keyString, data):
        def append_data(f, group, keyList, data):
            if len(keyList) == 1:
                f.append_data(group, keyList[0], data)
            else:
                try:
                    group.create_group(keyList[0])
                except ValueError:
                    pass
                return append_data(f, group[keyList[0]], keyList[1:], data)

        keyList = keyString.split('.')
        with SlabFile(self.path, 'a') as f:
            append_data(f, f, keyList, data)

    def set(self, route, data):
        """
        add a datapoint to the current data stack.
        """
        try:
            if self.current_stack != '':
                if self.current_stack[-1] == '.': self.current_stack = self.current_stack[:-1]
                route = self.current_stack + "." + route
        except AttributeError:
            pass
        self.add(route, data)

    def post(self, route, data):
        """
        append a datapoint to the current data stack.
        """
        try:
            if self.current_stack != '':
                if self.current_stack[-1] == '.': self.current_stack = self.current_stack[:-1]
                route = self.current_stack + "." + route
        except AttributeError:
            pass
        self.append(route, data)

    def find(self, keyString):
        def get_data(f, keyList):
            if len(keyList) == 1:
                return f[keyList[0]][...];
            else:
                return get_data(f[keyList[0]], keyList[1:])

        keyList = keyString.split('.')
        with SlabFile(self.path, 'r') as f:
            return get_data(f, keyList)

    def get(self, route):
        """
        get a dataset from the current data stack.
        """
        try:
            if self.current_stack != '':
                if self.current_stack[-1] == '.': self.current_stack = self.current_stack[:-1]
                route = self.current_stack + '.' + route
        except AttributeError:
            pass
        return self.find(route)

    def get_next_stack_index(self):
        """ this also mutates the current_stack pointer, to point to the enxt stack
        """
        try: index = int(self.current_stack[-5:]) + 1
        except: index = 0
        self.current_stack = self.stack_prefix + str(100000 + index)[1:]
        return index

    def find_last_stack(self):
        try:
            index = int(self.current_stack[-5:])
        except AttributeError:
            index = 0
        except ValueError:
            index = 0
        while True:
            self.current_stack = self.stack_prefix + str(100000 + index)[1:]
            try:
                self.index()
                print("the last stack is ", self.current_stack)
                break
            except IOError:
                print('file does not exist, current_stack is ', self.current_stack);
                break
            except:
                index += 1
                print('.', end=' ')

    def index(self, keyString=''):
        def get_indices(f, keyList):
            if len(keyList) == 0 or (keyList == ['', ]):
                try:
                    return list(f.keys())
                except AttributeError as e:
                    return e
            else:
                return get_indices(f[keyList[0]], keyList[1:])

        try:
            if keyString[-1] == '.':
                keyString = keyString[:-1]
        except IndexError as e:
            pass
        keyList = keyString.split('.')
        with SlabFile(self.path, 'r') as f:
            return get_indices(f, keyList)

    def index_current_stack(self, keyString=''):
        if self.current_stack == '':
            return self.index(keyString)
        else:
            return self.index(self.current_stack + '.' + keyString)

    def new_stack(self):
        index = self.get_next_stack_index()
        with SlabFile(self.path) as f:
            try:
                f.create_group(self.current_stack)
                print("new stack: ", self.current_stack);
            except ValueError as error:
                print("{} already exists. Move to next index.".format(self.current_stack))
                self.new_stack()

    def note(self, string, keyString=None, printOption=False, maxLength=79):
        if keyString == None:
            keyString = 'notes'
        if printOption:
            print(string)
        for line in textwrap2.wrap(string, maxLength):
            self.post(keyString, line + ' ' * (maxLength - len(line)))

    def set_dict(self, keyString, d):
        """

        :param keyString:
        :param d:
        :return:
        """

        if not isinstance(d, dict):
            print('object is not a dictionary object')
            return
        for key, value in d.items():
            if isinstance(value, dict):
                self.set_dict(keyString + '.' + key, value)
            else:
                self.set(keyString + '.' + key, value)

    def get_dict(self, keyString):
        try:
            return self.get(keyString)
        except AttributeError:
            d = {}
            for key in self.index_current_stack(keyString):
                d[key] = self.get_dict(keyString + '.' + key)
            return d

if __name__ == "__main__":
    print("running a test...")

    # Setting up the instance
    exp = lambda: None;
    exp.expt_path = './'
    exp.prefix = 'test'
    cache = dataCacheProxy(exp)

    # test data
    test_data_x = np.arange(0, 10, 0.01)
    test_data_y = np.sin(test_data_x)

    # example usage
    cache.append('key1', (test_data_x, test_data_y) )
    #plt.plot(cache.get('key1')[0][1])
    cache.add('key2', (test_data_x, test_data_y) )
    #plt.plot(cache.get('key2')[1])
    # plt.show()

    cache.add('group1.key1', (test_data_x, test_data_y) )
    #plt.plot(cache.get('group1.key1')[1])
    cache.append('group1.key2', (test_data_x, test_data_y) )
    #plt.plot(cache.get('group1.key2')[0][1])
    # plt.show()

    cache.add('group1.subgroup.key2', (test_data_x, test_data_y) )
    #plt.plot(cache.get('group1.subgroup.key2')[1])
    cache.append('group1.subgroup.key3', (test_data_x, test_data_y) )
    #plt.plot(cache.get('group1.subgroup.key3')[0][1])
    #plt.show()

    ### now test the set_dict method
    d = {
            'key0': 'haha',
            'key1': 'haha',
            'd0': {
                'key2': 'some stuff is here',
                'key3': 'some more stuff is here'
            }
        }
    ## new file then find the last stack
    cache = dataCacheProxy(exp, newFile=True)
    cache.find_last_stack()
    cache.set_dict('dictionary', d)
    print(cache.index_current_stack('dictionary'))
    d_copy = cache.get_dict('dictionary')
    print(d, d_copy)
    assert d == d_copy, 'set_dict and get_dict round-trip failed'

    ## create new file and just add to the root
    cache = dataCacheProxy(exp, newFile=True)
    cache.set_dict('dictionary', d)
    print(cache.index_current_stack('dictionary'))
    d_copy = cache.get_dict('dictionary')
    print(d, d_copy)
    assert d == d_copy, 'set_dict and get_dict round-trip failed'

    cache.find_last_stack()
    print('the latest stack is: ', cache.current_stack)

    #now data is saved in the dataCache
    #now I want to move some data to a better file
    #each experiment is a file, contains:
    # - notes
    # - stack_<numeric>
    #   - configs
    #   - mags
    #   - fpts
    #   - vpts