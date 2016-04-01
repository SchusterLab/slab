# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:55:47 2011

@author: Dai
"""
import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('128.135.35.166', 23))
s.send('READ');
time.sleep(1);
print s.recv(1024);
s.send('READ_AVG');
time.sleep(.8);
print s.recv(1024);
s.send('invalid command');
time.sleep(1);
print s.recv(1024);
s.send('END');
time.sleep(1);
print s.recv(1024);