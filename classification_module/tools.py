# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:38:06 2017

@author: Administrator
"""

import threading

class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

#  def __next__(self):
#      with self.lock:
#          return self.it.__next__()
  def next(self):
      with self.lock:
          return self.it.next()

#def threadsafe_generator(f):
#  """
#    A decorator that takes a generator function and makes it thread-safe.
#    """
#  def g(*a):#, **kw):
#      return threadsafe_iter(f(*a))#, **kw))
#  return g

def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a):#*a, **kw):
      return threadsafe_iter(f(*a))#*a, **kw))
  return g