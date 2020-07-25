# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:58:20 2018

@author: Kai Qin
"""
import time

testTag = 'results: '


def debug_test(mes):
    print(testTag + str(mes))


def info_test(mes):
    print(testTag + str(mes))


def debug_tag(tag, mes):
    print(tag + ': ' + str(mes))


def info_tag(tag, mes):
    print(tag + ': ' + str(mes))


def get_time():
    return time.time()


def time_diff_tag(tag, t1, t2):
    dif = round(t2 - t1, 4)
    print(tag, ':', dif, 's')
    return dif


def time_diff(t1, t2):
    dif = round(t2 - t1, 4)
    return dif


def time_diff_now(tag, startTime):
    if startTime is None:
        return
    endTime = time.time()
    dif = round(endTime - startTime, 4)
    print(tag + ':', dif, 's')
    return dif
