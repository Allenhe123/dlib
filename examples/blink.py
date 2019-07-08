#!/usr/bin/python2

import sys
# import proto.allen_blink_pb2
from proto.allen_blink_pb2 import *
import google.protobuf

def readBlink(blk):
    for ear in blk.ears:
        print("ear: ", ear)
    for idx in blk.blink_idx:
        print("idx: ", idx)
    print("blink_ear num: ", len(blk.ears))
    print("blink_idx num: ", len(blk.blink_idx))
    print("blink_num: ", blk.blink_num)

def readYaw(yaw):
    for ear in yaw.ears:
        print("yaw ear: ", ear)
    for idx in yaw.yaw_idx:
        print("yaw idx: ", idx)
    print("yaw_ear num: ", len(yaw.ears))
    print("yaw_idx num: ", len(yaw.yaw_idx))
    print("yaw num: ", yaw.yaw_num)

if __name__ == "__main__":
    print("hello world!")
    f = open('blink.dat', "rb")
    blink = Blink()
    blink.ParseFromString(f.read())
    f.close()
    readBlink(blink)

    f = open("yaw.dat", "rb")
    yw = Yaw()
    yw.ParseFromString(f.read())
    f.close()
    readYaw(yw)
