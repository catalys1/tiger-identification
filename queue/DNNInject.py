#!/usr/bin/python3

import os
import torch


import sys
import json
import os.path
import time
import math
import random

#from Messages import *

import platform
import subprocess

import beanstalkc as BSC



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobfile', type=str, required=True )
    parser.add_argument('--jobpath', type=str, default='.' )
    parser.add_argument('--rundir', type=str, required=True )
    parser.add_argument('--config', type=str, default='queue_config.json')
    args = parser.parse_args()

    #job_desc_file='random_filename'
    job_desc_file=args.jobfile


    hostname = platform.node()


    CONFIG = json.load(open(args.config,'r'))

    config = CONFIG['universal']
    if hostname in CONFIG['machine_specific']:
        local = CONFIG['machine_specific'][hostname]
        for k in local.keys():
            config[k] = local[k]


    bs_conn = BSC.Connection( host=config['beanstalk_host'],
                              port=config['beanstalk_port'] ) 
    bs_conn.use('jobs_incoming')

    jobpath = os.path.abspath(args.jobpath)
    msg = {'job_desc_file':job_desc_file,'job_path':jobpath,'rundir':args.rundir,'start':True,'other_params':[]}
    #msg = {'command':f'main.py -D {args.rundir} start 

    bs_conn.put(json.dumps(msg))

    bs_conn.close()




