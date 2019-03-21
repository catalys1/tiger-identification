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
    parser.add_argument('--jobfile', type=str, required=True,
        help='JSON file containing information needed to run the job '
             '(like a config file')
    parser.add_argument('--jobpath', type=str, default='.',
        help='Path to directory where the job code lives.')
    parser.add_argument('-r', '--rundir', type=str, required=True,
        help='Path to directory where the job output will be saved.')
    parser.add_argument('--mainfile', type=str, default='main.py',
        help='Python file containing the main function to be run')
    parser.add_argument('--config', type=str, default='queue_config.json')
    parser.add_argument('--rid', type=int, default=0)
    args = parser.parse_args()

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
    msg = {'job_desc_file':job_desc_file,
           'job_path':os.path.realpath(jobpath),
           'main_file':args.mainfile,
           'rundir':os.path.realpath(args.rundir),
           'runid':args.rid,
           'other_params':[]}
    #msg = {'command':f'main.py -D {args.rundir} start 

    bs_conn.put(json.dumps(msg))
    bs_conn.close()




