#!/usr/bin/python3
import time
import math
#from PyQt4 import QtGui, QtCore

#from Queue import Queue
#from sets import Set

import beanstalkc as BSC
import json
import platform
import sys
import os
import subprocess

#from Messages import *


class DNNManager():
    #onupdate = QtCore.pyqtSignal(dict)
    #batchCompleted = QtCore.pyqtSignal(dict)
    def __is_port_in_use__(self, port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def __init__( self, config_file):
        #super(DNNManager, self).__init__(parent=parent)
        host = platform.node()

        CONFIG = json.load(open(config_file,'r'))
        self.config = CONFIG['universal']
        if host in CONFIG['machine_specific']:
            local = CONFIG['machine_specific'][host]
            for k in local.keys():
                self.config[k] = local[k]

        targ = self.config['beanstalk_host']
        port = self.config['beanstalk_port']
        if not targ == host:
            print(f'\rERROR!!!  {config_file} indicates that manager should be run on {targ.upper()} not {host.upper()}')
            sys.exit(-1)
        else:
            # FORK the beanstalk daemon, if not already running
            if self.__is_port_in_use__(port):
                print(f'\rERROR!!!  It appears that port {port} is already in use!!!')
                sys.exit(-1)
            else:
                ps = subprocess.Popen(['beanstalkd','-V','-p',f'{port}'])
       

                self.bs_conn = BSC.Connection(host=self.config['beanstalk_host'],port=self.config['beanstalk_port'])
                #ps.wait()


        #self.current_batch = None

        #self.batches = Queue()

        #self.waitingForIds = Set()
        #self.completed = Set()
        #self.resultsAccum = {}
        #self.batchQueueLock = QtCore.QMutex()




    def is_valid_path( self, jpath ):
        valid_prefixes = self.config['shared_locations']
        shared = False
        for p in valid_prefixes:
            if jpath.startswith(p):
                shared = True
        return (os.path.exists(jpath) and shared)

    def run(self):
        self.job_id = 0
        self.running = True
        print( 'DNNManager running...' )

        while self.running:
            print('\rChecking for incoming jobs...'.ljust(40, ' '), end='')
            #self.bs_conn.watch('jobs_completed')
            self.bs_conn.watch('jobs_incoming')
            self.bs_conn.use('jobs_todo')

            msg = self.bs_conn.reserve(1)
            while msg:
                # process incoming job
                injob = json.loads(msg.body)
                self.job_id += 1
                injob['job_id'] = f'ID-{self.job_id}'
                jid = injob['job_id']
                # Check if valid/shared file
                jfile = injob['job_desc_file']
                jpath = injob['job_path']
                joined = os.path.join(jpath,jfile)
                if not self.is_valid_path( jpath ):
                    print(f'\r    ERROR:  Path {jpath} is not shared or doesn\'t exist!!')
                    self.job_id -= 1 # redo this number
                elif not os.path.exists( joined ):
                    print(f'\r    ERROR:  File {joined} doesn\'t exist!!')
                    self.job_id -= 1 # redo this number
                else:
                    print(f'\r    Posting Job {jid}...', end='')
                    self.bs_conn.put( json.dumps(injob) )
                msg.delete()
                print(f'\rdone!'.ljust(40, ' '))

                # see if others!
                msg = self.bs_conn.reserve(1)
            self.bs_conn.ignore('jobs_incoming')



            print('\rChecking for completed jobs...'.ljust(40, ' '), end='')
            self.bs_conn.watch('jobs_completed')

            msg = self.bs_conn.reserve(1)
            while msg:
                jid = json.loads(msg.body)['job_id']
                who = json.loads(msg.body)['completed_by']
                print(f'\r    Found completed job {jid} (completed by {who})')

                msg.delete()
                msg = self.bs_conn.reserve(1)

            self.bs_conn.ignore('jobs_completed')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='queue_config.json')
    args = parser.parse_args()
    
    mgr = DNNManager(config_file=args.config)

    mgr.run()

    # beanstalkd -V -p 52700
