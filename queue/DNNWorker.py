#!/usr/bin/python3

import os
import torch


import sys
import json
import os.path
import time
import math
import random


import platform
import subprocess

import beanstalkc as BSC

import main


def count_gpus():
    result = subprocess.check_output(['nvidia-smi','-L'])
    num_gpu = len( [1 for x in result.decode('ascii').split('\n') if 'GPU' in x] )
    return num_gpu

class DNNWorker:

    def __init__(self, gpu_num, config_file):

        hostname = platform.node()

        ngpu = count_gpus()
        if not (gpu_num >= 0 and gpu_num < ngpu):
            print(f'ERROR!!! Only found {ngpu} GPUs [0..{ngpu-1}], cannot assign GPU {gpu_num}')
            sys.exit(-1)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_num}'
            self.gpu_info = (hostname,gpu_num,ngpu)
            self.my_id    = f'{hostname}-gpu{gpu_num}'

        CONFIG = json.load(open(config_file,'r'))

        self.config = CONFIG['universal']
        local = CONFIG['machine_specific'][hostname]
        for k in local.keys():
            self.config[k] = local[k]


        self.bs_conn = BSC.Connection( host=self.config['beanstalk_host'],
                                       port=self.config['beanstalk_port'] ) 

        self.current_jid = None


    def start(self):
        self.__workerLoop()


    def __jobProgressCallback(self, updateDictionary):

        # SEND UPDATE TO MANAGER
        self.bs_conn.use('jobs_progress')
        update = {'job_id':self.current_jid,'progress':updateDictionary}
        self.bs_conn.put(json.dumps(update))

    def __workerLoop(self):


        print(f'Worker ID={self.my_id} Running...')
        #flickrLink = FlickrLink.FlickrLink(CONFIG)

        self.bs_conn.watch('jobs_todo')
        self.bs_conn.use('jobs_completed')

        processingQueries = True

        while processingQueries:

            print(f'{self.my_id}: trying to get another message')
            msg = self.bs_conn.reserve(1)

            if msg:
                jbody = json.loads(msg.body)
                print(f'\n\n{self.my_id}: Found Message: {jbody}...')
                #print(f'{self.my_id}: trying to delete message... ', end="")
                msg.delete()
                #print('done')




                # PERFORM TH REQUESTED JOB




                job_id = jbody['job_id']
                self.current_jid = job_id

                job_file = jbody['job_desc_file']
                print(f'    Loading job info from file:  {job_file}')

                # TODO SHOULD CHECK HERE IF VALID JOB FILE
                jfile = injob['job_desc_file']
                jpath = injob['job_path']
                joined = os.path.join(jpath,jfile)
                rundir  = injob['rundir']
                start_and_config = f'start {joined}' if injob['start'] else 'continue'
                # TODO NEED TO HANDLE RESUME WITH SPECIFIC RUN ID

                command_string = f'main.py -D {rundir} {start_and_config}'


                print(f'    Running job {job_id}...')
                # TODO Do ACTUAL Processing here
                #time.sleep(5)

                main.main( commands=command_string,
                           callback=lambda response: self.__jobProgressCallback(response) )


                print(f'job {job_id} completed...\n\n')






                # RESPONSE
                response = {'job_id':job_id,'completed_by':self.my_id}
                print(f'{self.my_id}: trying to send completion message... ', end="")
                self.bs_conn.put(json.dumps(response))
                print('done')
                #self.sendLock.release()

                self.current_jid = None

            else:
                print(f'{self.my_id}: Sleeping... ')
                time.sleep(1)



        
    #def stopProcessing(self):
        #self.processingQueries = False





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--config', type=str, default='queue_config.json')
    args = parser.parse_args()

    worker = DNNWorker(gpu_num=args.gpu, config_file=args.config)
    worker.start()

