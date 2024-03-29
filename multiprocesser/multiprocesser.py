from __future__ import print_function, division

from numbers import Integral
import multiprocessing
import time
import sys

import numpy as np


class MultiProcesser:
    def __init__(self, cpus=multiprocessing.cpu_count(), info=False, stop_on_error=True, delay=0.3):
        # Verify that the number of processes is not more that available
        if cpus > multiprocessing.cpu_count():
            cpus = multiprocessing.cpu_count()
        self.cpus = cpus
        self.worker_pool = None
        self.delay = delay
        self.info = info
        self.stop_on_error = stop_on_error

    def start(self):
        if self.worker_pool is not None:
            self.terminate()
        self.worker_pool = multiprocessing.Pool(processes=self.cpus)

    def run_jobs(self, jobs, timeout):
        cpus = self.cpus
        # Start timer
        start_time = time.time()

        try:
            #      Assemble the command for each job
            job_args = []
            for job in jobs:
                function = job[0]
                arguments = job[1]
                kwarguments = job[2]

                if (arguments is None) and (kwarguments is None):
                    job_args.append([function, [], {}])
                elif arguments is None:
                    job_args.append([function, [], kwarguments])
                elif kwarguments is None:
                    job_args.append([function, arguments, {}])
                else:
                    job_args.append([function, arguments, kwarguments])

        except:
            print(" ERROR: multiProcessor - The received arguments could not be interpreted")
            print("        The data must be on the following form:")
            print("        [  ( functionReference, arguments, key word arguments as dictionary )\n\t , "
                  "( myFun,[x,y,z], {'a':2 ,'b':3} ) \n\t ,  (myFun2, None, None),   ]\n")
            raise

        try:
            queue = []
            for job_arg in job_args:
                function = job_arg[0]
                arguments = job_arg[1]
                kwarguments = job_arg[2]
                queue.append(self.worker_pool.apply_async(function, arguments, kwarguments))
                time.sleep(self.delay)
            # Get results
            results = []

            for i, item in enumerate(queue):
                try:  # noinspection PyBroadException
                    results.append(item.get(timeout=timeout))
                    if self.info:
                        print(" Completed %s of %s" % (i + 1, len(queue)))
                        sys.stdout.flush()
                except multiprocessing.TimeoutError:
                    print("\n ERROR: Timeout\n")
                    print("        To avoid dead lock when workers do not operate as intended")
                    print("        or an unrecoverable error arises a time out time is set.")
                    print(" ")
                    print("        The default timeout is 10s. ")
                    print(" ")
                    print("        This parameter can be manually adjusted as an argument when")
                    print("        calling this module, append timeout='number of seconds'")
                    print("        to adjust when timeout should occur.")

                    if self.stop_on_error:
                        print("\n\n Terminating child processes")
                        self.worker_pool.terminate()
                        print("-Done\n")
                        raise
                    else:
                        results.append(False)
                        continue
                except:
                    print("\n\n The following problem was encountered:")
                    print(sys.exc_info()[0])  # - Exit type:', sys.exc_info()[0]
                    print(sys.exc_info()[1])  # - Exit type:', sys.exc_info()[0]

                    print("Stop on error: ", self.stop_on_error)
                    if self.stop_on_error:
                        print("\n\n Terminating child processes")
                        self.worker_pool.terminate()
                        print("-Done\n")
                        raise
                    else:
                        results.append(False)
                        continue

            end_time = float(round((time.time() - start_time)*10))/10
            if self.info:
                print(" Program used %s sub processes with a total duration of: %ss" % (cpus, end_time))
            return results

        except:
            print("\n\n The following problem was encountered:")
            print(sys.exc_info()[0])  # - Exit type:', sys.exc_info()[0]
            print(sys.exc_info()[1])  # - Exit type:', sys.exc_info()[0]
            raise

    def process_function(self, function, data_list, keyword_data=None, force_multiprocessing=False,
                         timeout=10):
        if keyword_data is None:
            keyword_data = {}
        if self.cpus > 1 or force_multiprocessing:
            data_chunks_list = []
            start = 0
            for i in range(self.cpus):
                tot_size = len(data_list)
                if i < tot_size % self.cpus:
                    size = tot_size // self.cpus + 1
                else:
                    size = tot_size // self.cpus
                data_chunks_list.append(data_list[start:start + size])
                start += size
            job_list = [(_loop_func, (function, data_chunk), {"keyword_data": keyword_data})
                        for data_chunk in data_chunks_list]
            results = self.run_jobs(job_list, timeout=timeout)
            return [item for sublist in results for item in sublist]
        else:
            return [function(data, **keyword_data) for data in data_list]

    def apply(self, function, data_list, keyword_data=None, axis_split=0, force_multiprocessing=False,
              timeout=10):
        """
          Multiprocessing one or several arrays given in the *args data in the function function
          :param function:              Function to process

          :param data_list:             A list with one or more iterables of the same length that will be processed in
                                        parallel
          :param keyword_data:          A dict with additional keyword arguments to function that is common for all
                                        points

          :param cpus                   int giving the number of cpus to be used

          :param axis_split             if data_list contains multidimensional arrays, axis_split can be given as an
                                        int or a list having the same length as data_list along which axis will be
                                        processed. For one-dimensional arrays, this argument is neglected
          :param force_multiprocessing  if True, a new process will be started even though num_cpus=1


          :param timeout:               This is a timeout value to avoid dead lock of processes. If the process takes
                                        longer than the prescribed value a timeOutError will be raised and the child
                                        process terminated.

          :return:                      A numpy array with return values from function
          """
        if keyword_data is None:
            keyword_data = {}
        if self.cpus > 1 or force_multiprocessing:
            if isinstance(axis_split, Integral):
                axis_split = [axis_split]*len(data_list)
            data_chunks_list = [list() for _ in range(self.cpus)]
            for data, axis in zip(data_list, axis_split):
                if isinstance(data, np.ndarray):
                    if len(data.shape) == 1:
                        axis = 0
                    data_chunks = np.array_split(data, self.cpus, axis=axis)
                else:
                    data_chunks = []
                    start = 0
                    for i in range(self.cpus):
                        tot_size = len(data)
                        if i < tot_size % self.cpus:
                            size = tot_size // self.cpus + 1
                        else:
                            size = tot_size // self.cpus
                        data_chunks.append(data[start:start + size])
                        start += size
                for i, chunk in enumerate(data_chunks):
                    data_chunks_list[i].append(chunk)
            job_list = [(function, data_chunks_list[i], keyword_data) for i in range(self.cpus)]
            results = self.run_jobs(job_list, timeout=timeout)
            return np.vstack(results)
        else:
            return function(*data_list, **keyword_data)

    def terminate(self):
        self.worker_pool.terminate()
        self.worker_pool.join()


def multi_processer(jobs, cpus=multiprocessing.cpu_count(), info=False, timeout=10, stop_on_error=True, delay=0.3,
                    worker_pool=None):
    """

    Jobs is expected to be of list/array type and be structured as:
    [  ( functionReference, arguments, key word arguments as dictionary ) , ... ]
    
    stop_on_error: [True]  If an error is encountered the processing of all child processes is to be stopped
                   and the error is raised.
                   [False] If an error is encountered while processing, the error will be ignored and the
                   corresponding place in the result vector will be replaced with False.
    
    timeout:     This is a timeout value to avoid dead lock of processes. If the process takes longer than 
                 the prescribed value a timeOutError will be raised and the child process terminated.
    
    info:        Just add some output regarding progress

    worker_pool:           Use an existing worker pool to avoid creating new for each call
    
    delay:       Delay between submission of jobs
    """

    # Verify that the number of processes is not more that available    
    if cpus > multiprocessing.cpu_count():
        cpus = multiprocessing.cpu_count()
    if len(jobs) < cpus:
        cpus = len(jobs)

    # Start timer
    start_time = time.time()
    
    try:
        #      Assemble the command for each job
        job_args = []
        for job in jobs:
            function = job[0]
            arguments = job[1] 
            kwarguments = job[2]
            
            if (arguments is None) and (kwarguments is None):
                job_args.append([function, [], {}])
            elif arguments is None:
                job_args.append([function, [], kwarguments])            
            elif kwarguments is None:
                job_args.append([function, arguments, {}])            
            else:                 
                job_args.append([function, arguments, kwarguments])
                
    except:
        print(" ERROR: multiProcessor - The received arguments could not be interpreted")
        print("        The data must be on the following form:")
        print("        [  ( functionReference, arguments, key word arguments as dictionary )\n\t , "
              "( myFun,[x,y,z], {'a':2 ,'b':3} ) \n\t ,  (myFun2, None, None),   ]\n")
        raise

    try:
        # Spawn worker Pool
        pool_created = False
        if worker_pool is None:
            worker_pool = multiprocessing.Pool(processes=cpus)
            pool_created = True

        # Submit jobs to queue
        queue = []
        for job_arg in job_args:
            function = job_arg[0]
            arguments = job_arg[1]
            kwarguments = job_arg[2]
            queue.append(worker_pool.apply_async(function, arguments, kwarguments))
            time.sleep(delay)

        # Get results
        results = []
        
        for i, item in enumerate(queue):
            try:   # noinspection PyBroadException
                results.append(item.get(timeout=timeout))
                if info:
                    print(" Completed %s of %s" % (i+1, len(queue)))
                    sys.stdout.flush()
            except multiprocessing.TimeoutError:
                print("\n ERROR: Timeout\n")
                print("        To avoid dead lock when workers do not operate as intended")
                print("        or an unrecoverable error arises a time out time is set.")
                print(" ")
                print("        The default timeout is 10s. ")
                print(" ")
                print("        This parameter can be manually adjusted as an argument when")
                print("        calling this module, append timeout='number of seconds'")
                print("        to adjust when timeout should occur.")

                if stop_on_error:
                    print("\n\n Terminating child processes")
                    worker_pool.terminate()
                    if pool_created:
                        worker_pool.join()
                    print("-Done\n")
                    raise
                else:
                    results.append(False)
                    continue
            except:
                print("\n\n The following problem was encountered:")
                print(sys.exc_info()[0])  # - Exit type:', sys.exc_info()[0]
                print(sys.exc_info()[1])  # - Exit type:', sys.exc_info()[0]
        
                print("Stop on error: ", stop_on_error)
                if stop_on_error:
                    print("\n\n Terminating child processes")
                    worker_pool.terminate()
                    if pool_created:
                        worker_pool.join()
                    print("-Done\n")
                    raise
                else:
                    results.append(False)
                    continue

        # Close queue and kill workers
        if pool_created:
            worker_pool.close()
            worker_pool.join()

        end_time = float(round((time.time() - start_time)*10))/10
        if info:
            print(" Program used %s sub processes with a total duration of: %ss" % (cpus, end_time))
        return results
        
    except:
        print("\n\n The following problem was encountered:")
        print(sys.exc_info()[0])  # - Exit type:', sys.exc_info()[0]
        print(sys.exc_info()[1])  # - Exit type:', sys.exc_info()[0]
        raise


def _loop_func(function, data_list, keyword_data):
    results = []
    for data in data_list:
        try:
            len(data)
        except TypeError:
            data = (data, )
        results.append(function(*data, **keyword_data))
    return results


def process_function(function, data_list, cpus, keyword_data=None, force_multiprocessing=False,
                     info=False, timeout=10, stop_on_error=True, delay=0.3):
    m = MultiProcesser(cpus, info=info, stop_on_error=stop_on_error, delay=delay)
    result = m.process_function(function, data_list, keyword_data, force_multiprocessing=force_multiprocessing, timeout=timeout)
    # m.terminate()
    return result


def apply(function, data_list, cpus, keyword_data=None, axis_split=0, force_multiprocessing=False,
          info=False, timeout=10, stop_on_error=True, delay=0.3, worker_pool=None):
    """
    Multiprocessing one or several arrays given in the *args data in the function function
    :param function:              Function to process

    :param data_list:             A list with one or more iterables of the same length that will be processed in
                                  parallel
    :param keyword_data:          A dict with additional keyword arguments to function that is common for all points

    :param cpus                   int giving the number of cpus to be used

    :param axis_split             if data_list contains multidimensional arrays, axis_split can be given as an int or
                                  a list having the same length as data_list along which axis will be processed. For
                                  one-dimensional arrays, this argument is neglected
    :param force_multiprocessing  if True, a new process will be started even though num_cpus=1

    :param stop_on_error:         [True]  If an error is encountered the processing of all child processes is to be
                                  stopped and the error is raised.
                                  [False] If an error is encountered while processing, the error will be ignored and the
                                  corresponding place in the result vector will be replaced with False.

    :param timeout:               This is a timeout value to avoid dead lock of processes. If the process takes longer
                                  than the prescribed value a timeOutError will be raised and the child process
                                  terminated.

    :param info:                  Just add some output regarding progress

    :param delay:                 Delay between submission of jobs

    :param worker_pool:           Use an existing worker pool to avoid creating new for each call

    :return:                      A numpy array with return values from function
    """
    if keyword_data is None:
        keyword_data = {}
    if cpus > 1 or force_multiprocessing:
        if isinstance(axis_split, Integral):
            axis_split = [axis_split]*len(data_list)
        data_chunks_list = [list() for _ in range(cpus)]
        for data, axis in zip(data_list, axis_split):
            if isinstance(data, np.ndarray):
                if len(data.shape) == 1:
                    axis = 0
                data_chunks = np.array_split(data, cpus, axis=axis)
            else:
                data_chunks = []
                start = 0
                for i in range(cpus):
                    tot_size = len(data)
                    if i < tot_size % cpus:
                        size = tot_size // cpus + 1
                    else:
                        size = tot_size // cpus
                    data_chunks.append(data[start:start + size])
                    start += size
            for i, chunk in enumerate(data_chunks):
                data_chunks_list[i].append(chunk)
        job_list = [(function, data_chunks_list[i], keyword_data) for i in range(cpus)]
        results = multi_processer(job_list, cpus=cpus, info=info, timeout=timeout, stop_on_error=stop_on_error,
                                  delay=delay, worker_pool=worker_pool)
        return np.vstack(results)
    else:
        return function(*data_list, **keyword_data)
