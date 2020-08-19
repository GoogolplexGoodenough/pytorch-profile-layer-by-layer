from __future__ import division

from collections import defaultdict
from functools import lru_cache, reduce
import math
import operator
import torch


def profile(module, display_cpu=True, display_gpu=True):
    assert issubclass(module, torch.nn.Module)
    monkey_patch_init(module)
    return module

def monkey_patch_init(_class):
    old_init = _class.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.profiler = Profiler(self)
        _class.__str__ = self.profiler.__str__
        _class.start_warming_up = self.profiler.start_warming_up
        _class.stop_warming_up = self.profiler.stop_warming_up
        _class.sum_total_time = self.profiler.sum_total_time
        _class.set_metric_times = self.profiler.set_metric_times
    _class.__init__ = new_init

class Profiler(object):
    def __init__(self, module):
        """
        An operation is a graph node that performs computation on tensors.
        """
        self._module = module
        self._events = {
            'forward': defaultdict(Event),
            'backward': defaultdict(Event)}
        self._operations = {}
        self._enable = True

        self.warmming_up = False

        #consume generator
        list(map(self._hook_operation, operations(self._module)))
    
    def set_metric_times(self, metric_times):
        self.metric_times = metric_times

    def start_warming_up(self):
        self.warmming_up = True
        print('Start warming up')

    def stop_warming_up(self):
        self.warmming_up = False
        print('Stop warming up')

    def _hook_operation(self, op):
        def wrapper_call(op, *inputs, **kwargs):
            # Wrapper function to "__call__", with time counter in it.
            if not self._enable:
                return self._operations[op.__class__](op, *inputs, **kwargs)

            if self.warmming_up:
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                torch.cuda.synchronize()
                result = self._operations[op.__class__](op, *inputs, **kwargs)
                torch.cuda.synchronize()


            else:
                import time
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    torch.cuda.synchronize()
                    for i in range(self.metric_times - 1):
                        self._operations[op.__class__](op, *inputs, **kwargs)
                    result = self._operations[op.__class__](op, *inputs, **kwargs)
                    torch.cuda.synchronize()

                prof.function_events.populate_cpu_children()


                self._events['forward'][op] += Event(
                    cpu_time=sum([item.self_cpu_time_total for item in prof.function_events]),
                    gpu_time=sum([item.cuda_time_total - sum([child.cuda_time_total for child in item.cpu_children]) for item in prof.function_events]),
                    # gpu_time=prof.function_events[0].cuda_time,
                    parameters=count_elements(op.parameters()),
                    input_size=count_elements(inputs),
                    input_shape=format_shape(inputs),
                    input_data_type=format_dtype(inputs),
                    hits=self.metric_times)


                for item in prof.function_events:
                    item = []
                prof.function_events = []


                # self._events['forward'][op] += Event(
                #     # cpu_time=sum([item.cpu_time_total - sum(child.cpu_time_total for child in item.cpu_children) for item in prof.function_events]),
                #     cpu_time=sum(prof.total_average().cpu_time),
                #     gpu_time=sum([item.cuda_time_total - sum(child.cuda_time_total for child in item.cpu_children) for item in prof.function_events]),
                #     parameters=count_elements(op.parameters()),
                #     input_size=count_elements(input),
                #     input_shape=format_shape(input),
                #     input_data_type=format_dtype(input),
                #     hits=1)
            
            def backward_pre_hook(*args):
                if not self._enable:
                    return
                self._events['backward'][op].append(time.time())
            #result.grad_fn.register_pre_hook(backward_pre_hook);
            return result

        # monky patch "__call__" with "wrapper_call"  for this operation`
        if op.__class__ not in self._operations:
            self._operations[op.__class__] = op.__class__.__call__
            op.__class__.__call__ = wrapper_call

        #def backward_post_hook(*args):
        #    if not this_profiler.profiling_on:
        #        return
        #    # adds ending time
        #    backward = this_profiler.record['backward']
        #    backward[-1] = backward[-1] + (time.time(),) 
        #op.register_backward_hook(backward_post_hook)   
    
    def sum_total_time(self):
        total_cpu_time = 0
        total_gpu_time = 0
        for module in self._events['forward']:
            total_cpu_time += self._events['forward'][module].cpu_time
            total_gpu_time += self._events['forward'][module].gpu_time

        self.total_cpu_time = total_cpu_time
        self.total_gpu_time = total_gpu_time
        print(self.total_cpu_time, format_time(self.total_cpu_time))
        print(self.total_gpu_time, format_time(self.total_gpu_time))
        pass

    @lru_cache(maxsize=None)
    def get_metrics(self, module):
        if module in self._events['forward']:
            #it's an operation
            return self._events['forward'][module]
        return reduce(operator.add, map(self.get_metrics, module._modules.values()))
    
    def __str__(self, module=None, indentation=0, pre_msg=''):
        tmpstr = ''
        if module is None:
            module = self._module
            tmpstr += Event.header()

        # metrics = self.get_metrics(module)
        metrics = self.get_metrics(module).tostring(self.total_cpu_time, self.total_gpu_time)
    
        if module.__class__ in self._operations:
            return  tmpstr + metrics + ',' '\"' + pre_msg + module.__repr__() + '\"' + '\n'
        
        name = module.__class__.__name__

        for key, sub_module in module._modules.items():
            tmpstr +=  self.__str__(sub_module, indentation+2, pre_msg='(' + key + '): ')

        return tmpstr        

class Event(object):
    def __init__(self, cpu_time=0, gpu_time=0, parameters=0, input_size=0, input_shape=None, input_data_type=None, hits=0, average_cpu = 0, average_cuda = 0, total_cpu_ = 0, total_cuda_ = 0):
        self.cpu_time = cpu_time
        self.gpu_time = gpu_time
        self.parameters = parameters
        self.input_size = input_size
        self.input_shape = input_shape
        self.input_data_type = input_data_type

        self.average_cpu = average_cpu
        self.average_cuda = average_cuda
        self.total_cpu_ = total_cpu_
        self.total_cuda_ = total_cuda_
        self.hits = hits
    
    @classmethod
    def header(cls):
        header = format_columns(['Total CPU Time','Total CUDA Time','Total CPU time%','Total CUDA Time %','Average CPU Time','Average CUDA Time','Parameters','Input Size','Input Shape','Input Data Type','Numbers of Calls','Architecture'])
        return header + '\n'

    def tostring(self, total_cpu_time, total_gpu_time):
        # print(self.input_data_type)
        return format_columns([
                format_time(self.cpu_time),
                format_time(self.gpu_time),
                format_percent(self.cpu_time / total_cpu_time),
                format_percent(self.gpu_time / total_gpu_time),
                format_time(self.cpu_time/self.hits),
                format_time(self.gpu_time/self.hits),
                format_count(self.parameters),
                format_count(self.input_size),
                format_str(self.input_shape),
                format_str(self.input_data_type),
                format_str(self.hits)])
    
    def __add__(self, other):
        # print(format_time(self.cpu_time), format_time(other.cpu_time), format_time(self.cpu_time + other.cpu_time), self.hits, other.hits)
        return Event(
            self.cpu_time + other.cpu_time,
            self.gpu_time + other.gpu_time,
            # self.parameters + other.parameters,
            # self.input_size + other.input_size,
            other.parameters,
            other.input_size,
            other.input_shape,
            other.input_data_type,
            self.hits + other.hits)

    def __radd__(self, other):
        return self.__add__(other)

def format_columns(cols, width=10):
    assert isinstance(cols, list)
    return  ','.join(cols)

def format_time(time_in_ns):
    if not time_in_ns:
        return '-'

    human_powers = ['u','m','']
    power = int(math.log(time_in_ns, 10) // 3)
    return '{:.2f}{}s '.format(
            time_in_ns/1000.**power,
            human_powers[power])
    # return str(time_in_ns)

def format_count(n):
    if not n:
        return '-'

    human_powers = ['','k','m','g']
    power = int(math.log(n, 10) // 3)
    return '{:.2f}{} '.format(
            n/1000.**power,
            human_powers[power])

def operations(module):
    """
    Given a module recursively transverse it
    to find all atomic operations.

    Atomic operations are the nodes in the graph which
    perform computations on the tensors.
    """
    if not len(list(module.children())):
        # nn.Module who doesn't have sub nn.Module, hook it.
        yield module

    for name, sub_module in module.named_children():
        if (isinstance(sub_module, torch.nn.Container)
            or isinstance(sub_module, torch.nn.Sequential)
            or isinstance(sub_module, torch.nn.ModuleList)
            or isinstance(sub_module, torch.nn.Module)):
            # Recursively visit their decendants.
            for op in operations(sub_module): #python2 compatibility
                yield op

def indent(s, indent):
    return '\n'.join((indent* ' ') + line for line in s.split('\n'))

def count_elements(tensors):
    return sum([reduce(operator.mul, t.size()) for t in tensors])

def format_str(s):
    return str(s)

def format_shape(input):
    def get_shape(x):
        if not isinstance(x, str):
            x = str(x)
        begin = x.find('(')
        end = x.find(')')
        s = x[begin + 1: end]
        return '\"' + s + '\"'

    if len(input) == 1:
        return get_shape(input[0].shape)
    if len(input) == 0:
        return None
    if len(input) > 1:
        shapes = [str(t.shape) for t in input]
        shape = get_shape(shapes[0])
        for i in range(1, len(shapes)):
            shape += ', ' + get_shape(shapes[i])
        return shape

def format_dtype(input):
    if len(input) == 1:
        return input[0].dtype
    if len(input) == 0:
        return None
    if len(input) > 1:
        dtypes = [str(t.dtype) for t in input]
        dtype = dtypes[0]
        for i in range(1, len(shapes)):
            dtype += ', ' + dtypes[i]
        return dtype

def format_percent(input):
    return '{:.3%}'.format(input)