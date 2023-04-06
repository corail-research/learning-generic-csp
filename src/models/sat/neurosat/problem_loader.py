import os
import pickle

class ProblemsLoader:
    def __init__(self, directory):
        self.directory = directory
        self.filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory)])
        self.next_file_num = 0

    def has_next(self):
        return self.next_file_num < len(self.filenames)

    def get_next(self):
        if not self.has_next():
            self.reset()
        filename = self.filenames[self.next_file_num]
        print(f"Loading {filename}...")
        with open(filename, 'rb') as f:
            problems = pickle.load(f)
        self.next_file_num += 1
        assert len(problems) > 0
        return problems, filename

    def reset(self):
        self.next_file_num = 0

def init_problems_loader(directory):
    return ProblemsLoader(directory)

def ilit_to_var_sign(x):
    assert abs(x) > 0
    var = abs(x) - 1
    sign = x < 0
    return var, sign

def ilit_to_vlit(x, n_vars):
    assert x != 0
    var, sign = ilit_to_var_sign(x)
    if sign:
        return var + n_vars
    else:
        return var

def shift_ilit(x, offset):
    assert x != 0
    if x > 0:
        return x + offset
    else:
        return x - offset

def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]
