import numpy as np;
import sys;
#...Why does python 3.x change next to __next__......

def read_metadata(f, line):
    if "[Metadata]" in line:
        while "[Difficulty]" not in line:
            print(line, end = "")
            line = f.__next__()

def read_note(f, line):
    if "[HitObjects]" in line:
        line = f.__next__()
        while line != None:
            parse_hit_object(line)
            line = f.__next__()

def parse_hit_object(object_line):
    params = object_line.split(",")
    column = int(params[0])
    # 64 -- column 1.  192 -- column2. etc.
    column = (column + 64) / 128
    starting_point = int(params[2])
    note_type = int(params[3])
    last_param_chunk = params[5].split(":")
    note_end = int(last_param_chunk[0])

class parser():
    def __init__(self, file_path):
        # Need to find some way to escape \. 
        self.file_path = file_path.replace("\\", "\\\\")
    def process(self):
        with open(self.file_path, "r+") as f:
            try:
                for line in f:
                    read_metadata(f, line)
                    read_note(f, line)
            except StopIteration:
                pass
