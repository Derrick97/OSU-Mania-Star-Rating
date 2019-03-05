import numpy as np;

#...Why does python 3.x change next to __next__......

# Read metadata from .osu file.
def read_metadata(f, line):
    if "[Metadata]" in line:
        while "[Difficulty]" not in line:
            print(line, end = "")
            line = f.__next__()

# Read mode: key count.
def read_column_number(f, line):
    column_number = -1
    if "CircleSize:" in line:
        temp = line.strip()
        column_number = temp[-1]
        line = f.__next__()
    return string_to_int(column_number)

def string_to_int(str):
    return int(float(str))

def read_Timing_Points(f, line):
    if "[TimingPoints]" in line:
        line = f.__next__()
        params = object_line.split(",")
        offset = string_to_int(params[0])
        # mpb = 60000 / bpm...
        mpb = string_to_int(params[1])
        # meter: number of beats in a measure. aka 节拍.
        meter = string_to_int(params[2])
        # Other parameters are not important for measuring difficulty.


# Main function for parsing note data.
# https://osu.ppy.sh/help/wiki/osu!_File_Formats/Osu_(file_format)
def read_note(f, line, column_number):
    if "[HitObjects]" in line:
        line = f.__next__()
        while line != None:
            parse_hit_object(f, line, column_number)
            line = f.__next__()

# Helper function for read_note().
def parse_hit_object(f, object_line, column_number):
    params = object_line.split(",")
    column = string_to_int((params[0]))
    column_width = int(512/column_number)
    column = int(column / column_width)
    starting_point = int(params[2])
    note_type = int(params[3])
    last_param_chunk = params[5].split(":")
    note_end = int(last_param_chunk[0])

# Parser Class that can be used on other class.
class parser():
    def __init__(self, file_path):
        # Need to find some way to escape \.
        #self.file_path = file_path.replace("\\", "\\\\")
        self.file_path = file_path
        self.column_number = -1
    def process(self):
        with open(self.file_path, "r+") as f:
            try:
                for line in f:
                    read_metadata(f, line)
                    temp = read_column_number(f, line)
                    if temp != -1:
                        self.column_number = temp
                    if self.column_number != -1:
                        read_note(f, line, self.column_number)
            except StopIteration:
                pass
