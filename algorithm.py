import numpy as np;
import osu_file_parser as osu_parser
import sys;

def parse_osu_file(file_path):
    p = osu_parser.parser(file_path)
    p.process()

def calculate_SR():
    pass

def calculate_X():
    pass

def calculate_Y():
    pass

def calculate_Z():
    pass

length_of_arguments = len(sys.argv)
file_path = ""
if length_of_arguments != 1:
    file_path = sys.argv[1]
    parse_osu_file(file_path)
