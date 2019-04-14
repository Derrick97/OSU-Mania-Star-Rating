import numpy as np
import osu_file_parser as osu_parser
import functools
import math


def parse_osu_file(file_path):
    p = osu_parser.parser(file_path)
    p.process()
    return p.get_parsed_data()


class star_calculator():
    def __init__(self):
        self.column_count = []
        self.columns = []
        self.note_starts = []
        self.note_ends = []
        self.note_types = []

    def calculate_SR(self, file_path):
        p = parse_osu_file(file_path)
        self.column_count = p[0]
        self.columns = p[1]
        self.note_starts = p[2]
        self.note_ends = p[3]
        self.note_types = p[4]
        pass

    def calculate_X(self):
        pass

    # Calculate Note value v for X Dimension.
    # v = 2T + 1. T = length of the note.
    def note_value_forX(self):
        vs = []
        for i in range(len(self.note_types)):
            if self.note_types[i] == 128:
                vs.append(2 * (self.note_ends[i] - self.note_starts[i]) + 1)
            # T = 0 for single note.
            else:
                vs.append(1)
        return vs

    def intensity_forX(self):
        delta_t = []
        for i in range(len(self.note_starts) - 1):
            delta_t.append(self.note_starts[i + 1] - self.note_starts[i])
        # Questionable: x = 0.5 here.
        intensity_func = lambda t: 1/(t * (t + 0.3 * math.sqrt(0.5)))
        intensities = list(map(intensity_func, delta_t))
        return intensities

    def calculate_Y(self):
        pass

    def calculate_Z(self):
        pass
