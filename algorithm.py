import numpy as np
import osu_file_parser as osu_parser
from functools import reduce
import math


def parse_osu_file(file_path):
    p = osu_parser.parser(file_path)
    p.process()
    return p.get_parsed_data()

def is_empty(some_list):
    for item in some_list:
        if item != 0:
            return False
    return True

# Helper function for asperity.
def asperity_help(note_counts_wrt_columns):
    note_counts_wrt_columns.sort()
    column_gaps = []
    for i in range(len(note_counts_wrt_columns) - 1):
        column_gaps.append(note_counts_wrt_columns[i + 1] - note_counts_wrt_columns[i])
    return sum(list(map(lambda x: math.sqrt(x), column_gaps))) ** 2


class star_calculator():
    def __init__(self):
        self.column_count = 0
        self.columns = []
        self.note_starts = []
        self.note_ends = []
        self.note_types = []
        self.od = -1

    def calculate_SR(self, file_path):
        p = parse_osu_file(file_path)
        self.column_count = p[0]
        self.columns = p[1]
        self.note_starts = p[2]
        self.note_ends = p[3]
        self.note_types = p[4]
        self.od = p[5]
        # print(self.columns)
        # print(len(self.calculate_asperity()))

    def calculate_X(self):
        pass


    def smoother_forX(self):
        note_value = self.note_value_forX(self)
        asperities = self.calculate_asperity(self)
        intensities = self.intensity_forX(self)
        pass


    # Calculate Note value v for X Dimension.
    # v = 2T + 1. T = length of the note.
    # Return: array of note values for every note.
    def note_value_forX(self):
        vs = []
        for i in range(len(self.note_types)):
            if self.note_types[i] == 128:
                vs.append(2 * (self.note_ends[i] - self.note_starts[i]) + 1)
            # T = 0 for single note.
            else:
                vs.append(1)
        return vs

    # Calculate asperity. The asperity is count inside every [-0.5s, +0.5s] window.
    # Step = 10ms here.
    # I changed it to 1 ms.
    def calculate_asperity(self):
        time_interval_start = self.note_starts[0] - 500
        left_half_columns = right_half_columns = -1
        start_evaluating_note_index = 0
        if (self.column_count % 2 == 0):
            asperities = []
            left_half_columns = right_half_columns = self.column_count / 2
            while time_interval_start <= self.note_starts[-1]:
                local_asperity = self.asperity_in_window(left_half_columns, right_half_columns, time_interval_start, start_evaluating_note_index)
                asperities.append(local_asperity[0])
                start_evaluating_note_index = local_asperity[1]
                time_interval_start += 1
            return asperities
        else:
            asperities_1 = asperities_2 = []
            left_half_columns = (self.column_count - 1) / 2
            right_half_columns = left_half_columns + 1
            while time_interval_start <= self.note_starts[-1]:
                local_asperity = self.asperity_in_window(left_half_columns, right_half_columns, time_interval_start, start_evaluating_note_index)
                asperities_1.append(local_asperity[0])
                local_asperity = self.asperity_in_window(right_half_columns, left_half_columns, time_interval_start, start_evaluating_note_index)
                asperities_2.append(local_asperity[0])
                start_evaluating_note_index = local_asperity[1]
                time_interval_start += 1
            two_sets_of_asperities = np.array([asperities_1, asperities_2])
            return np.average(two_sets_of_asperities, axis = 0).tolist()
    #Technically the array is not the “asperities” defined in the paper but rather “the asperities/N,” N for the number of notes in the window.


    def asperity_in_window(self, left_half_columns, right_half_columns, time_interval_start, start_evaluating_note_index):
        note_counts_wrt_columns = [0] * self.column_count
        next_start_note_index = start_evaluating_note_index

        for i in range(start_evaluating_note_index, len(self.note_starts)):
            if (time_interval_start + 500 > self.note_starts[i] >= time_interval_start - 500):
                count = 0
                note_counts_wrt_columns[self.columns[i]] += 1
                # Step is 10ms, so next time starts from the first note that is included in the next window.
                if (count == 0 and time_interval_start + 1 <= self.note_starts[i]):
                    next_start_note_index = i
                    count += 1
        left_half_columns_int = int(left_half_columns)
        note_counts_wrt_left_columns = note_counts_wrt_columns[:left_half_columns_int]
        note_counts_wrt_right_columns = note_counts_wrt_columns[left_half_columns_int:]
        if (is_empty(note_counts_wrt_left_columns) and is_empty(note_counts_wrt_right_columns)): # If no notes at all:
            return (1/3, next_start_note_index)
        else:
            note_count = sum(note_counts_wrt_left_columns) + sum(note_counts_wrt_right_columns)
            return ((asperity_help(note_counts_wrt_left_columns) + asperity_help(note_counts_wrt_right_columns)) / note_count, next_start_note_index)

    def time_to_note(self, self.note_starts, self.note_ends):
        index = []
        for m in range (0, self.note_ends[len(self.note_starts)-1]+0.001, 0.001): #from t=0
            i=len(note_starts_wrt_columns) - 1
            while note_starts_wrt_columns[i]>m:
                i=i-1
            index.append(i) #can be as low as -1 and as high as len(self.note_starts) - 1
 
    def time_to_note_end(self, self.note_starts, self.note_ends):
        index = []
        for m in range (0, self.note_ends[len(self.note_starts)-1]+0.001, 0.001): #from t=0
            i=len(note_starts_wrt_columns) - 1
            while note_ends_wrt_columns[i]>m:
                i=i-1
            index.append(i) #can be as low as -1 and as high as len(self.note_starts) - 1

    # Calculate f(t) for X.
    # Per column.
    def X_collection(self):
        note_starts_wrt_columns = [[] for x in range(self.column_count) ]
        note_ends_wrt_columns = [[] for x in range(self.column_count) ]
        J = [[] for x in range(self.column_count) ]
        for i in range(len(self.note_starts)):
            note_starts_wrt_columns[self.columns[i]].append(self.note_starts[i])
            note_ends_wrt_columns[self.columns[i]].append(self.note_ends[i])
        for j in range(self.column_count):
            J.append(self, J(note_starts_wrt_columns[j]), note_ends_wrt_columns[j], self.note_starts, self.note_ends, j)
        return J
 
    def time_to_note_per_column(self, self.note_starts, self.note_ends, note_starts_wrt_columns, note_ends_wrt_columns, column):
        D = []
        for i in range (self.column_count):
            index = []
            for m in range (0, note_ends_wrt_column[i][len(note_starts_wrt_columns)-1]+0.001, 0.001): #from t=0
                j=len(note_starts_wrt_columns[i]) - 1
                while note_starts_wrt_columns[i][j]>m:
                    j=j-1
                index.append(j)
            D.append(index)
        return D

    def J(self, note_starts_wrt_columns, note_ends_wrt_columns, self.note_starts, self.note_ends, time_to_note_per_column, column):
        J_collection=[]
        for k in range (self.column_count):
            note_starts_fixed_column=note_starts_wrt_columns[k]
            note_ends_fixed_column=note_ends_wrt_columns[k]
            time_to_note_fixed_column=time_to_note_per_column[k]
            delta_t = []
            for i in range(len(note_starts_fixed_column) - 1):
                delta_t.append(note_starts_fixed_column[i + 1] - note_starts_fixed_column[i])
            x = (64.5 - math.ceil(self.od * 3))/500
            intensity_func = lambda t: 1/(t * (t + 0.3 * math.sqrt(x)))
            intensities = list(map(intensity_func, delta_t))
            time_based=[]
            for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001): #to match the time range
                i=time_to_note_fixed_column[int(m*1000)] #m is in sec, but the time_to_note array is per ms.
                time_based.append(intensities[i])
            vs=[]
            for i in range(len(note_starts_fixed_column) - 1):
                append(1+2*(note_ends_fixed_column[i]-note_starts_fixed_column[i]))
            dist=[]
            for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001): #same, agreememt
                i=time_to_note_fixed_column[int(m*1000)]
                dist.append(vs[i] * (calculate_asperity[int(m*1000)]+5)/5 * intensities[int(m*1000)])
            J_single = []
            for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
                convolution=[]
                for j in range (i-0.5, i+0.5, 0.001):
                    convolution.append(dist_for_ht_time[int(j*1000)])
                indicator = 0.001*sum(convolution)
                if note_starts_fixed_column[0] <= i < note_ends_fixed_column[len(self.note_starts)-1]:
                    j=time_to_note_fixed_column[int(i*1000)]
                    weight = 1/(note_starts_fixed_column[j+1]-note_starts_fixed_column[j])
                else:
                    weight = 1
                J_single.append((indicator)**4 * weight) #this is J to the 4th power times the no. of notes
            J_collection.append(sum(J_single))  
        return (J_collection)

    def X(self, J):
        print((sum(J)/self.column_count)**(1/4))
        return ((sum(J)/self.column_count)**(1/4))

    def calculate_Y(self):
        pass

    def intensity_for_gt_note(self, self.note_starts):
        delta_t=[]
        for i in range(len(self.note_starts) - 1):
            delta_t.append(self.note_starts[i+1] - self.note_starts[i])
        x = (64.5 - math.ceil(self.od * 3))/500
        if delta_t = 0: #jumps/chords
            intensity_func = 1000*(0.08/x*(1-4.5*x))**(1/4) #for numerical estimation of Dirac delta function
        if 0<delta_t & delta_t<=2*x/3:
            intensity_func = lambda t: t**(-1)*(0.08*x**(-1)*(1-18*x**(-1)*(t-x/2)**2))**(1/4)
        else:
            intensity_func = lambda t: t**(-1)*(0.08*x**(-1)*1-18*x*(x/6)**2))**(1/4)
        intensities = list(map(intensity_func, delta_t))
        return intensities

    def note_value_for_gt(self, self.note_starts, self.note_ends):
        vs=[]
        for i in range(len(self.note_starts) - 1):
            ln_parts = []
            for j in range(len(self.note_starts) - 1):
                if self.note_ends[j]>self.note_starts[i] & self.note_ends[j+1]>self.note_starts[j]:
                    ln_parts.append(min(self.note_starts[i+1], self.note_ends[i]) - max(self.note_starts[j], self.note_starts[i])) #length contained for each j from i to i+1
            vs.append(1+2*sum(ln_parts))
        return vs

    def intensity_for_gt_time(self, self.note_starts, intensity_for_gt_note)
        gt=[]
        for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001):
        #1 ms per step
        #+0.001 is necessary as the last notes may be concurrent
            i=time_to_note[int(m*1000)]
            single=[]
            for j in range(max(i-9,0) i+1): #at most 10 concurrent notes w/o stacking
                if self.note_starts_j==self.note_starts_i:
                    single.append(intensity_for_gt_note[j])
            gt.append(sum(single))
        return gt

    def dist_for_gt(self, self.note_starts, self.note_ends, note_value_for_gt, calculate_asperity, intensity_for_gt_time): #...What are the variables called?
        dist=[]
        for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001):
            i=time_to_note[int(m*1000)]
            dist.append(note_value_for_gt[i] * (calculate_asperity[int(m*1000)]+5)/5 * intensity_for_gt_time[int(m*1000)])
        return dist

    def smoother_for_gt(self, dist_for_gt_time):
        indicators=[]
        for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
            convolution=[]
            for j in range (i-0.5, i+0.5, 0.001):
                convolution.append(dist_for_gt[int(j*1000)])
            indicators.append(0.001*sum(convolution))
        return indicators

    def intensity_for_ht_note(self, self.note_ends):
        delta_t=[]
        for i in range(len(self.note_ends) - 1):
            delta_t.append(self.note_ends[i+1] - self.note_ends[i])
        x = (64.5 - math.ceil(self.od * 3))/500
        intensity_func = lambda t: 0.08*t**(-0.5)/x
        intensities = list(map(intensity_func, delta_t))
        return intensities

    def note_value_for_ht(self, self.note_starts, self.note_ends):
        vs=[]
        for i in range(len(self.note_ends) - 1):
            append(1+2*(self.note_ends[i]-self.note_starts[i]))
        return vs

    def intensity_for_ht_time(self, self.note_ends, intensity_for_ht_note)
        ht=[]
        for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001):
            i=time_to_note_end[int(m*1000)]
            single=[]
            for j in range(max(i-9,0) i+1):
                if self.note_starts_j==self.note_starts_i:
                    single.append(intensity_for_ht_note[j])
            ht.append(sum(single))
        return ht

    def dist_for_ht(self, self.note_starts, self.note_ends, note_value_for_ht, calculate_asperity, intensity_for_ht_time):
        dist=[]
        for m in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1]+0.001, 0.001):
            i=time_to_note_end[int(m*1000)]
            dist.append(note_value_for_ht[i] * (calculate_asperity[int(m*1000)]+5)/5 * intensity_for_ht_time[int(m*1000)])
        return dist

    def smoother_for_ht(self, dist_for_ht_time):
        indicators=[]
        for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
            convolution=[]
            for j in range (i-0.5, i+0.5, 0.001):
                convolution.append(dist_for_ht[int(j*1000)])
            indicators.append(0.001*sum(convolution))
        return indicators

    def smoother_forY(self, smoother_for_gt, smoother_for_ht):
        indicators=[]
        for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
            indicators.append(smoother_for_gt[int(i*1000)]+smoother_for_ht[int(i*1000)])
        return indicators

    def weight_forY(self, self.note_starts) #preparation for taking the definite integral
        #essentially the transformation from dt to dn.
        sequence=[]
        for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
            if i in range (self.note_starts[0], self.note_ends[len(self.note_starts)-1], 0.001):
                if self.note_starts.count(i)<=1:
                    j=time_to_note[int(i*1000)]
                    sequence.append(1/(self.note_starts[j+1]-self.note_starts[j]))
                if self.note_starts.count(i)>1:
                    j=time_to_note[int(i*1000)]
                    sequence.append(1/(self.note_starts[j+1]-self.note_starts[j]) + 1000 * (self.note_starts.count(i) - 1))
            else:
                sequence.append(1)

    def Y(self, smoother_forY, weight_forY):
        Y_set=[]
        for i in range (self.note_starts[0]-0.499, self.note_ends[len(self.note_starts)-1]+0.501, 0.001):
            Y_set.append((smoother_forY[int(i*1000)])**4 * weight_forY[int(i*1000)])
        print ((sum(Y_set)/len(self.note_starts))**(1/4))
        return ((sum(Y_set)/len(self.note_starts))**(1/4)) #in the denominator +1 is unnecessary.
        #Not rewarding extremely short maps

    def calculate_Z(self):
        pass
