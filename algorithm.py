import numpy as np
import osu_file_parser as osu_parser
from functools import reduce
import math
import itertools

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

class star_calculator:
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
        self.od = 8
        # print(self.columns)
        self.asperity = self.calculate_asperity()
        # print(len(self.calculate_asperity()))

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

    # Calculate asperity. The asperity is count inside every [-0.5s, +0.5s) window.
    # Step = 1 ms here.

    def time_to_note(self, note_starts, note_ends):
        index = []
        for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_starts)-1])+1): #from t=0
            i=len(self.note_starts) - 1
            while self.note_starts[i]>m and i>=0:
                i=i-1
            index.append(i) #can be as low as -1 and as high as len(self.note_starts) - 1
        return index
 
    def time_to_note_end(self, note_starts, note_ends):
        index = []
        for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_starts)-1])+1): #from t=0
            i=len(self.note_starts) - 1
            while max(self.note_starts[i], self.note_ends[i])>m and i>=0:
                i=i-1
            index.append(i) #can be as low as -1 and as high as len(self.note_starts) - 1
        return index
    
    def X_collection(self):
        note_starts_wrt_columns = [[] for x in range(self.column_count) ] 
        note_ends_wrt_columns = [[] for x in range(self.column_count) ]
        for i in range(self.column_count):
            time_set=[]
            for j in range(len(self.note_starts)):
                if self.columns[j]==i:
                    time_set.append(self.note_starts[j])
            time_set_ends=[]
            for j in range(len(self.note_starts)):
                if self.columns[j]==i:
                    time_set_ends.append(self.note_ends[j])
            for k in range(len(time_set)):
                note_starts_wrt_columns[i].append(time_set[k])
                note_ends_wrt_columns[i].append(time_set_ends[k])
        return [note_starts_wrt_columns, note_ends_wrt_columns]

    def calculate_asperity(self):
        note_starts_wrt_columns = [[] for x in range(self.column_count) ] 
        note_ends_wrt_columns = [[] for x in range(self.column_count) ]
        for i in range(self.column_count):
            time_set=[]
            for j in range(len(self.note_starts)):
                if self.columns[j]==i:
                    time_set.append(self.note_starts[j])
            time_set_ends=[]
            for j in range(len(self.note_starts)):
                if self.columns[j]==i:
                    time_set_ends.append(self.note_ends[j])
            for k in range(len(time_set)):
                note_starts_wrt_columns[i].append(time_set[k])
                note_ends_wrt_columns[i].append(time_set_ends[k])

        D = []
        for i in range(self.column_count):
            index = []
            for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_ends)-1])): #from t=0   
                # j=len(note_starts_wrt_columns[i]) - 1   
                # while note_starts_wrt_columns[i][j]>m and j>=0:
                #     j=j-1
                l = 0
                r = len(note_starts_wrt_columns[i]) - 1
                while l <= r: 
                    mid = int(l + (r - l)/2)           
                    if note_starts_wrt_columns[i][mid] == m: 
                        break
        
                    elif note_starts_wrt_columns[i][mid] < m: 
                        l = mid + 1
                    else: 
                        r = mid - 1
                while note_starts_wrt_columns[i][mid]>m and mid>=0:
                    mid=mid-1  
                index.append(mid)
                if math.floor(100*(m+1)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*m)/self.note_starts[len(self.note_starts) - 1]):
                    print("Index " + str(i+1) + "/" + str(self.column_count) + " " + str(math.floor(100*m/self.note_starts[len(self.note_starts) - 1])+1) + "%  ", end='\r')
            D.append(index)

        asp = []
        for i in range(0, self.note_starts[len(self.note_starts)-1]):
            asperity = (1/2)**(1/3)
            gap = []
            for k in range(self.column_count):
                if D[k][i]>0:
                    gap.append(note_starts_wrt_columns[k][D[k][i]] - note_starts_wrt_columns[k][D[k][i] - 1])
                else:
                    gap.append(0)
            for k in range(self.column_count - 1):
                if gap[k] < 300 and gap[k+1]<300:
                    gap_difference = abs(gap[k+1]-gap[k])
                else:
                    gap_difference = abs(gap[k+1]-gap[k]) + max(gap[k], gap[k+1])-300
                if gap_difference < 20:
                    asperity *= (min(0.75 + 0.0005 * max(gap[k], gap[k+1]), 1))**(1/3)
                elif gap_difference < 70:
                    asperity *= (min(0.005*gap_difference + 0.65 + 0.0005 * max(gap[k], gap[k+1]), 1))**(1/3)
            asp.append(asperity)
            if math.floor(100*(i+1)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
                print("Asperity " + str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%    ", end='\r')
        print("Asperity 100%    ", end='\r')
        return asp
    # def calculate_asperity(self):
    #     asp = []
    #     step=99
    #     if self.note_starts[len(self.note_starts)-1]>300000:
    #         step+=int((self.note_starts[len(self.note_starts)-1]-300000)/3000)
    #     if step>299:
    #         step=299
    #     for i in range (0, self.note_starts[len(self.note_starts)-1], step):
    #         if math.floor(100*(i+step)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
    #             print(str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%", end='\r')
    #         count_per_column = []
    #         for c in range (self.column_count):
    #             notes_in_this_columnwindow = []
    #             for j in range (max(0, i-500), min(self.note_starts[len(self.note_starts)-1], i+500)):
    #                 if j in self.note_starts: #if time j has at least a note on it
    #                     for x in range (len(self.note_starts) - 1):
    #                         if self.columns[x] == c and self.note_starts[x] == j:
    #                             notes_in_this_columnwindow.append(1)
    #                         else:
    #                             pass
    #                 else:
    #                     pass
    #             count_per_column.append(len(notes_in_this_columnwindow))
    #         if (self.column_count % 2 == 0):
    #             gap1 = []
    #             count_left = []
    #             for i in range (0, int(self.column_count*0.5)):
    #                 count_left.append(count_per_column[i])
    #             count_left_sorted = sorted(count_left)
    #             gap1.append(count_left_sorted[0] ** (1/2)) # Already sqrt. No need to sqrt later.
    #             for i in range(len(count_left_sorted)-1):
    #                 gap1.append((count_left_sorted[i+1] - count_left_sorted[i]) ** (1/2))
    #             left_asp = sum(gap1) ** 2
    #             gap2 = []
    #             count_right = []
    #             for i in range (int(self.column_count*0.5), self.column_count):
    #                 count_right.append(count_per_column[i])
    #             count_right_sorted = sorted(count_right)
    #             gap2.append(count_right_sorted[0] ** (1/2))
    #             for i in range(len(count_right_sorted)-1):
    #                 gap2.append((count_right_sorted[i+1] - count_right_sorted[i]) ** (1/2))
    #             right_asp = sum(gap2) ** 2
    #             if sum(count_per_column) == 0:
    #                 asp.append(1/3)
    #             else:
    #                 asp.append(1/2 * (left_asp + right_asp) / (sum(count_per_column)))  
    #         else:
    #             gap1 = []
    #             count_left = []
    #             for i in range (0, int(self.column_count*0.5)):
    #                 count_left.append(count_per_column[i])
    #             count_left_sorted = sorted(count_left)
    #             gap1.append(count_left_sorted[0] ** (1/2))
    #             for i in range(len(count_left_sorted)-1):
    #                 gap1.append((count_left_sorted[i+1] - count_left_sorted[i]) ** (1/2))
    #             left_asp = sum(gap1) ** 2
    #             gap2 = []
    #             count_right = []
    #             for i in range (int(self.column_count*0.5)+1, self.column_count):
    #                 count_right.append(count_per_column[i])
    #             count_right_sorted = sorted(count_right)
    #             gap2.append(count_right_sorted[0] ** (1/2))
    #             for i in range(len(count_right_sorted)-1):
    #                 gap2.append((count_right_sorted[i+1] - count_right_sorted[i]) ** (1/2))
    #             right_asp = sum(gap2) ** 2
    #             if sum(count_per_column) - count_per_column[int((len(count_per_column)+1)/2)] == 0:
    #                 asp.append(1/3)
    #             else:
    #                 asp.append(1/2 * (left_asp + right_asp) / (sum(count_per_column) - count_per_column[int((len(count_per_column)+1)/2)]))      
    #     asp = list(itertools.chain.from_iterable(itertools.repeat(x, step) for x in asp))    
    #     print ("Asperity OK")
    #     return asp
    # Calculate f(t) for X.
    # Per column.
    
    def time_to_note_per_column(self, note_starts, note_ends, note_starts_wrt_columns, note_ends_wrt_columns):
        D = []
        for i in range(self.column_count):
            index = []
            for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_ends)-1])): #from t=0   
                # j=len(note_starts_wrt_columns[i]) - 1   
                # while note_starts_wrt_columns[i][j]>m and j>=0:
                #     j=j-1
                l = 0
                r = len(note_starts_wrt_columns[i]) - 1
                while l <= r: 
                    mid = int(l + (r - l)/2)           
                    if note_starts_wrt_columns[i][mid] == m: 
                        break
        
                    elif note_starts_wrt_columns[i][mid] < m: 
                        l = mid + 1
                    else: 
                        r = mid - 1
                while note_starts_wrt_columns[i][mid]>m and mid>=0:
                    mid=mid-1  
                index.append(mid)
                if math.floor(100*(m+1)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*m)/self.note_starts[len(self.note_starts) - 1]):
                    print("Index " + str(i+1) + "/" + str(self.column_count) + " " + str(math.floor(100*m/self.note_starts[len(self.note_starts) - 1])+1) + "%  ", end='\r')
            D.append(index)
        # print (D)
        return D

    def J(self, note_starts_wrt_columns, note_ends_wrt_columns, note_starts, note_ends, time_to_note_per_column, asperity):
        J_collection=[]
        for k in range(self.column_count):
            note_starts_fixed_column=note_starts_wrt_columns[k]
            note_ends_fixed_column=note_ends_wrt_columns[k]
            time_to_note_fixed_column=time_to_note_per_column[k]
            delta_t = []
            for i in range(len(note_starts_fixed_column) - 1):
                delta_t.append(note_starts_fixed_column[i + 1] - note_starts_fixed_column[i])
            x = 0.3 * ((64.5 - math.ceil(8 * 3))/500)**0.5
            
            intensity_func = lambda t: 1/((t/1000) * ((t/1000) + 0.2 * math.sqrt(x)))
            intensities = list(map(intensity_func, delta_t))
            time_based=[]
            for m in range(0, self.note_starts[len(self.note_starts)-1]):
                i=time_to_note_fixed_column[m] #m is in sec, but the time_to_note array is per ms.
                if m in range(note_starts_fixed_column[0], note_starts_fixed_column[len(note_starts_fixed_column)-2]):
                    time_based.append(intensities[i])
                else:
                    time_based.append(0)
            nv=[]
            for p in range(len(note_starts_fixed_column)):
                if note_ends_fixed_column[p]<=note_starts_fixed_column[p]:
                    length = 0
                elif note_ends_fixed_column[p]-note_starts_fixed_column[p]<200:
                    length = 0.5*(note_ends_fixed_column[p]-note_starts_fixed_column[p])
                else:
                    length = note_ends_fixed_column[p]-note_starts_fixed_column[p] - 100
                nv.append(1+0.0065*length)
            dist=[]
            nvt=[]
            for m in range(0, self.note_starts[len(self.note_starts)-1]):
                if m in range(note_starts_fixed_column[0], note_starts_fixed_column[len(note_starts_fixed_column)-2]):
                    nvt.append(nv[time_to_note_fixed_column[m]])
                else:
                    nvt.append(0)
            for m in range (0, self.note_starts[len(self.note_starts)-1]):
                temp=nvt[m] * (self.asperity[m]) * time_based[m]
                dist.append(temp)
            J_single = []
            for i in range(0, self.note_starts[len(self.note_starts)-1]+200, 49):
                if math.floor(100*(i+49)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
                    print("Column " + str(k+1) + "/" + str(self.column_count) + " " + str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%  ", end='\r')
                convolution=[]
                for j in range (max(0,i-200), min(i+200, self.note_starts[len(self.note_starts)-1])):
                    if j in range(0, self.note_starts[len(self.note_starts)-1]):
                        convolution.append(dist[j])
                    else:
                        convolution.append(0)
                indicator = 0.0025 * sum(convolution)
                if note_starts_fixed_column[0] <= i < note_starts_fixed_column[len(note_starts_fixed_column)-1]:
                    j=time_to_note_fixed_column[i]
                    weight = 1/(note_starts_fixed_column[j+1]-note_starts_fixed_column[j])
                else:
                    weight = 1/(note_starts_fixed_column[0] + self.note_starts[len(self.note_starts)-1]+500 - note_starts_fixed_column[len(note_starts_fixed_column)-1])
                J_single.append((indicator)**4 * weight) #this is J to the 4th power times the no. of notes
            J_collection.append(49*sum(J_single))
        return J_collection

    def X(self, J):
        # print ("X Dimension:", end=' ')
        # print ((sum(J)/len(self.note_starts))**(1/4)) #paper中需要把X-dimension最终步骤相应修改
        return ((sum(J)/len(self.note_starts))**(1/4))

    def calculate_Y(self, note_starts, note_ends, asperity):
        ttn = []
        for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_starts)-1])+1): #from t=0
            i=len(self.note_starts) - 1
            # while self.note_starts[i]>m and i>=0:
            #     i=i-1
            l = 0
            r = len(self.note_starts) - 1
            while l <= r: 
                mid = int(l + (r - l)/2)           
                if self.note_starts[mid] == m: 
                    j = mid
                    break
    
                elif self.note_starts[mid] < m: 
                    l = mid + 1
                else: 
                    r = mid - 1
            if self.note_starts[mid]>m:
                while self.note_starts[mid]>m and mid>=0:
                    mid-=1 
            if mid<len(self.note_starts) and self.note_starts[mid]<=m:
                while mid<len(self.note_starts) and self.note_starts[mid]<=m: 
                    mid+=1
                mid-=1
            ttn.append(mid) #can be as low as -1 and as high as len(self.note_starts) - 1
            if math.floor(100*(m+1)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*m)/self.note_starts[len(self.note_starts) - 1]):
                print("g-index " + str(math.floor(100*m/self.note_starts[len(self.note_starts) - 1])+1) + "%      ", end='\r')

        delta_t=[]
        for i in range(len(self.note_starts) - 1):
            delta_t.append(self.note_starts[i+1] - self.note_starts[i])
        x = 0.3 * ((64.5 - math.ceil(self.od * 3))/500)**0.5
        
        intensity_for_gt_note=[]
        for i in range(len(self.note_starts) - 1):
            if delta_t[i] != 0:
                bpm = 15000/delta_t[i]
            if delta_t[i] == 0:
                bpm = 0
            bpm_booster = 1
            if 180<bpm<340:
                bpm_booster = 1 + 0.2 * (bpm-180)**3 * (bpm-340)**6 * 10**(-18)
            if delta_t[i] == 0: #jumps/chords
                intensity_for_gt_note.append(1000*(0.08/x*(1-4.5*x))**(1/4)) #for numerical estimation of Dirac delta function
            elif 0<delta_t[i] and delta_t[i]<=2*x/3:
                intensity_for_gt_note.append((delta_t[i]/1000)**(-1)*(0.08*x**(-1)*(1-8*x**(-1)*((delta_t[i]/1000)-x/2)**2))**(1/4)*bpm_booster)
            else:
                intensity_for_gt_note.append((delta_t[i]/1000)**(-1)*(0.08*x**(-1)*(1-8*x**(-1)*(x/6)**2))**(1/4)*bpm_booster)
        # print (intensity_for_gt_note)

        note_value_for_gt=[]
        for i in range(len(self.note_starts) - 1):
            ln_parts = []
            for j in range(len(self.note_starts) - 1):
                if max(self.note_starts[j], self.note_ends[j])>self.note_starts[i] and note_starts[j]<note_starts[i+1]:
                    if min(self.note_starts[i+1], max(self.note_starts[j], self.note_ends[j])) - max(self.note_starts[j], self.note_starts[i]) < 100:
                        length = 0.5*(min(self.note_starts[i+1], max(self.note_starts[j], self.note_ends[j])) - max(self.note_starts[j], self.note_starts[i]))
                    else:
                        length = min(self.note_starts[i+1], max(self.note_starts[j], self.note_ends[j])) - max(self.note_starts[j], self.note_starts[i]) - 50
                    ln_parts.append(length) #length contained for each j from i to i+1
            ln_parts.sort(reverse=True)        
            sum_ln_parts=0    
            for x in range(len(ln_parts)):
                sum_ln_parts += ln_parts[x]*0.3**(x)
            note_value_for_gt.append(1+0.0065*sum_ln_parts)
        # print (note_value_for_gt)

        intensity_for_gt_time=[]
        in_starts = []
        in_ends = []

        note_real_ends = []
        for i in range(len(self.note_starts)):
            note_real_ends.append(max(self.note_starts[i], self.note_ends[i]))
        
        note_ends_sequence = []
        for i in range(len(self.note_starts)):
            if self.note_ends[i] == 0:
                pass
            else:
                note_ends_sequence.append(self.note_ends[i])
        note_ends_sorted = sorted(note_ends_sequence)
        if len(note_ends_sorted) != 0:
            length_ = max(self.note_starts[len(self.note_starts)-1], note_ends_sorted[len(note_ends_sorted)-1])
        else:
            length_ = self.note_starts[len(self.note_starts)-1]
        ln_ends = sorted(self.note_ends)
        for m in range (length_):
            cond = False
            l = 0
            r = len(self.note_starts) - 1
            while l <= r: 
                mid = int(l + (r - l)/2)           
                if self.note_starts[mid] == m: 
                    cond = True
                    break
                elif self.note_starts[mid] < m: 
                    l = mid + 1
                else: 
                    r = mid - 1
            in_starts.append(cond)

            cond2 = False
            l = 0
            r = len(ln_ends) - 1
            while l <= r: 
                mid = int(l + (r - l)/2)           
                if ln_ends[mid] == m: 
                    cond2 = True
                    break
                elif ln_ends[mid] < m: 
                    l = mid + 1
                else: 
                    r = mid - 1
            in_ends.append(cond2)


        # print (note_ends_sorted)
        for m in range (self.note_starts[len(self.note_starts)-1]):
        #1 ms per step
        #+0.001 is necessary as the last notes may be concurrent
            i=ttn[m]
            single=[]
            if in_starts[m]:
                for j in range(max(i-9,0), min(i+1, len(self.note_starts)-1)): #at most 10 concurrent notes w/o stacking
                    if self.note_starts[j]==self.note_starts[i]:
                        single.append(intensity_for_gt_note[j])
                intensity_for_gt_time.append(sum(single))
            else:
                if m in range(self.note_starts[len(self.note_starts)-1]):
                    intensity_for_gt_time.append(intensity_for_gt_note[i])
                else:
                    intensity_for_gt_time.append(0)   
        # print (intensity_for_gt_time[max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_starts)-1])])
        # print (len(intensity_for_gt_time))  
        
        dist=[]
        for m in range (0, self.note_starts[len(self.note_starts)-1]): #之后要改成max(starts, ends), 10改成1
            i=ttn[m]
            if m in range (self.note_starts[0], self.note_starts[len(self.note_starts)-1]) and not in_starts[m]:
                temp=note_value_for_gt[i] * (self.asperity[m]) * intensity_for_gt_time[m] #paper中需要更新相关定义
                dist.append(temp)
            elif m in range (self.note_starts[0], self.note_starts[len(self.note_starts)-1]):
                temp=(self.asperity[m]) * intensity_for_gt_time[m] #note value assumed to be 1 at the point of multiple notes to avoid explosion
                dist.append(temp)
            else:
                dist.append(0)

        indicator_for_gt=[]
        for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 49):
            convolution=[]
            for j in range (max(0,i-500), min(i+500, self.note_starts[len(self.note_starts)-1])): #之后要改成max(starts, ends)
                if j in range(0, self.note_starts[len(self.note_starts)-1]): #之后要改成max(starts, ends), 10改成1
                    convolution.append(dist[j])
                else:
                    convolution.append(0)
            if math.floor(100*(i+49)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
                print("g(t) " + str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%         ", end='\r')
            indicator_for_gt.append(0.001 * sum(convolution))
            # if i>self.note_starts[len(self.note_starts)-1]+450:
            #     print ("Indicator for g(t) OK")
        
        # for x in range (0, self.note_starts[len(self.note_starts)-1], 100):
        #     print (indicator_for_gt[x])
        
        if len(note_ends_sorted) != 0:
            delta_t_end = []
            for i in range(len(note_ends_sorted) - 1):
                delta_t_end.append(note_ends_sorted[i+1] - note_ends_sorted[i])
            x = 0.3 * ((64.5 - math.ceil(self.od * 3))/500)**0.5
            

            intensity_for_ht_note=[]
            for i in range(len(note_ends_sorted) - 1):
                if delta_t_end[i] == 0:
                    intensity_for_ht_note.append(0.08/0.001**(0.5)/x) #approximation for the Dirac Delta function
                else:
                    intensity_for_ht_note.append(0.08/(delta_t_end[i]/1000)**(0.5)/x)

            #ttne treats regular notes as long notes with zero length, while ttne2 only considers long notes
            ttne2 = []          
            for m in range (note_ends_sorted[len(note_ends_sorted)-1]): #from t=0
                i=len(note_ends_sorted)-1
                # while note_ends_sorted[i]>m and i>=0:
                #     i=i-1
                l = 0
                r = len(note_ends_sorted) - 1
                while l <= r: 
                    mid = int(l + (r - l)/2)           
                    if note_ends_sorted[mid] == m: 
                        j = mid
                        break
        
                    elif note_ends_sorted[mid] < m: 
                        l = mid + 1
                    else: 
                        r = mid - 1
                if note_ends_sorted[mid]>m:
                    while note_ends_sorted[mid]>m and mid>=0:
                        mid-=1 
                if mid<len(note_ends_sorted) and note_ends_sorted[mid]<=m:
                    while mid<len(note_ends_sorted) and note_ends_sorted[mid]<=m: 
                        mid+=1
                    mid-=1  
                ttne2.append(mid) #can be as low as -1 and as high as len(self.note_starts) - 1
                if math.floor(100*(m+1)/note_ends_sorted[len(note_ends_sorted) - 1]) > math.floor((100*m)/note_ends_sorted[len(note_ends_sorted) - 1]):
                    print("h-index " + str(math.floor(100*m/note_ends_sorted[len(note_ends_sorted) - 1])+1) + "%      ", end='\r')

            intensity_for_ht_time=[]
            for m in range (note_ends_sorted[len(note_ends_sorted)-1]):
            #1 ms per step
                i=ttne2[m]
                single=[]
                if in_ends[m] and m != 0:
                    for j in range(max(i-9,0), min(i+1, len(self.note_starts)-1)): #at most 10 concurrent notes w/o stacking
                        if self.note_ends[j]==self.note_ends[i]:
                            single.append(intensity_for_ht_note[j])
                    intensity_for_ht_time.append(sum(single))
                elif m in range (note_ends_sorted[0], note_ends_sorted[len(note_ends_sorted)-1]):
                    intensity_for_ht_time.append(intensity_for_ht_note[i])
                else:
                    intensity_for_ht_time.append(0)
            # print (len(intensity_for_ht_time))  
            
            note_value_for_ht=[]
            for i in range(len(note_ends_sorted) - 1):
                j = len(self.note_starts) - 1
                while self.note_ends[j] != note_ends_sorted[i]:
                    j = j-1   
                length = min(max(0.0065*(note_ends_sorted[i]-self.note_starts[j]), 0), 10)
                # print(str(i) + " " + str(length))
                note_value_for_ht.append(1 + length) #paper修改note_value定义！
                #无须担心多押同时放问题
            # print (note_value_for_ht)

            dist_for_ht = []
            for m in range (note_ends_sorted[len(note_ends_sorted)-1]):
                i=ttne2[m]
                if m in range (note_ends_sorted[0], min(note_ends_sorted[len(note_ends_sorted)-1], self.note_starts[len(self.note_starts)-1])) and not in_ends[m]:
                    temp=note_value_for_ht[i] * (self.asperity[m]) * intensity_for_ht_time[m]
                    dist_for_ht.append(temp)  
                elif m in range (note_ends_sorted[0], min(note_ends_sorted[len(note_ends_sorted)-1], self.note_starts[len(self.note_starts)-1])):
                    temp = (self.asperity[m]) * intensity_for_ht_time[m] #should be changed in the paper
                    dist_for_ht.append(temp)
                else:
                    dist.append(0)

            indicator_for_ht=[]
            for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 49):
                convolution=[]
                if math.floor(100*(i+49)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
                    print("h(t) " + str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%         ", end='\r')
                for j in range (max(0,i-500), min(i+500, self.note_starts[len(self.note_starts)-1])): #之后要改成max(starts, ends)
                    if j in range(0, len(dist_for_ht)): #之后要改成max(starts, ends), 10改成1
                        convolution.append(dist_for_ht[j])
                    else:
                        convolution.append(0)
                indicator_for_ht.append(0.001 * sum(convolution))
        
        else:
            indicator_for_ht=[]
            for i in range (max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_starts)-1]) + 500):
                indicator_for_ht.append(0)
            pass

        indicator = []
        for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 49):
            indicator.append(indicator_for_gt[int(i/49)] + indicator_for_ht[int(i/49)]) #paper should be changed
            # if i%4900==0:
                # print(str(i) + ": " + str(indicator_for_gt[int(i/49)]) + " " + str(indicator_for_ht[int(i/49)]))

        # for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 1000):
        #     print (indicator_for_gt[i], indicator[i])
        
        weight = []
        for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 49):
            if i in range (len(ttn) - 1):
                n = ttn[i]
            else:
                n = len(self.note_starts) - 1
            if self.note_starts[0] <= i and i < self.note_starts[len(self.note_starts)-1]:
                for j in range(max(n-9,0), min(n+1, len(self.note_starts)-1)): #at most 10 concurrent notes w/o stacking
                    single = []
                    if self.note_starts[j]==i: #记录押数，0与1押实质等价    
                       single.append(1)
                weight.append(max(sum(single)-1, 0) + 1/(self.note_starts[n+1] - self.note_starts[n]))
            else:
                weight.append(1/(self.note_starts[0] + 500))

        Y_set = []
        for i in range (0, self.note_starts[len(self.note_starts)-1] + 500, 49):
            Y_set.append(indicator[int(i/49)]**8 * weight[int(i/49)])
        
        Y = (49*sum(Y_set)/len(self.note_starts))**(1/8)
        # print ("Y Dimension:", end=' ')
        # print (Y)
        return Y

    def Z_collection(self):
        note_starts_wrt_pairs = [[] for x in range(self.column_count+1) ] 
        note_ends_wrt_pairs = [[] for x in range(self.column_count+1) ]
        for i in range(self.column_count+1):
            time_set=[]
            for j in range(len(self.note_starts)):
                if self.columns[j]==i or self.columns[j]==i-1:
                    time_set.append(self.note_starts[j])
            time_set_ends=[] #this is real end
            for j in range(len(self.note_starts)):
                if self.columns[j]==i or self.columns[j]==i-1:
                    time_set_ends.append(max(self.note_ends[j], self.note_starts[j]))
            for k in range(len(time_set)):
                note_starts_wrt_pairs[i].append(time_set[k])
                note_ends_wrt_pairs[i].append(time_set_ends[k])
        return [note_starts_wrt_pairs, note_ends_wrt_pairs]

    def time_to_note_per_pair(self, note_starts, note_ends, note_starts_wrt_pairs, note_ends_wrt_pairs):
        E = []
        for i in range(self.column_count+1):
            index = []
            for m in range(0, max(self.note_starts[len(self.note_starts)-1], self.note_ends[len(self.note_ends)-1])): #from t=0   
                # j=len(note_starts_wrt_pairs[i]) - 1   
                # while note_starts_wrt_pairs[i][j]>m and j>=0:
                #     j=j-1
                l = 0
                r = len(note_starts_wrt_pairs[i]) - 1
                while l <= r: 
                    mid = int(l + (r - l)/2)           
                    if note_starts_wrt_pairs[i][mid] == m: 
                        break
        
                    elif note_starts_wrt_pairs[i][mid] < m: 
                        l = mid + 1
                    else: 
                        r = mid - 1
                if note_starts_wrt_pairs[i][mid]>m:
                    while note_starts_wrt_pairs[i][mid]>m and mid>=0:
                        mid-=1 
                if mid<len(note_starts_wrt_pairs[i]) and note_starts_wrt_pairs[i][mid]<=m:
                    while mid<len(note_starts_wrt_pairs[i]) and note_starts_wrt_pairs[i][mid]<=m: 
                        mid+=1
                    mid-=1
                index.append(mid)
                if math.floor(100*(m+1)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*m)/self.note_starts[len(self.note_starts) - 1]):
                    print("Index " + str(i+1) + "/" + str(self.column_count+1) + " " + str(math.floor(100*m/self.note_starts[len(self.note_starts) - 1])+1) + "%  ", end='\r')
            E.append(index)
        return E

    def O(self, note_starts_wrt_pairs, note_ends_wrt_pairs, note_starts, note_ends, time_to_note_per_pair, asperity):
        O_collection=[]
        for k in range(self.column_count+1):
            note_starts_fixed_pair=note_starts_wrt_pairs[k]
            note_ends_fixed_pair=note_ends_wrt_pairs[k] #real ends
            time_to_note_fixed_pair=time_to_note_per_pair[k]

            A1 = [0.075, 0.075]
            A2 = [0.125, 0.05, 0.125]
            A3 = [0.125, 0.125, 0.125, 0.125]
            A4 = [0.175, 0.25, 0.05, 0.25, 0.175]
            A5 = [0.175, 0.25, 0.175, 0.175, 0.25, 0.175]
            A6 = [0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225]
            A7 = [0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225]
            A8 = [0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275]
            A9 = [0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275]
            AX = [0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325]
            constant_matrix = [A1, A2, A3, A4, A5, A6, A7, A8, A9, AX]
            A = constant_matrix[self.column_count - 1]

            delta_t = []
            for i in range(len(note_starts_fixed_pair) - 1):
                delta_t.append(note_starts_fixed_pair[i + 1] - note_starts_fixed_pair[i])
            x = (64.5 - math.ceil(self.od * 3))*2
            
            # intensity_func = lambda t: 1/((t/1000) * ((t/1000) + 0.3 * math.sqrt(x)))
            intensity_func = lambda t: 0.1*A[k]*(max(x, t)/1000)**(-2)
            intensities = list(map(intensity_func, delta_t))
            time_based=[]
            for m in range(0, self.note_starts[len(self.note_starts)-1]):
                i=time_to_note_fixed_pair[m] #m is in sec, but the time_to_note array is per ms.
                if m in range(note_starts_fixed_pair[0], note_starts_fixed_pair[len(note_starts_fixed_pair)-1]):
                    time_based.append(intensities[i])
                else:
                    time_based.append(0)


            nv=[]
            for i in range(len(note_starts_fixed_pair) - 1):
                ln_parts = []
                for j in range(len(note_starts_fixed_pair) - 1):
                    if note_ends_fixed_pair[j]>note_starts_fixed_pair[i] and note_starts_fixed_pair[j]<note_starts_fixed_pair[i+1]:
                        if min(note_starts_fixed_pair[i+1], note_ends_fixed_pair[j]) - max(note_starts_fixed_pair[j], note_starts_fixed_pair[i])<100:
                            length = 0.5*(min(note_starts_fixed_pair[i+1], note_ends_fixed_pair[j]) - max(note_starts_fixed_pair[j], note_starts_fixed_pair[i]))
                        else:
                            length = min(note_starts_fixed_pair[i+1], note_ends_fixed_pair[j]) - max(note_starts_fixed_pair[j], note_starts_fixed_pair[i]) - 50
                        ln_parts.append(length) #length contained for each j from i to i+1
                nv.append(1+0.0065*sum(ln_parts))
            
            in_pair_starts = []
            for m in range(self.note_starts[len(self.note_starts)-1]):
                cond_ = False
                l = 0
                r = len(note_starts_fixed_pair) - 1
                while l <= r: 
                    mid = int(l + (r - l)/2)           
                    if note_starts_fixed_pair[mid] == m: 
                        cond_ = True
                        break
                    elif note_starts_fixed_pair[mid] < m: 
                        l = mid + 1
                    else: 
                        r = mid - 1
                in_pair_starts.append(cond_)

            dist=[]
            nvt=[]
            sv=1
            for m in range(0, self.note_starts[len(self.note_starts)-1]):
                if m in range(note_starts_fixed_pair[0], note_starts_fixed_pair[len(note_starts_fixed_pair)-1]):
                    nvt.append(nv[time_to_note_fixed_pair[m]])
                else:
                    nvt.append(0)
            for m in range (0, self.note_starts[len(self.note_starts)-1]):
                temp=nvt[m] * (self.asperity[m]) * sv * time_based[m]
                if not in_pair_starts[m]:
                    dist.append(temp)
                else:
                    dist.append((self.asperity[m]) * sv * time_based[m])
                # if m % 100 == 0 and k==6:
                #     print (temp, nvt[m], (asperity[m]), time_based[m], m)

            O_single = []
            for i in range(0, self.note_starts[len(self.note_starts)-1]+500, 49):
                if math.floor(100*(i+49)/self.note_starts[len(self.note_starts) - 1]) > math.floor((100*i)/self.note_starts[len(self.note_starts) - 1]):
                    print("Pair " + str(k+1) + "/" + str(self.column_count+1) + " " + str(math.floor(100*i/self.note_starts[len(self.note_starts) - 1])+1) + "%     ", end='\r')
                convolution=[]
                for j in range (max(0,i-500), min(i+500, self.note_starts[len(self.note_starts)-1])):
                    if j in range(0, self.note_starts[len(self.note_starts)-1]):
                        convolution.append(dist[j])
                    else:
                        convolution.append(0)
                indicator = 0.001 * sum(convolution)
                # if (i % 100 == 0):
                    # print (indicator)
                if note_starts_fixed_pair[0] <= i < note_starts_fixed_pair[len(note_starts_fixed_pair)-1]:
                    j=time_to_note_fixed_pair[i]
                    if note_starts_fixed_pair[j-1] == i:
                        weight = 1 + 1/(note_starts_fixed_pair[j+1]-note_starts_fixed_pair[j])
                    else:
                        weight = 1/(note_starts_fixed_pair[j+1]-note_starts_fixed_pair[j])
                else:
                    weight = 1/(note_starts_fixed_pair[0] + self.note_starts[len(self.note_starts)-1]+500 - note_starts_fixed_pair[len(note_starts_fixed_pair)-1])
                O_single.append(49*(indicator)**4 * weight) #this is O to the 4th power times the no. of notes
            O_collection.append(sum(O_single))
        # print (O_collection)  
        return O_collection

    def Z(self, O):
        # print ("Z Dimension:", end=' ')
        # print ((sum(O)/(2*len(self.note_starts)))**(1/4)) #paper中需要把X-dimension最终步骤相应修改
        return ((sum(O)/(2*len(self.note_starts)))**(1/4))
