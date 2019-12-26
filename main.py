import sys
import osu_file_parser as osu_parser

length_of_arguments = len(sys.argv) # argv:运行程序时在输入窗口打上的变量
file_path = ""

if length_of_arguments != 1:
            file_path = sys.argv[1]

def parse_osu_file(file_path):
    p = osu_parser.parser(file_path)
    p.process()
    return p.get_parsed_data()

p = parse_osu_file(file_path)
column_count = p[0]
columns = p[1]
note_starts = p[2]
note_ends = p[3]

total = 0
for i in range(len(note_starts)):
    if note_ends[i] != 0:
        total += note_ends[i] - note_starts[i]
print(total/column_count/(note_starts[len(note_starts)-1]-note_starts[0]))
