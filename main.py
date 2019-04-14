import algorithm
import sys

length_of_arguments = len(sys.argv)
file_path = ""
if length_of_arguments != 1:
    file_path = sys.argv[1]
    calculator = algorithm.star_calculator()
    calculator.calculate_SR(file_path)
