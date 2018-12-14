import csv

def make_output_map():
    reader = csv.DictReader(open('symbolToLatex.csv'))
    CNN_output_map = {}
    
    for row in reader:
        for i in range(84):
            CNN_output_map[i] = row[str(i)]
    
    return CNN_output_map