import pandas as pd

def make_output_map():
    import pandas as pd
    df = pd.read_csv('dict.csv', delimiter=',', header=None)
    
    key = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    
    return key