import pandas as pd

def make_output_map():
    import pandas as pd
    df = pd.read_csv('dict.csv', delimiter=',', header=None)
    
    key = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

    key[37] = '\\infty'
    key[38] = '\\int'
    key[31] = 'f'
    key[65] = '\\sqrt'
#     key[2] = '\\right)'
#     key[1] = '\\left('
    key[26] = 'd'
#     key[8] = '2'
    key[16] = '-'
#     key[75] = 'x'
    key[56] = '\\pi'
#     key[6] = '0'
    
    return key