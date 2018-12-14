import json

import numpy as np
from numpy import array

def process_output(results_classes, label_to_location, CNN_output_map):
    output_labels = dict(zip(range(1,len(results_classes) + 1), results_classes))
    lr_order = sorted([lab for lab in label_to_location], key = lambda x: label_to_location[x]['tl'][1])
    symbol_medians = []
    
    for component in lr_order:
        component_location = label_to_location[component]
        symbol_medians.append((component_location['tl'][0] + component_location['br'][0]) // 2) # in order from left to right

    avg_height = np.mean(symbol_medians)

    height_diff = []
    for symb, med in zip(lr_order, symbol_medians):
        curr_height_diff = abs(med - avg_height)
        height_diff.append(curr_height_diff)
        
    avg_diff = np.mean(height_diff)
    std_diff = np.std(height_diff)
    
    height_diff_z = list(map((lambda x: (avg_diff - x)/std_diff), height_diff))
    symbol_to_height_diff = dict(zip(lr_order, height_diff_z))

    groups2 = [[label_to_location[l]['tl'][0], label_to_location[l]['br'][0]] for l in label_to_location]
    groups = [groups2[0]]
    for group in groups2[1:]:
        if group[0] < groups[-1][1]:
            groups[-1][1] = max(group[1], groups[-1][1])
        else:
            groups.append(group)

    vsep = [[tuple(group), []] for group in groups]
    for l in lr_order:
        t = label_to_location[l]['tl'][0]
        b = label_to_location[l]['br'][0]
        for i, g in enumerate(vsep):
            if g[0][0] <= t <= b <= g[0][1]:
                vsep[i][1].append(l)
                break
                
    
    chars = ['$']
    for group, ls in vsep:
        for l in ls:
            if symbol_to_height_diff[l] <= -2:
                chars.append(CNN_output_map[82])
    #             print(''.join(CNN_output_map[82]), end='')
            elif symbol_to_height_diff[l] >= 2:
                chars.append(CNN_output_map[83])
    #             print(''.join(CNN_output_map[83]), end='')
            chars.append(CNN_output_map[output_labels[l]])
    #         print(''.join(CNN_output_map[output_labels[l]]), end = '')
    
    chars.append('$')
    final_latex_string = ''.join(chars)
    
    return final_latex_string


def predict(data, persistent_sess, x, y, CNN_output_map):
    all_labels = data['labels']
    results = []
    for i in range(len(all_labels)):
        test = np.asarray(all_labels[i]).astype(np.float32)
        test = array(test).reshape(1,45,45,1)
        y_out = persistent_sess.run(y, feed_dict={
            x: test
        })
        results += y_out.tolist()
    expr = process_output(results, data['locations'], CNN_output_map)
    return json.dumps({'results': results, 'expr': expr})