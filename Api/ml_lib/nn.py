import json

import numpy as np
from numpy import array

def process_output(components, CNN_output_map):
    components, groups = assign_group(components)
    components = detect_script(components, groups)
    return construct_latex(components, groups)
    
def assign_group(components, offset_threshold=3):
    heights = [[components[i]['tl'][0], components[i]['br'][0]] for i in components]
    groups = [heights[0]]
    for height in heights[1:]:
        if height[0] + offset_threshold < groups[-1][1]:
            groups[-1][1] = max(height[1], groups[-1][1])
        else:
            groups.append(height)
    for i in components:
        for group in groups:
            if group[0] < components[i]['tl'][0] + offset_threshold < group[1]: 
                components[i]['group'] = group
    return components, groups

def detect_script(components, groups):
    for g in groups:
        bottoms = [components[i]['br'][0] for i in sorted(components.keys()) if components[i]['group'] == g]
        tops = [components[i]['tl'][0] for i in sorted(components.keys()) if components[i]['group'] == g]
        bottoms_mean = np.mean(bottoms)
        bottoms_std = np.std(bottoms)
        tops_mean = np.mean(tops)
        tops_std = np.std(tops)

        if len(bottoms) == 1: continue
        for i in components:
            if components[i]['group'] == g:
                s = (bottoms_mean - components[i]['br'][0])/bottoms_std - (components[i]['tl'][0] - tops_mean)/tops_std

                if s > 2.5:
                    components[i]['sup'] = True
                elif s < -2.5:
                    components[i]['sub'] = True
    return components

def construct_latex(components, groups):
    lr_order = sorted(components.keys(), key=lambda x: components[x]['tl'][1])
    vsep = {tuple(group):[] for group in groups}
    MODE_SUP = set()
    MODE_SUB = set()
    MODE_SQRT = {}

    for l in lr_order:
        t, left = components[l]['tl']
        b, right = components[l]['br']
        for g in vsep:
            if g[0] <= t <= b <= g[1]:

                if g in MODE_SQRT and left > MODE_SQRT[g]:
                    vsep[g].append('}')
                    del MODE_SQRT[g]

                if g in MODE_SUP and not components[l]['sup']:
                    vsep[g].append('}')
                    MODE_SUP.remove(g)
                if g in MODE_SUB and not components[l]['sub']:
                    vsep[g].append('}')
                    MODE_SUB.remove(g)

                if g not in MODE_SUP and components[l]['sup']:
                    vsep[g].append('^{')
                    MODE_SUP.add(g)
                if g not in MODE_SUB and components[l]['sub']:
                    vsep[g].append('_{')
                    MODE_SUB.add(g)

                vsep[g].append(components[l]['output'] + ' ')
                if components[l]['output'] == '\\\\sqrt':
                    MODE_SQRT[g] = right
                    vsep[g].append('{')

                break

    for i in MODE_SQRT:
        vsep[g].append('}')
    for g in vsep:
        vsep[g] = ''.join(vsep[g])

    # FRACTION PROCESSING (for now just 3 layers)
    if len(vsep) == 3:
        first_g, _, last_g = list(sorted([g for g in vsep], key = lambda g: g[0]))
        final = '\\\\frac{' + vsep[first_g] + '}{' + vsep[last_g] + '}'
    else:
        final = list(vsep.values())[0]
    final = '$' + final + '$'
    return final

def predict(components, persistent_sess, x, y, CNN_output_map):
    results = []
    for i in range(len(components)):
        test = np.asarray(components[i+1]['pic']).astype(np.float32)
        test = array(test).reshape(1,45,45,1)
        y_out = persistent_sess.run(y, feed_dict={
            x: test
        })
        components[i+1]['label'] = y_out[0]
        components[i+1]['output'] = CNN_output_map[y_out[0]]

        results += y_out.tolist()
    
    print(results)
    expr = process_output(components, CNN_output_map)
    return json.dumps({'results': results, 'expr': expr})