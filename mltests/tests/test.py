
import pandas as pd
import re



def pre_file(filename):
    fr = open(filename)
    datasets = fr.readlines()
    results = []
    for line in datasets:
        res_line = []
        pre_line = line.split()
        res_line.append(line[0])
        i = 1
        for ss in pre_line[1 :]:
            mark_index = ss.index(':')
            res_str = ss[mark_index + 1 :]
            res_line.append(str(i) + ':' + res_str)
            i += 1
        results.append(res_line)
    return results




def pre_file2(filename):
    fr = open(filename)
    datasets = fr.readlines()
    results = []
    for line in datasets:
        pre_line = line.split()
        res_line = []
        res_line.append(line[0])
        sort_res = sorted(pre_line[1 :], key = lambda x : int(re.match('(\d)+', x).group()))
        for ch in sort_res:
            res_line.append(ch)
        results.append(res_line)
    return results



if __name__ == '__main__':
    datasets = pre_file('part.txt')
    df = pd.DataFrame(datasets, dtype = int)
    # results = df.fillna(0)
    df.to_csv('results.csv', index = False, header = False)