import json
import random
import matplotlib
import matplotlib.pyplot as plt

def add_weight(fn, wfn, ofn, prob, method):
    weights = list()
    with open(f"./logs/{wfn}.txt", 'r', encoding='utf-8') as f:
        logs = f.read().strip().split('\n\n')
        for log in logs:
            weight = list(map(float, log.split('\n')[-1][11:].split(' ')))
            if method == 'random':
                for i in range(len(weight)):
                    if random.random() < prob:
                        weight[i] = 0.0
            else:
                bund = [(i, w) for i, w in enumerate(weight)]
                bund.sort(key=lambda x: x[1], reverse=(method=='largest'))
                for i in range(int(len(weight) * prob)):
                    bund[i] = (bund[i][0], 0.0)
                bund.sort(key=lambda x: x[0])
                weight = [b[1] for b in bund]
            weights.append(weight)
    fdata = json.load(open(f"./datasets/parsed/{fn}.json", 'r', encoding='utf-8'))
    num = 0
    for data in fdata:
        for aspect in data['aspects']:
            aspect['weight'] = weights[num]
            num += 1
    open(f"./datasets/inference/{ofn}.json", 'w', encoding='utf-8').write(json.dumps(fdata, sort_keys=False, indent=4))

def merge_obj(fn):
    texts = list()
    lengths = list()
    weights = list()
    with open(f"./logs/{fn}.txt", 'r', encoding='utf-8') as f:
        logs = f.read().strip().split('\n\n')
        for log in logs:
            text = log.split('\n')[1][6:]
            length = len(text.split(' '))
            weight = list(map(float, log.split('\n')[-1][11:].split(' ')))
            if len(texts) and text == texts[-1]:
                for i in range(len(weights[-1])):
                    weights[-1][i] = max(weight[i], weights[-1][i])
            else:
                texts.append(text)
                lengths.append(length)
                weights.append(weight)
    return texts, lengths, weights

def count_word(lengths, weights, tao):
    counts = list()
    for i, weight in enumerate(weights):
        num = sum([1 if wi <= tao else 0 for wi in weight])
        counts.append(num / lengths[i])
    avg = sum(counts) / len(counts)
    return avg

def compute(texts, weights, tao):
    opinions = open('Restaurant_Test_Opinions.txt', 'r', encoding='utf-8').read().split('\n')
    opinions = [opinion.split() for opinion in opinions]
    predicts = 0
    corrects = 0
    trues = 0
    for i, (text, weight) in enumerate(zip(texts, weights)):
        trues += len(opinions[i])
        for ti, wi in zip(text.split(), weight):
            if wi > tao:
                predicts += 1
                if ti in opinions[i]:
                    corrects += 1
    P = corrects / predicts
    R = corrects / trues
    F1 = (2 * P * R) / (P + R)
    return P, R, F1

def topn(fn, texts, weights, n):
    opinions = open(fn, 'r', encoding='utf-8').read().split('\n')
    opinions = [opinion.split() for opinion in opinions]
    corrects = 0
    trues = 0
    for i, (text, weight) in enumerate(zip(texts, weights)):
        text = text.split()
        trues += len(opinions[i])
        zips = zip(text, weight)
        zips = sorted(zips, key=lambda x: x[1], reverse=True)
        text, weight = zip(*zips)
        for j in range(n):
            if text[j] in opinions[i]:
                corrects += 1
    return corrects / trues

def scores(fn, opinionf):
    texts, lengths, weights = merge_obj(fn)
    top1 = topn(opinionf, texts, weights, 1)
    top2 = topn(opinionf, texts, weights, 2)
    top3 = topn(opinionf, texts, weights, 3)
    count = count_word(lengths, weights, 0.01)
    return count, top1, top2, top3

if __name__ == '__main__':
    
    rest = scores('1567485156_log', 'Restaurants_Test_Opinions.txt')
    laptop = scores('1567511333_log', 'Laptops_Test_Opinions.txt')
    '''
    for m in ['random', 'smallest', 'largest']:
        for i in range(11):
            add_weight('Restaurants_Test', '1567485156_log', f"Restaurants_Eval_{m}_{i}", i/10, m)
            add_weight('Laptops_Test', '1567511333_log', f"Laptops_Eval_{m}_{i}", i/10, m)
    '''
    myfont2 = matplotlib.font_manager.FontProperties(fname='C:\\times.ttf', size=16)
    myfont = matplotlib.font_manager.FontProperties(fname='C:\\times.ttf', size=14)
    
    
    ####
    fig = plt.figure(figsize=(6, 3.7))
    ax1 = fig.add_subplot(111)
    
    ds_name = ['Rest14', 'Laptop']
    count = [rest[0]*100, laptop[0]*100]
    top1 = [rest[1]*100, laptop[1]*100]
    top2 = [rest[2]*100, laptop[2]*100]
    top3 = [rest[3]*100, laptop[3]*100]
    
    x = range(len(ds_name))
    
    ax1.bar(x, top1, width=0.2, label='RCL@1')
    ax1.bar([i + 0.2 for i in x], top2, width=0.2, label='RCL@2')
    ax1.bar([i + 0.4 for i in x], top3, width=0.2, label='RCL@3')
    ax1.legend(bbox_to_anchor=(0.5, -0.35), loc='lower center', ncol=3, prop=myfont2, borderpad=0.12, columnspacing=1, handletextpad=0.3)
    ax1.set_xlim([-0.3, 1.7])
    ax1.set_xticks([i + 0.2 for i in x])
    ax1.set_xticklabels(ds_name, fontproperties=myfont)
    ax1.set_xlabel('Dataset', fontproperties=myfont2)
    ax1.set_ylabel('Recall (%)', fontproperties=myfont2)
    #plt.xticks(fontproperties=myfont)
    plt.yticks(fontproperties=myfont)
    plt.savefig('score_curve_1.pdf', format='pdf', dpi=900, bbox_inches='tight')
    
    x = list(range(0, 110, 10))
    
    fig = plt.figure(figsize=(6, 3.7))
    ax2 = fig.add_subplot(111)
    ax2.plot(x, [83.7, 81.79, 80.89, 79.55, 78.12, 76.07, 74.82, 72.41, 69.91, 67.23, 65], label='Random', marker='o')
    ax2.plot(x, [83.7, 83.7, 83.7, 83.7, 83.66, 83.7, 83.66, 83.7, 83.48, 82.32, 65], label='Low', marker='s')
    ax2.plot(x, [83.7, 73.84, 66.87, 65.8, 65.45, 65.09, 65, 65, 65, 65, 65], label='High', marker='^')
    ax2.legend(bbox_to_anchor=(0.5, -0.35), loc='lower center', ncol=3, prop=myfont2, borderpad=0.12, columnspacing=1.2, handletextpad=0.3)
    ax2.set_ylim([63, 86])
    ax2.set_xlabel('Drop rate (%)', fontproperties=myfont2)
    ax2.set_ylabel('Test accuracy (%)', fontproperties=myfont2)
    plt.xticks(fontproperties=myfont)
    plt.yticks(fontproperties=myfont)
    plt.savefig('score_curve_2.pdf', format='pdf', dpi=900, bbox_inches='tight')
    
    fig = plt.figure(figsize=(6, 3.7))
    ax3 = fig.add_subplot(111)
    ax3.plot(x, [77.9, 75.24, 73.82, 71.94, 65.67, 59.87, 53.45, 44.67, 36.99, 27.59, 20.06], label='Random', marker='o')
    ax3.plot(x, [77.9, 77.9, 77.9, 77.9, 77.9, 77.9, 77.74, 77.9, 76.96, 75.24, 20.06], label='Low', marker='s')
    ax3.plot(x, [77.9, 64.73, 39.18, 27.74, 21.94, 20.38, 20.06, 20.06, 20.06, 20.06, 20.06], label='High', marker='^')
    ax3.legend(bbox_to_anchor=(0.5, -0.35), loc='lower center', ncol=3, prop=myfont2, borderpad=0.12, columnspacing=1.2, handletextpad=0.3)
    ax3.set_ylim([15, 85])
    ax3.set_xlabel('Drop rate (%)', fontproperties=myfont2)
    ax3.set_ylabel('Test accuracy (%)', fontproperties=myfont2)
    plt.xticks(fontproperties=myfont)
    plt.yticks(fontproperties=myfont)
    plt.savefig('score_curve_3.pdf', format='pdf', dpi=900, bbox_inches='tight')
    '''
    x = np.linspace(0, 0.45, 30)
    avgs = [count_word(lengths, weights, xi) for xi in x]
    precs = [compute(texts, weights, xi)[0] for xi in x]
    recalls = [compute(texts, weights, xi)[1] for xi in x]
    f1s = [compute(texts, weights, xi)[2] for xi in x]
    plt.figure()
    plt.plot(x, avgs)
    plt.title('Average Opinion Words')
    plt.show()
    plt.figure()
    plt.plot(x, precs)
    plt.title('Precision')
    plt.show()
    plt.figure()
    plt.plot(x, recalls)
    plt.title('Recall')
    plt.show()
    plt.figure()
    plt.plot(x, f1s)
    plt.title('F1 score')
    plt.show()
    '''