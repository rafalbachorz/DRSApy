import pandas as pd
import numpy as np
import operator
from itertools import combinations
from collections import Counter

class Rule:
    
    def __init__(self, decision, rules):
        self.decision = decision
        self.values = []
        self.columns = []
        self.operators = []
        self.rules = rules
        for rule in self.rules:
            self.columns.append(rule[0])
            self.operators.append(rule[1])
            self.values.append(rule[2])
        self.coverage = None
        self.covered = -1
    
    def apply(self, df):
        result = pd.Series(np.ones_like(df.iloc[:,0]).astype(bool), index=df.index)
        for i in range(len(self.columns)):
            result &= self.operators[i](df[self.columns[i]], self.values[i])
        return result
    
    def add(self, rule):
        nRules = [x for x in self.rules if x[0] != rule[0] or rule[1](x[2], rule[2])]
        nRules.append(rule)
        self.__init__(self.decision, nRules)
        
    def __str__(self):
        s = "Class {}{} if ".format({-1: "<=", 1: ">="}[self.decision[0]], self.decision[1])
        l = []
        for rule in self.rules:
            l += ["{} {} {}".format(rule[0], {"ge": ">=", "le": "<="}[rule[1].__name__], rule[2])]
        return s + ' and '.join(l)
    
def LEM2(toCoverIdx, roughSet, direction, target, credibility=1.0):
    result = set()
    op = {-1: operator.le, 1: operator.ge}
    toCover = roughSet.df.loc[list(toCoverIdx)]
    left = toCover.copy()
    it = 0
    while len(left) > 0:
        it += 1
        options = set()
        for _, row in left.iterrows():
            for i in range(len(roughSet.df.columns)):
                options.add((roughSet.df.columns[i], op[direction * roughSet.direction[i]], row[roughSet.df.columns[i]]))
#         options = list(options)
#         options.sort(key=lambda x: -x[1](roughSet.df[x[0]], x[2]).sum())
        mainRule = Rule((direction, target), [])
        tmpLeft = left.copy()
        while (int(mainRule.apply(toCover).sum()) / int(mainRule.apply(roughSet.df).sum())) < credibility:
            cutoff = set()
            bestOption = None
            metricA = -1
            metricB = -1
            for option in options:
                if any([(x[1](option[2], x[2]) and x[0]==option[0]) for x in cutoff]):
                    continue
                tmpRule = Rule((direction, target), mainRule.rules)
                tmpRule.add(option)
                b = tmpRule.apply(roughSet.df).sum()
                a = tmpRule.apply(tmpLeft).sum()
                d = tmpRule.apply(toCover).sum()
                c = min(d/b, credibility)
#                 b = a
                if c <= metricA and b <= metricB:
                    continue
                if c < metricA :
                    cutoff.add(option)
                    continue
                if c > metricA or (c >= metricA and b > metricB):
                    bestOption = option
                    metricA = c
                    metricB = b

            mainRule.add(bestOption)
            tmpLeft = tmpLeft[mainRule.apply(tmpLeft)]
            options = set()
            for _, row in tmpLeft.iterrows():
                for i in range(len(roughSet.df.columns)):
                    options.add((roughSet.df.columns[i], op[direction * roughSet.direction[i]], row[roughSet.df.columns[i]]))
            print(bestOption)
            print(mainRule.apply(toCover).sum(), metricB)

        if len(mainRule.rules) < 13:
            for i in range(len(mainRule.rules)):
                for subset in combinations(mainRule.rules, i+1):
                    newRule = Rule((direction, target), subset)
                    if newRule.apply(toCover).sum() == newRule.apply(roughSet.df).sum():
                        mainRule = newRule
                        break
                else:
                    continue
                break
        print(mainRule, mainRule.apply(toCover).sum(), mainRule.apply(roughSet.df).sum())
        mainRule.covered = mainRule.apply(toCover).sum()
        result.add(mainRule) 
        left = left[~mainRule.apply(left)]
    print(len(result))
    result = list(result)
    result.sort(key=lambda x: -x.covered)
    covered = pd.Series(np.zeros_like(toCover.iloc[:,0]).astype(bool), index=toCover.index)
    reducedResults = []
    coveredNumber = 0
    for rule in result:
        applyResult = rule.apply(toCover)
        covered |= applyResult
        if coveredNumber < covered.sum():
            ruleidx = np.where(rule.apply(roughSet.df))[0]
            ruleTarget = roughSet.target[ruleidx]
            rule.coverage = {i: set(ruleidx[ruleTarget == i]) for i in range(roughSet.classes)}
            rule.covered = len(ruleidx)
            reducedResults.append(rule)
        coveredNumber = covered.sum()
    print(len(reducedResults))
    return reducedResults


def classify(rules, classes):
    if len(rules) == 0:
        return [0 for i in range(len(classes))]
    elif len(rules) == 1:
        rule = list(rules)[0]
        return [len(rule.coverage[i])**2/(rule.covered*len(classes[i])) for i in classes]
    else:
        result = []
        for i in range(len(classes)):
            positiveNominator = set()
            positiveDenominator = [set(), classes[i]]
            
            negativeNominator = set()
            negativeDenominator = [set(), set()]

            for r in rules:
                if r.decision[0]*r.decision[1] > i * r.decision[0]:
                    for j in range(r.decision[1]+1):
                        if r.decision[0]*r.decision[1] <= j * r.decision[0]: 
                            pass
                        else: 
                            negativeNominator |= r.coverage[j]
                            negativeDenominator[1] |= set([j])
                        negativeDenominator[0] |= r.coverage[j]
                else:
                    positiveNominator |= r.coverage[i]
                    for j in r.coverage:
                        positiveDenominator[0] |= r.coverage[j]
            tmp = set()
            
            for j in negativeDenominator[1]:
                tmp |= classes[j]
            negativeDenominator[1] = tmp
            if len(positiveNominator) == 0:
                positiveScore = 0
            else:
                positiveScore = len(positiveNominator)**2 / (len(positiveDenominator[0]) * len(positiveDenominator[1]))
            if len(negativeNominator) == 0:
                negativeScore = 0
            else:
                negativeScore = len(negativeNominator)**2 / (len(negativeDenominator[0]) * len(negativeDenominator[1]))
            result.append(positiveScore - negativeScore)
        return result
    
    
class DRSA:
    def __init__(self, direction, credibility=1.0):
        self.credibility = credibility
        self.direction = direction
        
    
    def fit(self, df, target):
        assert target.min() == 0
        assert target.max() > 0
        assert len(target) == len(df)
        assert len(self.direction) == len(df.columns)
        self.df = df
        self.target = target
        self.classes = self.target.max() + 1
        self.lowerApproximation = []
        self.upperApproximation = []
        self.__findApproximations()
        self.rules = []
        for i in range(len(self.lowerApproximation[0])):
            self.rules += LEM2(self.lowerApproximation[0][i], self, -1, i, credibility=self.credibility)
            self.rules += LEM2(self.lowerApproximation[1][i], self, 1, i+1, credibility=self.credibility)
            
        tmp = target.groupby(target).groups
        self.__classes = {i:set(tmp[i]) for i in tmp}
            
    def predict(self, df):
        res = []
        tmp = []
        for rule in self.rules:
            tmp.append(rule.apply(df))
        for i in range(df.shape[0]):
            res.append(classify([self.rules[j] for j in np.where(np.array(tmp).T[i])[0]], self.__classes))
        return np.array(res).argmax(1)
        
    def domination(self, x, y):
        return (y*self.direction >= x*self.direction).all(1)
    
    def __findApproximations(self):
        self.lowerApproximation.append([])
        self.upperApproximation.append([])
        for i in range(self.classes - 1):
            print("class <= {}".format(i))
            msk = self.target <= i
            lower = set()
            upper = set()
            for idx, row in self.df[msk].iterrows():
                dominant = self.domination(-row, -self.df[~msk])
                if dominant.sum() < 1:
                    lower.add(idx)
                else:
                    upper.add(idx)
                    upper |= set(dominant[dominant].index)
            self.lowerApproximation[0].append(lower)
            self.upperApproximation[0].append(upper)     
            
        self.lowerApproximation.append([])
        self.upperApproximation.append([])
        for i in range(1, self.classes):
            print("class >= {}".format(i))
            msk = self.target >= i
            lower = set()
            upper = set()
            for idx, row in self.df[msk].iterrows():
                upper.add(idx)
                
                dominant = self.domination(row, self.df[~msk])
                if dominant.sum() < 1:
                    lower.add(idx)
                else:
                    upper |= set(dominant[dominant].index)
            self.lowerApproximation[1].append(lower)
            self.upperApproximation[1].append(upper)  