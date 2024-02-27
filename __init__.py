import numpy as np
import math
import pandas as pd
from numba import jit, njit, vectorize
from itertools import combinations 
def score_calc(t1,t2,t3,p0 = 0,n0 = 0,p1 = 0,n1 = 0, a2 =0):  #t1,t2,t3 attribute_array,marker_array,len_S
    for i in range(0,len(t1)) :
        if (t1[i] == 0 ) & (t2[i] == 0) :
            p0 = p0 + 1
        if (t1[i] == 0 ) & (t2[i] == 1) :
            n0 = n0 + 1
        if (t1[i] == 1 ) & (t2[i] == 0)  :
            p1 = p1 + 1
        if (t1[i] == 1 ) & (t2[i] == 1)  :
            n1 = n1 + 1
    try:
        
        d01 = (((p0/(p0 + n0))*(math.log(p0/(p0 + n0))/math.log(2)))) + ((n0/(p0 + n0))*(math.log(n0/(p0 + n0))/math.log(2)))
    
    except:
        d01= 0
    e0 = d01     
    
    try: 
        d03 = ((p1/(p1 + n1)*(math.log(p1/(p1 + n1))/math.log(2))) + ((n1/(p1 + n1))*(math.log(n1/(p1 + n1))/math.log(2))))
    
    except :
        d03=0 
    e1 = d03 
    return int(((p0 +n0)*(e0 + e1))/t3)  
def pattern_check(arr_df3 , arr_df4, arr_p):
    c1= 0
    c2 = 0
    for i in list(arr_df3):
        if (i == arr_p).all():
            c1 = c1 +1
    for i in list(arr_df4):
        if (i == arr_p).all():
            c2 = c2 +1

#  if (c1 == 0 ) & (0 < c2) :
    if c2 > c1*9:
        return True
    else :
        return False
def neg_pattern_check(arr_df3 , arr_df4, arr_p):
    c1= 0
    c2 = 0
    for i in list(arr_df3):
        if (i == arr_p).all():
            c1 = c1 +1
    for i in list(arr_df4):
        if (i == arr_p).all():
            c2 = c2 +1
    if (c2 == 0 ) & (0 < c1) :
# if c1 > c2*4:
        return True
    else :
        return False
class model(object):
    def __init__(self):
        # self.S = [data]
        # self.s = [data][0]
        # self.V = list(data.columns)
        # self.V.remove('Marker')
        self.best_feature_0 = []
        self.Q = []
        self.feature_score_dict = {}
        self.degree_3_patterns={}
        self.neg_degree_3_patterns={}
        self.degree_2_patterns={}
        self.neg_degree_2_patterns={}
        self.filter_1_1={}
        self.filter_1_0={}
    
    
    def df_delete(self,l01):
        l02 = []
        for l1 in l01:
            if l1.shape[0] > 0 :
                l02.append(l1)

        return l02
     
    
    def find_feature_scores(self,data):
        score_calc_jitted_func =  njit()(score_calc)
        S = [data]
        s = [data][0]
        V = list(data.columns)
        V.remove('Marker')
        while len(S) > 0 :
            len_S = len(S)
            best_score = np.inf
            for ba in V :   
                score = 0
                for s in S :
                    if ba in s.columns :
                        attribute_array = np.array(s[ba])
                        marker_array = np.array(s['Marker'])
                        score = score - score_calc_jitted_func(attribute_array,marker_array,len_S)

                if (score < best_score) & (score > 0)  :
                    best_feature = ba
                    best_score = score
            
            self.feature_score_dict[best_feature] = best_score
            
            S2 = []
            for s in S :
                if best_feature in s.columns:
                    s1 = s[s[best_feature] == 1]
                    s0 = s[s[best_feature] == 0]
                    s = s[0:0]
                    S = self.df_delete(S)
                    if len(list(np.unique(s0['Marker']))) > 1 :
                        #s0.drop([best_feature], axis=1)
                        s0=s0.T.drop_duplicates().T

                        S2.append(s0)

                    if len(list(np.unique(s1['Marker']))) > 1 :
                        s1.drop([best_feature], axis=1)
                        #s1=s1.T.drop_duplicates().T

                        S2.append(s1)
                
            S = S2
            S = self.df_delete(S)
            V.remove(best_feature)
            if self.feature_score_dict[best_feature] == 0 :
                self.best_feature_0.append(best_feature)
            else:
                self.Q.append(best_feature)
        for key in self.feature_score_dict.keys():
            self.feature_score_dict[key] = self.feature_score_dict[key]/1000000
        self.feature_score_dict = pd.DataFrame(self.feature_score_dict, index = ['feature','score']).T
        self.feature_score_dict.to_csv(r'./a.csv')
        self.feature_score_dict = pd.read_csv(r'./a.csv')
    
    
    
    def finddegonepatterns(self,binarized_df,feature_score_dict):
        Q = list(feature_score_dict['Unnamed: 0'])
        Q.append('Marker')
        Q_df = binarized_df[Q]  
        drop_list = []
        for column in Q_df.columns.to_list():
            try:
                if sum(Q_df[column]) == len(Q_df) :
                    drop_list.append(column)
                elif sum(Q_df[column]) == 0 :
                    drop_list.append(column)
            except :
                pass
        Q_df1 = Q_df.copy()
        Q_df_natural = Q_df1[Q_df1['Marker'] == 0]
        Q_df2 = Q_df.copy()
        Q_df_attack = Q_df2[Q_df2['Marker'] == 1]
        filter_1_0 = []
        filter_1_1 = []
        f_1 = dict(Q_df_natural.sum() == len(Q_df_natural))
        for ba in f_1.keys():
            if f_1[ba] == True :
                if sum(Q_df_attack[ba]) < len(Q_df_attack):
                    filter_1_0.append(ba)

        f_2 = dict(Q_df_natural.sum() == 0)
        for ba in f_2.keys():
            if f_2[ba] == True :
                if sum(Q_df_attack[ba]) > 0 :
                    filter_1_1.append(ba)
                    
        filter_1_1.remove('Marker') 
        self.filter_1_0= filter_1_0
        self.filter_1_1 =filter_1_1
        if (len(filter_1_0) > 0) or (len(filter_1_1) > 0):
            binarized_df = Q_df_attack 
            l1=[]
            for pattern in range(0,len(filter_1_0)):
                p = pattern
                a1= binarized_df[binarized_df[filter_1_0[p]] == 0]
                l1.append(a1)
            for pattern in range(0,len(filter_1_1)):
                p = pattern
                a1= binarized_df[binarized_df[filter_1_1[p]] == 1]
                l1.append(a1)
            d_2 = pd.concat(l1)
            Q_df_attack.drop(list(d_2.index),inplace = True)
    
    def finddeg2patterns(self,binarized_df,feature_score_dict):
        pattern_check_jitted_func =  njit()(pattern_check)
        neg_pattern_check_jitted_func = njit()(neg_pattern_check)
        Q = list(feature_score_dict['Unnamed: 0'])
        Q.append('Marker')
        Q_df = binarized_df[Q]  
        Q_df1 = Q_df.copy()
        Q_df_natural = Q_df1[Q_df1['Marker'] == 0]
        Q_df2 = Q_df.copy()
        Q_df_attack = Q_df2[Q_df2['Marker'] == 1]
        attribute_1 = []
        attribute_2 = []
        value_1 = []
        value_2 = []
        attribute_3 = []
        attribute_4 = []
        value_4 = []
        value_3 = []

        comb = combinations([0,1,0,1],2)
        comb_triplets = list(set(comb))

        patterns = comb_triplets

        f_2 = []
        df1 = Q_df_attack.copy()
        df1.drop(columns = ['Marker'],inplace =True)
        df2 = Q_df_natural.copy()
        df2.drop(columns = ['Marker'],inplace =True)
        print(len(df1) +len(df2))
        comb = combinations(list(df1),2)
        comb_3 =list(set(comb))

        for pair in comb_3 :

            df4 = df1[list(pair)]
            df3 = df2[list(pair)]
            for p in patterns :                
                        
                        
                if pattern_check_jitted_func(arr_df3 = np.array(df3) , arr_df4 = np.array(df4), arr_p= np.array(p)):
                    a1= Q_df_attack[(Q_df_attack[pair[0]] == p[0])&(Q_df_attack[pair[1]] == p[1])]  
                    p1 = len(list(a1.index))
                    Q_df_attack.drop(list(a1.index),inplace = True)
                    
                    if p1 > 0 :
                        attribute_1.append(pair[0])
                        attribute_2.append(pair[1])
                        value_1.append(p[0])
                        value_2.append(p[1])
                        f_2.append(p1)
                        print(p1,sum(f_2), 'pos')
                        
        
                        
                if neg_pattern_check_jitted_func(arr_df3 = np.array(df3) , arr_df4 = np.array(df4), arr_p= np.array(p)):  
                    a2= Q_df_natural[(Q_df_natural[pair[0]] == p[0])&(Q_df_natural[pair[1]] == p[1])]  
                    p2 = len(list(a2.index))
                    Q_df_natural.drop(list(a2.index),inplace = True)

                    if p2 > 0 :
                        attribute_3.append(pair[0])
                        attribute_4.append(pair[1])
                        value_3.append(p[0])
                        value_4.append(p[1])
                        f_2.append(p2)
                        print(p2,sum(f_2),'neg') 
            if len(Q_df_attack) == 0:
                break
                        
            if len(Q_df_natural) == 0:
                break
                    
                
        degree_2_patterns = pd.DataFrame()
        degree_2_patterns['attribute_1'] =attribute_1
        degree_2_patterns['value_1'] =value_1
        degree_2_patterns['attribute_2'] =attribute_2
        degree_2_patterns['value_2'] =value_2
        neg_degree_2_patterns = pd.DataFrame()
        neg_degree_2_patterns['attribute_1'] =attribute_3
        neg_degree_2_patterns['value_1'] =value_3
        neg_degree_2_patterns['attribute_2'] =attribute_4
        neg_degree_2_patterns['value_2'] =value_4
        self.degree_2_patterns =  degree_2_patterns
        self.neg_degree_2_patterns  =neg_degree_2_patterns
    def finddeg3patterns(self,binarized_df,feature_score_dict):
        neg_pattern_check_jitted_func = njit()(neg_pattern_check)
        pattern_check_jitted_func =  njit()(pattern_check)
        Q = list(feature_score_dict['Unnamed: 0'])
        Q.append('Marker')
        Q_df = binarized_df[Q]  
        Q_df1 = Q_df.copy()
        Q_df_natural = Q_df1[Q_df1['Marker'] == 0]
        Q_df2 = Q_df.copy()
        Q_df_attack = Q_df2[Q_df2['Marker'] == 1]
        attribute_1 = []
        attribute_2 = []
        value_1 = []
        value_2 = []
        attribute_3 = []
        attribute_4 = []
        attribute_5 = []
        attribute_6 = []
        value_3 = []
        value_6 = []
        value_4 = []
        value_5 = []

        comb = combinations([0,1,0,1,0,1,0],3)
        comb_triplets = list(set(comb))

        patterns = comb_triplets

        f_3 = []
        df1 = Q_df_attack.copy()
        df1.drop(columns = ['Marker'],inplace =True)
        df2 = Q_df_natural.copy()
        df2.drop(columns = ['Marker'],inplace =True)

        comb = combinations(list(df1),3)
        comb_3 = list(set(comb))
        for pair in comb_3 :
            df4 = df1[list(pair)]
            df3 = df2[list(pair)]
            for p in patterns :
                if pattern_check_jitted_func(arr_df3 = np.array(df3) , arr_df4 = np.array(df4), arr_p= np.array(p)) :
                    a1= Q_df_attack[(Q_df_attack[pair[0]] == p[0])&(Q_df_attack[pair[1]] == p[1])&(Q_df_attack[pair[2]] == p[2])]  
                    p1 = len(list(a1.index))
                    Q_df_attack.drop(list(a1.index),inplace = True)
                    
                    if p1 > 0 :
                        attribute_1.append(pair[0])
                        attribute_2.append(pair[1])
                        value_1.append(p[0])
                        value_2.append(p[1])
                        attribute_3.append(pair[2])
                        value_3.append(p[2])
                if neg_pattern_check_jitted_func(arr_df3 = np.array(df3) , arr_df4 = np.array(df4), arr_p= np.array(p)):  
                    a2= Q_df_natural[(Q_df_natural[pair[0]] == p[0])&(Q_df_natural[pair[1]] == p[1])&(Q_df_natural[pair[2]] == p[2])]
                    p2 = len(list(a2.index))
                    Q_df_natural.drop(list(a2.index),inplace = True)

                    if p2 > 0 :
                        attribute_4.append(pair[0])
                        attribute_5.append(pair[1])
                        attribute_6.append(pair[1])
                        value_4.append(p[0])
                        value_5.append(p[1])
                        value_6.append(p[1])
            if len(Q_df_attack) == 0:
                break
            if len(Q_df_natural)==0:
                break
        degree_3_patterns = pd.DataFrame()
        degree_3_patterns['attribute_1'] =attribute_1 
        degree_3_patterns['value_1'] =value_1
        degree_3_patterns['attribute_2'] =attribute_2
        degree_3_patterns['value_2'] =value_2
        degree_3_patterns['attribute_3'] =attribute_3
        degree_3_patterns['value_3'] =value_3
        self.degree_3_patterns =  degree_3_patterns
        neg_degree_3_patterns = pd.DataFrame()
        neg_degree_3_patterns['attribute_1'] = attribute_4
        neg_degree_3_patterns['attribute_2'] = attribute_5
        neg_degree_3_patterns['attribute_3'] = attribute_6
        neg_degree_3_patterns['value_1'] =value_4
        neg_degree_3_patterns['value_2'] =value_5
        neg_degree_3_patterns['value_3'] =value_6
        self.neg_degree_3_patterns  = neg_degree_3_patterns


    def predict(self,binarized_df):
        binarized_df['index'] = binarized_df.index
        Q = list(self.feature_score_dict['Unnamed: 0'])
        Q.append('Marker')
        Q.append('index')
        accuracy_dict = {}
        l1 = []
        l2 = []
        l3 = []
        #degree_2_patterns = pd.read_csv(r'E:/IIT-R/WADI.A2_19 Nov 2019/Result_1/degree_2_patterns.csv')
        #degree_3_patterns = pd.read_csv(r'E:/IIT-R/WADI.A2_19 Nov 2019/Result_1/degree_3_patterns.csv')

        for pattern in range(0,len(self.degree_2_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.degree_2_patterns['attribute_1'][p]] == self.degree_2_patterns['value_1'][p])
                    &(binarized_df[self.degree_2_patterns['attribute_2'][p]] == self.degree_2_patterns['value_2'][p])]
            
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
            
        for pattern in range(0,len(self.neg_degree_2_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.neg_degree_2_patterns['attribute_1'][p]] == self.neg_degree_2_patterns['value_1'][p])
                    &(binarized_df[self.neg_degree_2_patterns['attribute_2'][p]] == self.neg_degree_2_patterns['value_2'][p])]
            
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l2.append(a1)

        for pattern in range(0,len(self.degree_3_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.degree_3_patterns['attribute_1'][p]] == self.degree_3_patterns['value_1'][p])
                    &(binarized_df[self.degree_3_patterns['attribute_2'][p]] == self.degree_3_patterns['value_2'][p])
                            &(binarized_df[self.degree_3_patterns['attribute_3'][p]] == self.degree_3_patterns['value_3'][p])]
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_3' ] = [correct,total]
            
            l3.append(a1)
        for pattern in range(0,len(self.neg_degree_3_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.neg_degree_3_patterns['attribute_1'][p]] == self.neg_degree_3_patterns['value_1'][p])
                    &(binarized_df[self.neg_degree_3_patterns['attribute_2'][p]] == self.neg_degree_3_patterns['value_2'][p])
                            &(binarized_df[self.neg_degree_3_patterns['attribute_3'][p]] == self.neg_degree_3_patterns['value_3'][p])]
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_3' ] = [correct,total]
            
            l3.append(a1)
        for pattern in range(0,len(self.filter_1_0)):
            p = pattern
            a1= binarized_df[binarized_df[self.filter_1_0[p]] == 0]
            
            total = len(a1)
            correct = sum(a1['Marker'])
        #  accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
        for pattern in range(0,len(self.filter_1_1)):
            p = pattern
            a1= binarized_df[binarized_df[self.filter_1_1[p]] == 1]
            
            total = len(a1)
            correct = sum(a1['Marker'])
        #  accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
        predicted_df_2 = pd.concat(l1) 
        predicted_df_3 = pd.concat(l3)
        predicted_df_not = pd.concat(l2)
        predicted_df_3 = predicted_df_3.drop_duplicates()

        predicted_df_not = predicted_df_not.drop_duplicates()
        drop_list = []
        for row in list(predicted_df_3.index):
            if row in list(predicted_df_not.index):
                drop_list.append(row)
        predicted_df_3.drop(drop_list, inplace=True)
        predicted_df = pd.concat([predicted_df_3,predicted_df_2])
        predicted_df = predicted_df.drop_duplicates()
        return predicted_df
    def train(self,binarized_df):
        self.find_feature_scores(binarized_df)
        self.finddegonepatterns(binarized_df,self.feature_score_dict)
        self.finddeg2patterns(binarized_df,self.feature_score_dict)
        self.finddeg3patterns(binarized_df,self.feature_score_dict)
        binarized_df['index'] = binarized_df.index
        Q = list(self.feature_score_dict['Unnamed: 0'])
        Q.append('Marker')
        Q.append('index')
        binarized_df = binarized_df[Q]
        accuracy_dict = {}
        l1 = []
        l2 = []
        l3 = []
        #degree_2_patterns = pd.read_csv(r'E:/IIT-R/WADI.A2_19 Nov 2019/Result_1/degree_2_patterns.csv')
        #degree_3_patterns = pd.read_csv(r'E:/IIT-R/WADI.A2_19 Nov 2019/Result_1/degree_3_patterns.csv')

        for pattern in range(0,len(self.degree_2_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.degree_2_patterns['attribute_1'][p]] == self.degree_2_patterns['value_1'][p])
                    &(binarized_df[self.degree_2_patterns['attribute_2'][p]] == self.degree_2_patterns['value_2'][p])]
            
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
            
        for pattern in range(0,len(self.neg_degree_2_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.neg_degree_2_patterns['attribute_1'][p]] == self.neg_degree_2_patterns['value_1'][p])
                    &(binarized_df[self.neg_degree_2_patterns['attribute_2'][p]] == self.neg_degree_2_patterns['value_2'][p])]
            
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l2.append(a1)

        for pattern in range(0,len(self.degree_3_patterns)):
            p = pattern
            a1= binarized_df[(binarized_df[self.degree_3_patterns['attribute_1'][p]] == self.degree_3_patterns['value_1'][p])
                    &(binarized_df[self.degree_3_patterns['attribute_2'][p]] == self.degree_3_patterns['value_2'][p])
                            &(binarized_df[self.degree_3_patterns['attribute_3'][p]] == self.degree_3_patterns['value_3'][p])]
            total = len(a1)
            correct = sum(a1['Marker'])
            accuracy_dict[str(p) + '_deg_3' ] = [correct,total]
            
            l3.append(a1)
            
        for pattern in range(0,len(self.filter_1_0)):
            p = pattern
            a1= binarized_df[binarized_df[self.filter_1_0[p]] == 0]
            
            total = len(a1)
            correct = sum(a1['Marker'])
        #  accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
        for pattern in range(0,len(self.filter_1_1)):
            p = pattern
            a1= binarized_df[binarized_df[self.filter_1_1[p]] == 1]
            
            total = len(a1)
            correct = sum(a1['Marker'])
        #  accuracy_dict[str(p) + '_deg_2'] = [correct,total]
            l1.append(a1)
        predicted_df_2 = pd.concat(l1) 
        predicted_df_3 = pd.concat(l3)
        predicted_df_not = pd.concat(l2)
        predicted_df_3 = predicted_df_3.drop_duplicates()

        predicted_df_not = predicted_df_not.drop_duplicates()
        drop_list = []
        for row in list(predicted_df_3.index):
            if row in list(predicted_df_not.index):
                drop_list.append(row)
        predicted_df_3.drop(drop_list, inplace=True)
        predicted_df = pd.concat([predicted_df_3,predicted_df_2])
        predicted_df = predicted_df.drop_duplicates()
        tp = sum(predicted_df['Marker'])
        fp = len(predicted_df['Marker']) - sum(predicted_df['Marker'])
        fn = sum(binarized_df['Marker']) - tp
        tn = len(binarized_df['Marker']) - len(predicted_df) - (fn)
        accuracy = ((tp + tn)/(tp+ tn + fp + fn))*100
        precision = (tp/(tp+ fp))*100
        recall = (tp/(tp+fn))*100 
        f1 = (2*precision*recall)/(precision + recall)
        print('accuracy is {}'.format(accuracy))
        print('precision is {}'.format(precision))
        print('recall is {}'.format(recall))
        print('f1 is {}'.format(f1))
binarized_df= pd.read_csv(r"C:/Users/opvv1/OneDrive/Desktop/binlad/binpat/'binarized_trainAD.csv")
model1= model()
model1.train(binarized_df)