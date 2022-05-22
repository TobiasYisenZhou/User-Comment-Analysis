# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:13:02 2022

@author: zsk98
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

# defining functions I need to proceed
def analyzer_fun(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer=SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

def line_keeper(df, txt, col):
    import pandas as pd
    ret=[]
    for i in df[col]:
        a=i.split()
        for j in txt:
            if j not in a:
                continue 
            else:
                ret.append(i)
                break
    return pd.DataFrame(data=ret,columns=[col])

def my_stop_words(var_in, new=[]):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    for i in new:
        sw.append(i)
    tmp = [word for word in str(var_in).split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def my_stem(var_in):
    from nltk.stem.porter import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var_in]
    return tmp

def gen_senti(text, pos, neg):
    tex_l=text.split()
    p=0
    n=0
    
    for i in tex_l:
        if i in pos:
            p+=1
        elif i in neg:
            n+=1
    if p+n ==0:
        return 0
    score=(p-n)/(p+n)
    
    return score

# creating positive and negative dictionaries using Hu&Bing dic
pos=open(r".data\positive-words.txt","r") # change path
pos_r=pos.read()
pos_l=pos_r.split()
pos_stemmed=my_stem(pos_l)
pos.close()

neg=open(r".data\negative-words.txt","r") # change path
neg_r=neg.read()
neg_l=neg_r.split()
neg_stemmed=my_stem(neg_l)
neg.close()

#read in data (use combined, solo amazon and solo twitter)
t_OS94=pd.read_csv(r".data\combined_94.csv") # change path
t_OS10=pd.read_csv(r".data\combined_10.csv")
t_OS105=pd.read_csv(r".data\combined_105.csv")


# removing new stopwords that are meaningless, like ultra
new_sw=["ultra"]
t_OS94["stemmed2"]=t_OS94["stemmed"].apply(lambda x: my_stop_words(x,new=new_sw))
t_OS10["stemmed2"]=t_OS10["stemmed"].apply(lambda x: my_stop_words(x,new=new_sw))
t_OS105["stemmed2"]=t_OS105["stemmed"].apply(lambda x: my_stop_words(x,new=new_sw))

# calculating sentiment score for each comments
t_OS94["sentiment"]=t_OS94["stemmed2"].apply(lambda x: gen_senti(x, pos_stemmed, neg_stemmed))
t_OS10["sentiment"]=t_OS10["stemmed2"].apply(lambda x: gen_senti(x, pos_stemmed, neg_stemmed))
t_OS105["sentiment"]=t_OS105["stemmed2"].apply(lambda x: gen_senti(x, pos_stemmed, neg_stemmed))

# calculate the mean for each version
OS94_mean=np.nanmean(t_OS94["sentiment"])
#OS94_median=np.nanmedian(t_OS94["sentiment"])
#all_std=np.std(df4["sentiment"])
OS10_mean=np.nanmean(t_OS10["sentiment"])
#OS10_median=np.nanmedian(t_OS10["sentiment"])

OS105_mean=np.nanmean(t_OS105["sentiment"])
#OS105_median=np.nanmedian(t_OS105["sentiment"])

#creating table that shows the change
Senti_94=t_OS94["sentiment"].tolist()
Senti_10=t_OS10["sentiment"].tolist()
Senti_105=t_OS105["sentiment"].tolist()

lable_94=["os94",]*len(Senti_94)
lable_10=["os10",]*len(Senti_10)
lable_105=["os105",]*len(Senti_105)
all_lab=lable_94+lable_10+lable_105

all_senti=Senti_94+Senti_10+Senti_105

df_dict2={"os":all_lab,"senti":all_senti}
senti_df=pd.DataFrame(df_dict2)


# Applying ANOVA to check whether the change in score for the three version is significant
data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
anova_table = pd.DataFrame(data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
anova_table.set_index('Source of Variation', inplace = True)


x_bar = senti_df['senti'].mean()
SSTR = senti_df.groupby('os').count() * (senti_df.groupby('os').mean() - x_bar)**2
anova_table['SS']['Between Groups'] = SSTR['senti'].sum()


SSE = (senti_df.groupby('os').count() - 1) * senti_df.groupby('os').std()**2
anova_table['SS']['Within Groups'] = SSE['senti'].sum()


SSTR = SSTR['senti'].sum() + SSE['senti'].sum()
anova_table['SS']['Total'] = SSTR


anova_table['df']['Between Groups'] = senti_df['os'].nunique() - 1
anova_table['df']['Within Groups'] = senti_df.shape[0] - senti_df['os'].nunique()
anova_table['df']['Total'] = senti_df.shape[0] - 1


anova_table['MS'] = anova_table['SS'] / anova_table['df']


F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
anova_table['F']['Between Groups'] = F


anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])


alpha = 0.05
tail_hypothesis_type = "two-tailed"
if tail_hypothesis_type == "two-tailed":
    alpha /= 2
anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

anova_table