#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:14:47 2022

@author: daniel
"""

import pandas as pd

import sys
# You need to replace this path!!
sys.path.append('/home/daniel/anaconda3/pkgs/scikit-surprise-1.1.1-py38hab2c0dc_1/lib/python3.8/site-packages/surprise/prediction_algorithms')
from predictions import Prediction

# https://towardsdatascience.com/top-3-python-package-to-learn-the-recommendation-system-bb11a916b8ff
# https://github.com/NicolasHug/Surprise
# http://surpriselib.com/
# https://github.com/tensorflow/recommenders
# https://github.com/statisticianinstilettos/recmetrics
# https://github.com/benfred/implicit
# https://lkpy.readthedocs.io/en/stable/
# https://muricoca.github.io/crab/
        
class Data:
    def __init__(self):
        
        self.available_databases=['ml_100k', 'ml_1m','jester']
        #self.the_data_reader= getattr(Data, 'read_'+database_name.lower())
        #self.the_data_reader= 'read_'+database_name.lower()
    def show_available_databases(self):
        print('The avaliable database are:')
        for i,database in enumerate(self.available_databases):
            print(str(i)+': '+database)            


        
    def read_data(self,database_name):
        self.database_name=database_name
        self.the_data_reader= getattr(self, 'read_'+database_name.lower())
        #self.the_data_reader()
        #eval(f"self.{self.the_data_reader}()")
        self.the_data_reader()
        # The datasets for collaborative filtering must be:
        # The dataframe containing the ratings. 
        # It must have three columns, corresponding to the user (raw) ids, 
        #the item (raw) ids, and the ratings, in this order.    

    def read_ml_100k(self):
        from surprise import Dataset
        data = Dataset.load_builtin('ml-100k')
        self.df = pd.DataFrame(data.__dict__['raw_ratings'], columns=['user_id','item_id','rating','timestamp'])
        self.df.drop(columns=['timestamp'],inplace=True)
        self.df.rename({'user_id':'userID','item_id':'itemID'},axis=1,inplace=True)

    def read_ml_1m(self):
        from surprise import Dataset
        data = Dataset.load_builtin('ml-1m')
        self.df = pd.DataFrame(data.__dict__['raw_ratings'], columns=['user_id','item_id','rating','timestamp'])
        self.df.drop(columns=['timestamp'],inplace=True)
        self.df.rename({'user_id':'userID','item_id':'itemID'},axis=1,inplace=True)

    def read_jester(self):
        from surprise import Dataset
        data = Dataset.load_builtin('jester')
        self.df = pd.DataFrame(data.__dict__['raw_ratings'], columns=['user_id','item_id','rating','timestamp'])
        self.df.drop(columns=['timestamp'],inplace=True)
        self.df.rename({'user_id':'userID','item_id':'itemID'},axis=1,inplace=True)

class Evaluator:

    def __init__(self,predictions_df):

        self.available_evaluators=['surprise.rmse','surprise.mse',
                                   'surprise.mae','surprise.fcp']
        self.predictions_df=predictions_df
        
    def show_evaluators(self):
        print('The avaliable evaluators are:')
        for i,evaluator in enumerate(self.available_evaluators):
            print(str(i)+': '+evaluator)
        


    def run(self,the_evaluator):        
        self.the_evaluator=the_evaluator
        if(self.the_evaluator[0:8]=='surprise'):
            self.run_surprise()
        else:
            print('This evaluator is not available!')

    def run_surprise(self):
        from surprise import accuracy
        predictions=[Prediction(row['user_id'],row['item_id'],row['rating'],row['predicted_rating'],{}) for index,row in self.predictions_df.iterrows()]
        self.predictions=predictions
     #   the_evaluator=locals()['accuracy.'+self.the_evaluator.replace("surprise.", "")]()
        from surprise import accuracy
        print(accuracy.rmse(predictions,verbose=False))

                   
#     def run_rouge(self):
#         def prepare_results(m, p, r, f):
#             return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)
        
#         self._prepare_references_and_hypothesis()


#         for aggregator in ['Avg', 'Best']:    
#             print('Evaluation with {}'.format(aggregator))
#             apply_avg = aggregator == 'Avg'
#             apply_best = aggregator == 'Best'
        
#             evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
#                                    max_n=4,
#                                    limit_length=True,
#                                    length_limit=100,
#                                    length_limit_type='words',
#                                    apply_avg=apply_avg,
#                                    apply_best=apply_best,
#                                    alpha=0.5, # Default F1_score
#                                    weight_factor=1.2,
#                                    stemming=True)
        
        
        
#             scores = evaluator.get_scores(self.hypotheses, self.references)
        
#             for metric, results in sorted(scores.items(), key=lambda x: x[0]):
#                 if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
#                     for hypothesis_id, results_per_ref in enumerate(results):
#                         nb_references = len(results_per_ref['p'])
#                         for reference_id in range(nb_references):
#                             print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
#                             print('\t' + prepare_results(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
#                     print()
#                 else:
#                     print(prepare_results(metric, results['p'], results['r'], results['f']))
#             print()



class Method:
    def __init__(self,df):
        
        self.df=df
        #evaluation_object=ROUGE()
        self.available_methods=['surprise.KNNBasic']        
        
    def show_methods(self):
        print('The avaliable methods are:')
        for i,method in enumerate(self.available_methods):
            print(str(i)+': '+method)



    def run(self,the_method):
        self.the_method=the_method
        if(self.the_method[0:8]=='surprise'):
            self.run_surprise()
        elif(self.the_method[0:6]=='Gensim'):
            self.run_gensim()
        elif(self.the_method[0:13]=='Transformers-'):
            self.run_transformers()
        else:
            print('This method is not defined! Try another one.')

    def run_surprise(self):
        from surprise import Reader
        from surprise import Dataset
        from surprise.model_selection import train_test_split
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.df[['userID', 'itemID', 'rating']], reader)        
        trainset, testset = train_test_split(data, test_size=.30)
        the_method=self.the_method.replace("surprise.", "")
        from surprise import KNNBasic
        the_algorithm=locals()[the_method]()
        the_algorithm.fit(trainset)
        self.predictions=the_algorithm.test(testset)
        #the_algorithm=importlib.import_module('matplotlib.text')
        #self.predictions_df
        list_predictions=[(uid,iid,r_ui,est) for uid,iid,r_ui,est,_ in self.predictions]        
        self.predictions_df = pd.DataFrame(list_predictions, columns =['user_id', 'item_id', 'rating','predicted_rating'])


if __name__ == '__main__':
    
#    sumyMethods=SumyMethod()
#    sumyMethods.show_method()
    
    data=Data()
    data.show_available_databases()
    data.read_data('ml_100k')
    #print(data.df.head(100))
    method=Method(data.df)    
    method.run('surprise.KNNBasic')
    predictions_df=method.predictions_df    
    evaluator=Evaluator(predictions_df)
    evaluator.run('surprise.rmse')
    
#df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]
#df.loc[df['column_name'] == some_value]
