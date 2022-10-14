import pandas as pd

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

class Method:
    def __init__(self,df):
        
        self.df=df
        self.available_methods=[
            'surprise.NormalPredictor',
            'surprise.BaselineOnly',
            'surprise.KNNBasic',
            'surprise.KNNWithMeans',
            'surprise.KNNWithZScore',
            'surprise.KNNBaseline',
            'surprise.SVD',
            'surprise.SVDpp',
            'surprise.NMF',
            'surprise.SlopeOne',
            'surprise.CoClustering',
        ]        
        
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
        eval(f"exec('from surprise import {the_method}')")
        the_algorithm=locals()[the_method]()
        the_algorithm.fit(trainset)
        self.predictions=the_algorithm.test(testset)
        list_predictions=[(uid,iid,r_ui,est) for uid,iid,r_ui,est,_ in self.predictions]        
        self.predictions_df = pd.DataFrame(list_predictions, columns =['user_id', 'item_id', 'rating','predicted_rating'])

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
        import surprise
        from surprise import accuracy
        predictions=[surprise.prediction_algorithms.predictions.Prediction(row['user_id'],row['item_id'],row['rating'],row['predicted_rating'],{}) for index,row in self.predictions_df.iterrows()]
        self.predictions=predictions
        self.the_evaluator= 'accuracy.' + self.the_evaluator.replace("surprise.", "")
        from surprise import accuracy
        print(eval(f'{self.the_evaluator}(predictions,verbose=True)'))