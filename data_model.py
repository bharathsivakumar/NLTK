import pandas as pd
data_cleaned_df = pd.read_pickle( 'cleaned_data.pickle' )
print( (data_cleaned_df.head()) )

def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict

#token_to_dict = lambda x : create_word_features(x)

dict_review = data_cleaned_df.iloc[:,0].apply(create_word_features)

#list_dict_review = []

#for review in data_cleaned_review:
#    list_dict_review.append(list(review))

#dict_review = pd.Series(list_dict_review, name = 'review')

data_model_df = pd.concat([dict_review,data_cleaned_df.iloc[:,1] ], axis = 1)
print( data_model_df.head() )
data_model_df.to_pickle('model_data.pickle')
#print( data_model_df.head() )