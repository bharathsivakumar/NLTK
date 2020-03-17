import pandas as pd
data_cleaned_df = pd.read_pickle('cleaned_data.pickle')

def get_reviews_for_model(cleaned_tokens_list):
    for reviews_tokens in cleaned_tokens_list:
        yield dict([reviews_tokens, True] for token in reviews_tokens)

data_cleaned_review = data_cleaned_df.iloc[:,0].apply(get_reviews_for_model)

list_dict_review = []

for review in data_cleaned_review:
    list_dict_review.append(list(review))

dict_review = pd.Series(list_dict_review, name = 'review')

data_model_df = pd.concat([dict_review,data_cleaned_df.iloc[:,1] ], axis = 1)
data_model_df.to_pickle('model_data.pickle')
print( data_model_df.head() )