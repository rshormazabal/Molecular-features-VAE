import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def multilabelizer():
    
    database = pd.read_csv(r'database_raw_fixing.csv')
    labels = database['Classification'].str.split(pat='; ')

    mlb = MultiLabelBinarizer().fit(labels)

    # CREATING ONE-HOT VECTOR FOR EACH CLASS
    labels_bin_df = pd.DataFrame(mlb.transform(labels), columns = mlb.classes_)

    # CHOOSE LABELS TO TRAIN WITH
    classes = ['alcohol', 'aldehyde', 'alicycle', 'amide', 'aromatic', 'carbocycle', 'carboxylic acid',\
           'chiral', 'ester', 'ether', 'fused rings', 'ketone', 'lactame', 'metal-organic', 'nitrogen heterocycle', \
           'oxygen heterocycle', 'sulfide', 'sulfur heterocycle', 'thiol', 'urea']

    labels_class = labels_bin_df.filter(classes, axis=1)
    counts_original = labels_class.apply(pd.value_counts)
    
    return labels_class, counts_original


# =============================================================================
# for i in range(database.shape(0)):
# check = ['Right' if database['InChI'][i].__contains__(database['Formula'][i]) else 'Wrong' for i in range(database.shape[0])]
# 
# database['asd'] = check
# 
# new = database[database['InChI'].str.contains("q|p")]
# database2 = database[~database['InChI'].str.contains("q|p")]
# 
# database2.to_csv(r'new_database_checked_nocharge.csv', index=False)
# 
# =============================================================================

