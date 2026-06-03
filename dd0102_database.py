import pandas as pd

def get_all_segment_data():
    return pd.DataFrame({
        'Patient':['1011']*3,
        'Seg_Reindexed':[1,2,3],
        'Ca_reported':[0.05,0.07,0.09]
    })
