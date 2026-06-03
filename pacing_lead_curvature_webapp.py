import pandas as pd

def run_web_analysis(files_dict, *args, **kwargs):
    # simplified mock returning structure expected by app
    return {'table': pd.DataFrame({'Max_Ca':[0.05,0.07,0.09]})}

MATERIAL_PROPERTIES={'MP35N':{}}
DEFAULT_SAFETY={'VisONE':0.88}
UNIT_LABELS={'um':'um'}
