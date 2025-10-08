import pandas as pd

class Dataset:
    data = None
    features = None
    
    def __init__(self, path):
        self.data = pd.read_csv(path)
        
    def __len__(self):
        return len(self.data)

    def getFeatures(self):
        return self.data.columns.tolist()
    
    def getData(self):
        return self.data.values
    