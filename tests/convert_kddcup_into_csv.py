from sklearn.datasets import fetch_kddcup99
import pandas as pd


def kddcup99_into_csv(subsets, filenames):
    for sub,name in zip(subsets, filenames):
        print("Début")
        # Download the full subset datset 
        x, y = fetch_kddcup99(subset=sub, percent10=False, return_X_y=True)
        print("Ok")

        # 0 normal 1 anomalie
        y_bin = (y != b'normal.').astype(int)

        # Export in csv
        pandas_into_csv(x,y_bin, name)
        print("Ok csv")



def pandas_into_csv(x,y, name):
    df = pd.DataFrame(x)
    df['label'] = y
    df.to_csv(name, index=False)
    print("csv généré :", name)

if __name__ == "__main__":
    
    kddcup99_into_csv(['http','SA'],['../data_rapport/KDDCUP99_HTTP_SKLEARN','../data_rapport/KDDCUP99_SA_SKLEARN'])
    