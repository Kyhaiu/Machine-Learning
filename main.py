import Screen as sc
import pandas as pd
import numpy  as np
import sklearn.model_selection as skms


def main():
    screen = sc.Screen()
    
    csv = pd.read_csv(screen.getFilename(), sep=',')
    classes = csv['Class']
    aux = skms.train_test_split(csv, test_size = 0.5, train_size = 0.5, shuffle = True, stratify=classes)
    train = aux[0]
    classes = aux[1]['Class']

    aux = skms.train_test_split(aux[1], test_size = 0.5, train_size = 0.5, shuffle = True, stratify=classes)
    validation = aux[0]
    test = aux[1]

    print(train)
    print(validation)
    print(test)
    


main()