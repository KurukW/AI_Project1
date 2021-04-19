'''
Compter le nombre de videos qu'on a pour chaque label

'''
import pandas as pd
labels_lists = pd.read_csv('labels_uses.csv')

def count(labels_path):
    labels_csv = pd.read_csv(labels_path)
    labels = labels_csv.values
    #Créé un dictionnaire qui associe le nom du label et un numérique afin de le mettre dans le modele
    labels_n = {}
    for label in labels_lists.values:
        labels_n[label[0]] = 0

    for label,video_name in labels:
        if label in labels_n:
            labels_n[label] += 1

    print(labels_n)

if __name__ == '__main__':
    labels_csv = pd.read_csv('labels_tests.csv')
    count(labels_csv)
