import pandas as pd
import os

RESOURCE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/'


class BaseData:

    def __init__(self, name, verbose):
        """
        Classe de base pour traiter les données.

        Encapsule un DataFrame pandas et quelques champs qui seront utiles.

        Args:
            verbose (bool): Le niveau de "verbosité"
        """

        self.data = None
        self.name = name

        self.verbose = verbose
        self.log = []

    def __getitem__(self, item):
        """
        Pour accéder au champ ``data`` directement.

        Args:
            item: la colonne à laquelle accéder

        Returns:
            L'objet contenu dans l'``item``.

        """

        return self.data[item]

    def head(self, n=5):
        """
        La tête du champ ``data``.

        Args:
            n (int): nombre de lignes à renvoyer.

        Returns:
            pd.DataFrame: la tête du champ ``data``.
        """
        return self.data.head(n)

    def tail(self, n=5):
        """
        La queue du champ ``data``.

        Args:
            n (int): nombre de lignes à renvoyer.

        Returns:
            pd.DataFrame: la queue du champ ``data``.
        """
        return self.data.tail(n)

    def __len__(self):
        """
        Renvoie la taille du champ ``data``.

        Returns:
            int: ``len(self.data)``
        """
        return len(self.data)

    def update_log(self, text=''):
        """
        Met à jour le champ ``log`` (utile pour le débuggage)

        Args:
            text (str): le texte à ajouter au log.

        """
        self.log.append(text)

    def save_log(self, path):
        """
        Enregistre le ``log`` à l'adresse donnée.

        Args:
            path (str): le chemin vers le fichier texte à écrire.

        """
        with open(path, 'w') as f:
            f.writelines(self.log)

    def print(self, text='', end='\n', force=False):
        """

        Args:
            text (str):
            end (str):
            force (bool):

        Returns:

        """

        if self.verbose or force:
            print(text, end=end)

    def save(self, path):
        """
        Enregistre les données sous forme de CSV.

        Args:
            path (str): le chemin du fichier à écrire.

        """

        self.print('>> Enregistrement des données ' + self.name + '... ', end='')
        self.data.to_csv(path, sep=';', index=False, encoding='utf-8-sig')
        self.print('terminé.')


