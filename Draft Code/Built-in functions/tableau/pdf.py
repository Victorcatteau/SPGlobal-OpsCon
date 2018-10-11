import tableauserverclient as tsc
from urllib import request

import os


class TableauTabCmd:

    def __init__(self, username, password, server_url='http://bigdata.ekimetrics.com'):
        """
        Encapsulateur pour obtenir des vues PDF.

        Cette méthode utilise ``tabcmd``, qui doit donc être installé.

        Voir le Notebook pour un exemple concret.

        Args:
            username (str): l'identifiant
            password (str): le mot de passe
            server_url (str): l'URL du serveur Tableau
        """

        self.username = username
        self.password = password

        self.server_url = server_url

    def login(self):
        os.system('tabcmd login -u ' + self.username + ' -p ' + self.password + ' -s ' + self.server_url)

    @staticmethod
    def logout():
        os.system('tabcmd logout')

    @staticmethod
    def get_pdf(workbook, view, filepath, filters=None, orientiation=None, pagesize=None):
        """
        Permet de récupérer une vue en PDF.

        Args:
            workbook (str): nom du workbook à récupérer.
            view (str): nom de la vue à récupérer.
            filepath (str): chemin vers le fichier PDF à créer.
            filters (dict): filtres éventuels à appliquer à la vue pour l'exportation.
            orientation (str): ``landscape`` ou ``portrait``. Par défaut, utilise les paramètres de Tableau Server.
            pagesize (str): taille du page. Par défaut, ``tabcmd`` utilise 'letter'.
                'unspecified', 'letter', 'legal', 'note folio', 'tabloid', 'ledger',
                'statement', 'executive', 'a3', 'a4', 'a5', 'b4', 'b5' ou 'quarto'.
        """

        query = '"' + workbook + '/' + view

        if isinstance(filters, dict):
            query += '?'
            for key in filters:
                query += request.quote(key) + '=' + request.quote(filters[key]) + '&'

        query += '"'

        file_path = '"' + filepath + '"'

        tabcmd_query = 'tabcmd export ' + query + ' --pdf -f ' + file_path

        if orientation is not None:
            tabcmd_query += ' --orientation ' + orientation

        if pagesize is not None:
            tabcmd_query += ' --pagesize "' + pagesize + '"'

        os.system(tabcmd_query)

class TableauREST:
    """
    Encapsulateur pour se connecter à Tableau Server.
    """

    def __init__(self, username, password, server_url='http://bigdata.ekimetrics.com'):

        self.auth = tsc.TableauAuth(username=username, password=password)
        self.server = tsc.Server(server_url)

        self.server_url = server_url

    # def __enter__(self):
    #     self.server.auth.sign_in(self.auth)
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.server.auth.sign_out()

    def get_views(self):
        with self.server.auth.sign_in(self.auth):
            all_views, pagination_item = self.server.views.get()
            # print([view.name for view in all_views])

        return all_views
