import pandas as pd
import numpy as np

import re

from cupidon.resources import BaseData
from cupidon import functions

from cupidon.decorators import description, main_description, sub_description, run_description

import warnings

warnings.filterwarnings('ignore')


class BaseRichemont(BaseData):

    def __init__(self, name, source, verbose=False, encoding='utf-8',
                 sep=';', path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        """
        Classe de base pour l'ouverture des bases.

        L'encodage étant aléatoire sur les bases qu'on reçoit, il est placé en paramètre.

        Args:
            name (str): le nom de l'objet/source que l'on manipule
            verbose (bool): la verbosité de l'objet
            encoding (str): l'encoding des données
            sep (str) : le séparateur des données
        """

        super().__init__(name=name, verbose=verbose)

        self.source = source

        self.path = path
        self.output_path = output_path

        self.encoding = encoding
        self.sep = sep

        self.name_columns = None
        self.phone_columns = None
        self.mail_columns = None
        self.address_columns = None
        self.postal_code_columns = None

        self.mirrored = None
        self.mirror_path = mirror_path
        self.column_mapping_path = column_mapping_path

        self.print('Source : ' + source + '\nZone : ' + name)

    def import_data(self):
        """
        Importation des données dans le champ ``data``.
        """

        self.data = pd.read_csv(
            self.path + self.source + '/' + self.name + '.csv',
            sep=self.sep,
            encoding=self.encoding,
            na_values=['?']
        )

        columns = [
            functions.transform_special_characters(column, ['\t', '\u3000', '\xa0']) for column in self.data.columns]
        self.data.columns = [functions.reformat(column) for column in columns]

        # Les pays KR et TW sont mal nommés...
        self.data.rename(columns={'PRENOM': 'FIRSTNAME', 'NOM': 'LASTNAME'}, inplace=True)

    @description('Transformation des caractères spéciaux')
    def transform_special_characters(self, special_chars):
        """
        Remplace les caractères spéciaux (``special_chars``) par des espaces.

        Args:
            special_chars (list): liste des caractères spéciaux à remplacer.

        """

        def transformation(cell):
            try:
                return functions.transform_special_characters(cell, special_chars)
            except AttributeError:
                return cell

        for column in self.data.columns:
            self.data[column] = self.data[column].apply(transformation)

    @description('Suppression des espaces redondants')
    def reformat_spaces(self):
        """
        Supprime les espaces redondants.
        """

        def reformat(cell):
            try:
                text = functions.reformat(cell)
                if len(text) == 0:
                    return 'N.A.'
                return text
            except AttributeError:
                return cell

        for column in self.data.columns.values:
            self.data[column] = self.data[column].apply(reformat)

        self.data.replace('N.A.', np.nan, inplace=True)

    def replace_null_values(self, columns, na_rep):
        """
        Transforme les valeurs correspondant à des valeurs manquantes par des champs vides.

        Args:
            columns (list): Liste des colonnes à modifier.
            na_rep (list): Liste des valeurs à transformer.

        """

        def replace_cell(cell):

            if str(cell).lower() in na_rep:
                return 'N.A.'

            return cell

        for column in columns:
            self.data[column] = self.data[column].apply(replace_cell)

        self.data.replace('N.A.', np.nan, inplaxce=True)

    @sub_description('Noms')
    def replace_null_values_name(self):
        """
        Suppression des valeurs nulles pour les champs nom.
        """

        self.replace_null_values(
            self.name_columns,
            ['na', 'n/a', 'n.a', 'n.a.', 'unknown', 'missing', 'missing long label', 'missing label', '#']
        )

        # Remplacement des 'aaaa' et des numéros en 'N.A.':
        for column in self.name_columns:
            self.data[column] = self.data[column].apply(functions.replace_null(r'^(.)\1+$|^\d+$'))

    @sub_description('Numéros de téléphones')
    def replace_null_values_phone(self):
        """
        Transformation en '' des champs nuls pour le téléphone.
        """

        self.replace_null_values(
            self.phone_columns,
            ['.', '!', '/', '#']
        )

        def f(cell):
            cell = str(cell)

            try:
                if len(cell) < 6:
                    return 'N.A.'

                # Six derniers chiffres identiques ?
                if re.search(r'^(\d)\1+$', cell[-6:]):
                    return 'N.A.'

                # Suite logique ?
                if cell in {'012345678910111213'[:len(cell)], '12345678910111213'[:len(cell)]}:
                    return 'N.A.'

                return cell
            except AttributeError:
                return cell
            except TypeError:
                return cell

        for column in self.phone_columns:
            self.data[column] = self.data[column].apply(f)

            self.data[column] = self.data[column].apply(functions.replace_null(r'^\+*[\(\)\-\d\s]+$', negated=True))

    @sub_description('Emails')
    def replace_null_values_mail(self):
        """
        Transformation en valeur vide des champs nuls pour le mail.
        """

        classic_mails = ['gmail', 'hotmail', 'orange', 'yahoo', 'tiscalli',
                         'binternet', 'googlemail', 'ntlworld', 'gmx']

        corrections = {
            'ayhoo': 'yahoo',
            'btinterent': 'btinternet',
            'btinternt': 'btinternet',
            'gmaill': 'gmail',
            'gmeil': 'gmail',
            'gmial': 'gmail',
            'gmiil': 'gmail',
            'gogglemail': 'googlemai',
            'goglemail': 'googlemai',
            'googlmail': 'googlemai',
            'hotamail': 'hotmail',
            'hotmai': 'hotmail',
            'hotmaill': 'hotmail',
            'hotmial': 'hotmail',
            'htmail': 'hotmail',
            'htomali': 'hotmail',
            'ntl.worl': 'ntlworld',
            'ntlwolrd': 'ntlworld',
            'ntlword': 'ntlworld',
            'ntlwrold': 'ntlworld',
            'omail': 'hotmail',
            'orane': 'orange',
            'tiscale': 'tiscali',
            'tiscalli': 'tiscali',
            'tiscally': 'tiscali',
            'tiscaly': 'tiscali',
            'uahoo': 'yahoo',
            'yaho': 'yahoo'
        }

        dummies = ['test', 'unknown', 'none']

        impossible_chars = ['?', '!', '#', '%', '^', "'", '"', '/', '\\']

        def treat_mail(cell):

            if not isinstance(cell, str):
                return 'N.A.'

            if len(cell) == 0:
                return 'N.A.'

            # Il y a des caractères bizarres pour l'arobase...
            cell = cell.replace('＠', '@')

            for impossible_char in impossible_chars:
                cell = cell.replace(impossible_char, '')

            # Position de l'arobase :
            at_pos = cell.find('@')

            # Si pas d'arobase :
            if at_pos == -1:
                for mail in classic_mails:
                    position = cell.find(mail)
                    if position != -1:
                        return cell[:position] + '@' + cell[position:]
                return 'N.A.'

            # Si moins de trois caractères après @
            if len(cell[at_pos:]) < 4:
                return 'N.A.'

            dot_pos = cell[at_pos:].find('.')

            # Si pas de point
            if dot_pos == -1:
                return 'N.A.'
            dot_pos += at_pos

            # Si l'adresse (avant @) fait partie de valeurs bouche-trou
            for dummy in dummies:
                if cell[:at_pos] == dummy:
                    return 'N.A.'

            address = cell[:at_pos]
            domain = cell[at_pos + 1:dot_pos]

            if domain in corrections:
                domain = corrections[domain]
                cell = address + '@' + domain + cell[dot_pos:]

            # Suppression des séquences logiques
            if address in {'012345678910111213'[:len(address)], '12345678910111213'[:len(address)]}:
                return 'N.A.'
            if domain in {'012345678910111213'[:len(domain)], '12345678910111213'[:len(domain)]}:
                return 'N.A.'

            return cell

        for column in self.mail_columns:
            self.data[column] = self.data[column].apply(treat_mail)

    @sub_description('Adresses')
    def replace_null_values_address(self):
        """
        Transformation en '' des champs nuls pour les adresses.
        """

        self.replace_null_values(
            self.address_columns,
            ['0', 'piaget', 'n/a', 'none', 'na', 'xx']
        )

        def f(cell):
            cell = str(cell)
            if cell in {'012345678910111213'[:len(cell)], '12345678910111213'[:len(cell)]}:
                return 'N.A.'
            return cell

        for column in self.postal_code_columns:
            self.data[column] = self.data[column].apply(functions.replace_null(r'^(9)+$'))
            self.data[column] = self.data[column].apply(f)

    def replace_null_values_birthdate(self):
        """
        Transformation en '' des champs nuls pour le téléphone.
        """

        pass

    @main_description('Suppression des valeurs nulles')
    def delete_null_values(self):
        """
        Suppression des champs nuls.
        """

        self.replace_null_values(
            self.data.columns,
            ['unknown', 'unknown location', 'missing', 'missing label', 'test', 'missing long label']
        )

        self.replace_null_values_name()
        self.replace_null_values_phone()
        self.replace_null_values_mail()
        self.replace_null_values_address()
        self.replace_null_values_birthdate()

        self.data.replace('N.A.', np.nan, inplace=True)

    def fill_rate(self):
        """
        Renvoie le taux de remplissage de la table

        Returns:
            pd.DataFrame: Une table présentant le taux de remplissage par colonne.

        """

        result = pd.DataFrame()
        result['Count'] = self.data.count()
        result['Fill Rate'] = self.data.count() / self.data.shape[0]
        return result

    def frequencies(self, n=50):
        """
        Renvoie le tableau des ``n`` modalités les plus fréquentes par colonne.

        Args:
            n (int): le nombre de lignes à conserver

        Returns:
            pd.DataFrame: Le tableau corresondant.

        """

        # TODO: ATTENTION aux '' qui comptent comme des champs remplis.

        data = self.data.copy()

        columns = data.columns

        if 'Unnamed: 36' in columns:
            data = data.loc[data['Unnamed: 36'].isnull()]
            data.drop('Unnamed: 36', axis=1, inplace=True)
            columns = columns[:-1]

        temp = []
        for column in columns:
            freq = data[[column]].copy()

            freq['count'] = 0
            freq = freq.groupby(column, as_index=False).count().sort_values('count', ascending=False).head(n)

            freq['freq'] = freq['count'] / len(data)
            freq['freq'] = freq['freq'].apply(lambda x: round(x, 3))

            freq.rename(columns={'freq': column + ' (freq)', 'count': column + ' (count)'}, inplace=True)

            temp.append(freq.reset_index(drop=True))

        return pd.concat(temp, axis=1)

    @run_description
    def run(self):
        self.import_data()
        self.transform_special_characters(['\t', '\u3000', '\xa0', '?', '*', '!', '~'])
        self.reformat_spaces()
        self.delete_null_values()

        self.data['Client code'] = self.data['Client code'].apply(
            lambda x: int(str(x).replace(' ', ''))
        )

        self.data.replace('N.A.', np.nan, inplace=True)

    def save_data(self):
        self.save(self.output_path + self.source + '/' + self.name + '.csv')

    @description('Sélection des colonnes')
    def select_columns(self):
        kept_columns = pd.read_csv(self.column_mapping_path + self.source + '/' + self.name + ' mapping.csv', sep=';')

        to_dico = kept_columns.dropna()
        dico = {to_dico.loc[i, 'Mapping CDB']: to_dico.loc[i, 'Champs théoriques'] for i in to_dico.index}

        columns = [column for column in kept_columns['Champs théoriques']]
        added_columns = [
            column for column in kept_columns.loc[kept_columns['Mapping CDB'].isnull()]['Champs théoriques']
        ]

        self.data.rename(columns=dico, inplace=True)

        for column in added_columns:
            self.data[column] = np.nan

        self.data = self.data[columns]

        def correct_date(cell):
            try:
                cell = cell.replace(' ', '')
                res = int(cell)

                if 0 < res < 1000 and not res == 1000:
                    return '0' + cell

                return cell
            except (TypeError, AttributeError):
                return cell

        # self.data['Month/Day of birth'] = self.data['Month/Day of birth'].apply(correct_date)
        # self.data['Year of birth'] = self.data['Year of birth'].apply(correct_date)

        self.name_columns = ['Client surname', 'Client first name']
        self.address_columns = [
            'Address Line 1', 'Address Line 2', 'Address Line 3', 'Address Line 4', 'Address Line 5', 'Address Line 6'
        ]
        self.phone_columns = ['Telephone Number']
        self.mail_columns = ['Email address']
        self.postal_code_columns = ['Address Line 6']

    @description('Génération de la base miroir')
    def generate_mirror_version(self):
        mirror = self.data.copy()

        def replace_with_void(characters):
            def f(text):
                if not isinstance(text, str):
                    return text

                for character in characters:
                    text = text.replace(character, '')
                return text
            return f

        def to_lower_no_accents(cell):
            if isinstance(cell, str):
                return cell.lower()
            return cell

        for column in mirror.columns:
            mirror[column] = mirror[column].apply(to_lower_no_accents)

        for column in self.name_columns + self.phone_columns:
            mirror[column] = mirror[column].apply(replace_with_void(['-', ' ', '+', '(', ')']))

        def check_validity(row):
            try:
                # if 'ARCHIVE' in self.data.columns:
                #     test = row['ARCHIVE'] == 1
                # else:
                #     test = False
                test = False

                for name in self.name_columns:
                    test = test or row[name].lower() in {'genericclient', 'generic'}
                return not test
            except (IndexError, AttributeError):
                return True

        try:
            mirror['Birthdate'] = mirror['Year of birth'].astype(str) + mirror['Month/Day of birth'].astype(str)
        except KeyError:
            mirror['Birthdate'] = mirror['Year of birth'].astype(str) + mirror['Month/day of birth'].astype(str)

        mirror['Data Origin'] = 'cdb'

        mirror['ToDeduplication'] = mirror.apply(check_validity, axis=1)

        self.mirrored = mirror

    def save_mirror_version(self, encoding='utf-8-sig'):
        self.mirrored.to_csv(
            self.mirror_path + self.source + '/' + self.name + ' mirroir.csv', sep=';', index=False, encoding=encoding
        )


class CDB(BaseRichemont):

    def __init__(self, encoding, name, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):

        super().__init__(name=name, verbose=verbose, encoding=encoding, source='1. CDB',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)

        self.name_columns = ['FIRSTNAME']
        self.phone_columns = ['FIXPHONENUMBER', 'MOBPHONENUMBER', 'WEBPHONENUMBER']
        self.mail_columns = ['EMAIL', 'EMAILMOBILE', 'EMAILOTHER']
        self.address_columns = ['ADRESS1', 'ADRESS2', 'ADRESS3', 'ADRESS4', 'ADRESS5', 'ADRESS6']
        self.postal_code_columns = ['ZIPLABEL']

        self.monitored_columns = [
            'FIRSTNAME', 'LASTNAME', 'ADRESS1', 'ADRESS2', 'ADRESS3', 'ADRESS4', 'ZIPLABEL',
            'PAYSCD', 'EMAIL', 'MOBPHONENUMBER', 'JOURBIRTH', 'MOISBIRTH', 'ANNEEBIRTH'
        ]

    @main_description('Importation des données')
    def import_data(self):
        """
        Vérification des colonnes.

        Import data est déjà définie dans BaseRichemont, on la redéfinie pour traiter le cas des colonnes en trop.
        """

        super().import_data()

        problem = False
        for column in self.data.columns:
            if 'Unnamed:' in column:
                problem = True
                break

        if problem:
            problems = [str(row) for row in self.data.dropna(subset=['EMAILMOBILE']).index.values]

            if len(problems) == 0:
                self.print('    > Problème de colonnes, aucune ligne directement concernée...')
            else:
                self.print('    > Problème de colonnes, aux lignes ' + ', '.join(problems))

                sound_data = self.data.loc[self.data['EMAILMOBILE'].isnull()].copy()
                sound_data = sound_data.iloc[:, :-1]

                problematic_data = self.data.dropna(subset=['EMAILMOBILE']).copy()

                def merge(row):
                    if isinstance(row['EMAIL'], str) and isinstance(row['EMAILMOBILE'], str):
                        return row['EMAIL'] + row['EMAILMOBILE']
                    return row['EMAIL']

                problematic_data['EMAIL'] = problematic_data.apply(merge, axis=1)

                problematic_data.iloc[:, 18:-1] = problematic_data.iloc[:, 19:]
                problematic_data = problematic_data.iloc[:, :-1]

                self.data = pd.concat([sound_data, problematic_data]).sort_index()

                self.print('    > Problème traité.')

    @sub_description('Dates de naissance')
    def replace_null_values_birthdate(self):
        """
        Modifications des valeurs nulles en 'N.A.'
        """

        def treat_date(scope):

            if scope == 'JOURBIRTH':
                min_date = 1
                max_date = 31
            elif scope == 'MOISBIRTH':
                min_date = 1
                max_date = 12
            else:
                min_date = 1900
                max_date = 2008

            def f(cell):
                try:
                    if isinstance(cell, str):
                        cell = int(cell)

                    if not np.isfinite(cell):
                        return 0

                    # if cell == 1000:
                    #     return 0

                    if cell < min_date or cell > max_date:
                        return 0
                    return cell
                except TypeError:
                    return 0
                except ValueError:
                    return 0

            return f

        for column in ['JOURBIRTH', 'MOISBIRTH', 'ANNEEBIRTH', 'ESTANNBIRTH']:
            self.data[column] = self.data[column].apply(treat_date(column))
            self.data[column] = pd.to_numeric(self.data[column], downcast='integer')
            self.data[column] = self.data[column].astype(str)

            # # TODO: vérifier que '' (string de longueur 0) fonctionne bien avec fill_rate().
            self.data[column] = self.data[column].replace('0', np.nan)


class SAP(BaseRichemont):

    def __init__(self, encoding, name, verbose=False, path='example-data/', output_path='outputs/'):

        super().__init__(name=name, verbose=verbose, encoding=encoding,
                         source='2. SAP', path=path, output_path=output_path)

        self.first_name_columns = ['First name', 'Full Name']
        self.phone_columns = ['Telephone', 'Telephone number', 'phone']
        self.mail_columns = ['Electronic Contact', 'E-Mail Address']
        self.address_columns = ['Street', 'Street 2', 'Street 3', 'Street 4']
        self.postal_code_columns = ['Postal Code']

        self.monitored_columns = [
            'Last name', 'First name', 'House Number', 'Street', 'Street 2', 'City', 'Postal Code', 'Country Key',
            'E-Mail Address', 'phone', 'Estimated Birthdate', 'Estimated Birthdate', 'Estimated Birthdate'
        ]

    @sub_description('Dates de naissance')
    def replace_null_values_birthdate(self):
        """
        Modifications des valeurs nulles en 'N.A.'
        """

        def treat_date(cell):
            try:
                if isinstance(cell, str):
                    cell = int(cell)

                if cell == 1000:
                    return 1000

                if not np.isfinite(cell):
                    return 0

                year = cell // 1000

                if year < 1900 or year > 2008:
                    return 0
                return cell
            except TypeError:
                return 0
            except ValueError:
                return 0

        for column in ['Partner Birthdate', 'Estimated Birthdate']:
            self.data[column] = self.data[column].apply(treat_date)
            self.data[column] = pd.to_numeric(self.data[column], downcast='integer')
            self.data[column] = self.data[column].astype(str)

            self.data[column] = self.data[column].replace('0', np.nan)
