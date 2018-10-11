import pandas as pd

import cupidon.cleansing.maisons.piaget as piaget
from cupidon.notifications import Slack


class Piaget:

    def __init__(self, verbosity=0, path='example-data/', output_path='outputs/'):

        self.verbose = verbosity > 0
        self.individual_verbose = verbosity > 1

        self.uae = piaget.UnitedArabEmirates(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.china = piaget.China(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.taiwan = piaget.Taiwan(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.korea = piaget.Korea(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.russia = piaget.Russia(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.middle_east = piaget.MiddleEast(verbose=self.individual_verbose, path=path, output_path=output_path)

        self.worldwide = piaget.Worldwide(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.japan = piaget.Japan(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.singapore = piaget.Singapore(verbose=self.individual_verbose, path=path, output_path=output_path)
        self.hongkong = piaget.HongKong(verbose=self.individual_verbose, path=path, output_path=output_path)

        self.zones = [self.uae, self.china, self.taiwan, self.korea,
                      self.russia, self.middle_east, self.worldwide,
                      self.japan, self.singapore, self.hongkong]

        self.fill_rates = pd.DataFrame()

        self.mirrored = None

        self.slack = Slack()

    def import_data(self):

        self.print('Importation des données :')
        self.print()

        for zone in self.zones:
            self.print('>> ' + zone.name + '... ', end='')
            zone.import_data()
            self.print('Terminé.')

    def counts(self):

        temp = []

        for zone in self.zones:
            t = zone.fill_rate().loc[zone.monitored_columns][['Count']]
            t.columns = [zone.name]
            t.index = [
                'First Name', 'Last Name', 'Address Line 1', 'Address Line 2', 'Address Line 3', 'City', 'Postal Code',
                'Country', 'E-Mail', 'Phone Number', 'Birth Date (Day)', 'Birth Date (Month)', 'Birth Date (Year)'
            ]

            t.loc['Count'] = len(zone)

            temp.append(t)

        return pd.concat(temp, axis=1)

    def run_fill_rates(self, timing='Before'):

        counts = self.counts()

        rate = counts.sum(axis=1)
        rate /= rate.loc['Count']

        self.fill_rates[timing + ' cleansing'] = rate.drop('Count')

    def run(self):

        self.print('Lancement de la procédure de nettoyage des données.')
        self.print()

        # self.import_data()
        #
        # self.run_fill_rates('Before')

        for zone in self.zones:
            self.print('>> ' + zone.name + '... ', end='')
            zone.run()
            self.print('Terminé.')

        self.run_fill_rates('After')

    def save(self):

        self.print('>> Enregistrement... ', end='')

        for zone in self.zones:
            zone.save_data()

        self.print('Terminé.')

    def generate_mirror_version(self):

        self.print('>> Création de la version miroir... ', end='')

        for zone in self.zones:
            zone.generate_mirror_version()

        self.print('Terminé.')

    def save_mirror_version(self):

        self.print('>> Enregistrement de la version miroir... ', end='')

        for zone in self.zones:
            zone.save_mirror_version()

        self.print('Terminé.')

    def full_run(self):

        self.slack.send_message('Lancement du nettoyage des données Piaget.')

        self.run()

        self.print()
        self.generate_mirror_version()

        self.print()
        self.save()
        self.save_mirror_version()

        self.slack.send_message('Nettoyage terminé !')

    def print(self, text='', end='\n'):
        if self.verbose:
            print(text, end=end)
