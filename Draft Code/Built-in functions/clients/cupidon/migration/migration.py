import pandas as pd
import numpy as np

from cupidon.resources import BaseData, RESOURCE_PATH


class Deduplication:

    def __init__(self, source, name, input_path, to_fill):
        self.data = None
        self.duplicates = None

        self.migration = None

        self.name = name
        self.source = source

        self.input_path = input_path

        self.to_fill = to_fill

    def import_data(self):
        self.data = pd.read_csv(self.input_path + self.source + '/' + self.name + '.csv', sep=';')
        self.duplicates = pd.read_csv(self.input_path + self.source + '/' + self.name + ' - Pairs.csv', sep=';')

    def get_migration_canvas(self):

        paired = self.duplicates.copy()

        paired['Client ID'] = paired['Parent']
        paired = pd.concat([paired, self.duplicates.copy()])

        self.migration = pd.merge(paired, self.data, on=['Client ID'], how='left')

    def flag_parent(self):
        self.migration['Is Parent'] = self.migration['Parent'] == self.data['Client ID']
