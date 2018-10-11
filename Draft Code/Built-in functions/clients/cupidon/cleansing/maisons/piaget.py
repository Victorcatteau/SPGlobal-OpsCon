import numpy as np

from cupidon.cleansing.base import CDB, SAP
from cupidon.decorators import sub_description


class UnitedArabEmirates(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='United Arab Emirates', verbose=verbose, encoding='iso-8859-1',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class Australia(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='Australia', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class Brazil(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='Brazil', verbose=verbose, encoding='iso-8859-1',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class China(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='China', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class Korea(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='Korea', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class Taiwan(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='Taiwan', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)


class Worldwide(SAP):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/'):
        super().__init__(name='Worldwide', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path)


class Japan(SAP):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/'):
        super().__init__(name='Japan', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path)


class HongKong(SAP):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/'):
        super().__init__(name='Hong Kong', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path)


class Singapore(SAP):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/'):
        super().__init__(name='Singapore', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path)


class MiddleEast(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/'):
        super().__init__(name='Middle East', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path)


class Russia(CDB):
    def __init__(self, verbose=False, path='example-data/', output_path='outputs/', mirror_path='', column_mapping_path=''):
        super().__init__(name='Russia', verbose=verbose, encoding='utf-8',
                         path=path, output_path=output_path, mirror_path=mirror_path, column_mapping_path=column_mapping_path)

        self.name_columns = ['Name', 'First name']
        self.phone_columns = ['Tel num.', 'Tel contact']
        self.mail_columns = ['Email contact', 'E-mail']
        self.address_columns = [
            'Address line 1', 'Address line 2', 'Address line 3', 'Address line 4', 'Address line 5',
            'Address line 5', 'Active address'
        ]
        self.postal_code_columns = []

        self.monitored_columns = [
            'Name', 'First name', 'Address line 1', 'Address line 2', 'Address line 3', 'Address line 4',
            'Address line 5', 'Country', 'E-mail', 'Tel num.', 'Year of birth',
            'Estimated year of birth', 'Month/day of birth'
        ]

    @sub_description('Dates de naissance')
    def replace_null_values_birthdate(self):
        """
        Modifications des valeurs nulles en 'N.A.'
        """

        def treat_date(scope):

            if scope[:9] == 'Month/day':
                min_date = 101
                max_date = 1231
                threshold = 1000
            elif scope[:3] == 'Age':
                min_date = 1
                max_date = 120
                threshold = 0
            else:
                min_date = 1900
                max_date = 2008
                threshold = 1000

            def f(cell):

                try:
                    if isinstance(cell, str):
                        cell = int(cell.replace(' ', ''))

                    if not np.isfinite(cell):
                        return '0'

                    if cell == 1000:
                        return '1000'

                    if cell < min_date or cell > max_date:
                        return '0'

                    if cell < threshold:
                        res = '0' + str(cell)
                    else:
                        res = str(cell)

                    return res

                except TypeError:
                    return '0'
                except ValueError:
                    return '0'

            return f

        for column in ['Month/day of birth', 'Year of birth', 'Year of birth estimate',
                       'Month/day of wedding', 'Year of wedding', 'Age estimate']:
            self.data[column] = self.data[column].apply(treat_date(column)).astype(str)
            self.data[column] = self.data[column].replace('0', '')

