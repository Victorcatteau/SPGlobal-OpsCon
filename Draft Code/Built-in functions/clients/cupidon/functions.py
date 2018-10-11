import pandas as pd
import numpy as np
import re

from tqdm import tqdm


def reformat(text):
    """
    Supprime les espaces redondants du champ ``text``.

    Args:
        text: texte à reformater.

    Returns:
        Le texte reformaté. Un seul espace **entre** les mots.

    Examples:

        >>>reformat(' Hello      World  !   ')
        'Hello World !'

    """

    words = [word for word in text.split(' ') if len(word) != 0]
    return ' '.join(words)


def transform_special_characters(text, special_characters):
    """
    Transforme les caractères d'espace bizarres (comme ``\t``, ``\u3000`` ou ``\xa0``) en espace latin simple.

    Args:
        text (str): le texte à modifier.
        special_characters (list): liste des caractères à transformer par une espace.

    Returns:
        str: le texte transformé.

    Examples:

        >>> transform_special_characters('Hello World!', ['!'])
        'Hello World '

        >>> transform_special_characters('Hello World!', ['!', 'o'])
        'Hell  W rld '

    """

    for special_char in special_characters:
        text = text.replace(special_char, ' ')

    return text


def replace_null(pattern, negated=False):
    """

    Args:
        pattern:
        negated:

    Returns:

    Examples:

        >>>replace_null(r'\D')('000E000')
        'N.A.'

    """

    def f(cell):
        try:
            if (re.search(pattern, cell) is None) == negated:
                return 'N.A.'
            else:
                return cell
        except AttributeError:
            return cell
        except TypeError:
            return cell

    return f


def correct_mail(domain):
    """

    Args:
        domain:

    Returns:

    """


