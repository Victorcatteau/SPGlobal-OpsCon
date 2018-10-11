from time import time

import gc

from functools import wraps

LINE_LENGTH = 60


def description(desc):
    """
    Decorator function generator in order to print a description with ``self.print``.

    Args:
        desc (str): The text description to print.

    Returns:
        A decorator function.

    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]
            self.print('>> ' + desc + '... ', end='')
            t = time()

            res = func(*args, **kwargs)

            self.print('Terminé.   (' + str(round(time() - t, 2)) + 's)')

            return res

        return f

    return decorator


def run_description(func):
    """
    Decorator function that wraps ``func`` wit a description and prints the elapsed time.

    Args:
        func (type.FunctionType): the function to decorate.

    Returns:
        A decorated function.
    """

    @wraps(func)
    def f(*args, **kwargs):

        self = args[0]

        if self.verbose:
            line = '-' * LINE_LENGTH
            message = ' Traitement de ' + self.name + ' '

            length = len(line) // 2 - len(message) // 2

            print()
            print(line)
            print('-' * length + message + '-' * (len(line) - length - len(message)))
            print(line)
            print()

        else:
            print('>> Traitement de ' + self.name + '... ', end='')

        t = time()

        res = func(*args, **kwargs)

        if self.verbose:
            line = '-' * 60
            message = ' Terminé après ' + str(round(time() - t, 2)) + 's '

            length = len(line) // 2 - len(message) // 2

            print()
            print('-' * length + message + '-' * (len(line) - length - len(message)))
            print(line)
            print()

        else:
            print('Terminé.   (' + str(round(time() - t, 2)) + 's)')

        return res

    return f


def describe_name(desc):
    """
    Decorator function generator in order to print a description with ``self.print``.

    Args:
        desc (str): The text description to print.

    Returns:
        A decorator function.

    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]
            self.print('>> ' + desc + ' ' + self.name + '... ', end='')
            t = time()

            res = func(*args, **kwargs)

            self.print('Terminé.   (' + str(round(time() - t, 2)) + 's)')

            return res

        return f

    return decorator


def main_description(desc):
    """
    Decorator function generator in order to print a description with ``self.print``.

    Args:
        desc (str): The text description to print.

    Returns:
        A decorator function.

    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]
            self.print('>> ' + desc + ' :')
            t = time()

            res = func(*args, **kwargs)

            self.print('   Terminé.   (' + str(round(time() - t, 2)) + 's)')

            return res

        return f

    return decorator


def sub_description(desc):
    """
    Decorator function generator in order to print a description with ``self.print``.

    Args:
        desc (str): The text description to print.

    Returns:
        A decorator function.

    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]
            self.print('   > ' + desc + '... ', end='')
            t = time()

            res = func(*args, **kwargs)

            self.print('Terminé.   (' + str(round(time() - t, 2)) + 's)')

            return res

        return f

    return decorator


def slack(message):
    """
    Returns a decorator function that sends out notifications.

    Args:
        message (message): The text description for the launch.

    Returns:
        A decorator function.

    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]

            self.slack.send_message(message)

            try:
                res = func(*args, **kwargs)
                self.slack.send_message('Procédure terminée !')
                return res
            except Exception as e:
                self.slack.send_message(
                    'Aie aie aie, une erreur est survenue ! ```' + str(e) + '```',
                    attachments={
                        'text': '```' + str(e.__traceback__) + '``'
                    }
                )

        return f

    return decorator


def memory(func):

    @wraps(func)
    def f(*args, **kwargs):
        res = func(*args, **kwargs)
        gc.collect()
        return res

    return f


def delete(func):

    @wraps(func)
    def f(*args, **kwargs):
        self = args[0]

        res = func(*args, **kwargs)

        del self
        gc.collect()

        return res

    return f
