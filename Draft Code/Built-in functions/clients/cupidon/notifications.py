import requests


class Slack:

    def __init__(self, endpoint=None):
        """
        Encapsulateur pour générer des messages Slack sur la chaîne #cupidon

        Args:
            endpoint (str): URL du webhook.
        """
        if endpoint is None:
            endpoint = 'https://hooks.slack.com/services/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'
        self.endpoint = endpoint

    def send_message(self, text, attachments=None):
        """
        Envoie un message Slack.

        Args:
            text (str): Le message à envoyer.
            attachments (dict): des pièces jointes.

        Raises:
            ValueError: si le message n'est pas envoyé.

        """

        message = {'text': text}

        if attachments is not None:
            message['attachments'] = attachments

        response = requests.post(self.endpoint, json=message)

        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )
