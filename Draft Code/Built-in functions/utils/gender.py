#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
GENDER GUESSER

Started on the 20/09/2017

Creators : 
    - Gregoire BOUSSAC
    - Theo ALVES DA COSTA

------------------------------------------------------------------------
"""


try:
    from gender_guesser.detector import Detector
except:
    pass





#=============================================================================================================================
# USING LIBRARY
#=============================================================================================================================


def guess_gender(first_name,detector = None):
    if detector is None:
        detector = Detector(case_sensitive = False)

    gender = detector.get_gender(first_name)
    mapping = {
        "mostly_male":"male",
        "mostly_female":"female",
        "andy":"androgynous"
    }

    if gender in mapping:
        gender = mapping[gender]

    return gender








#=============================================================================================================================
# CUSTOM FUNCTION
#=============================================================================================================================



def get_gender(self):
    ########################################
    ## Temporary solution to get gender ? ##
    ########################################

    first_name = self.name.partition(' ')[0].lower()

    # Composite names
    if '-' in first_name:
        # We only keep the first part of the first name
        first_name = first_name.partition('-')[0]

    # Scrapping
    url_search = 'https://www.behindthename.com/names/search.php?terms=' + first_name
    response = requests.get(url_search)

    # If behinthename doesn't 'work', we use the genderize API (1000
    # request per day)
    genderize = False

    # The URL redirection is in the Location header
    if not response.history:  # No 302 response
        genderize = True
    else:
        try:
            request_text = request.urlopen(url).read()
            page = bs4.BeautifulSoup(request_text, "lxml")
            html_extract = page.find('div', {'class': 'namesub'})

            if html_extract is None:
                genderize = True
        except:
            genderize = True

    if genderize:
        url = 'https://api.genderize.io/?name=' + first_name
        response = requests.get(url).json()
        self.gender = response.get('gender', 'Unknown')

        if self.gender is not None:
            self.gender = self.gender[0]  # 'male' or 'female'
            self.gender = self.gender[0].upper()  # 'M' or 'F'

        return(self.gender)
    else:
        gender = html_extract.getText()
        # 'GENDER: Masculine' or 'GENDER: Feminine'
        gender = gender.partition(' ')[2]
        self.gender = gender[0]

        return(self.gender)  # Returns the first letter of the gender




