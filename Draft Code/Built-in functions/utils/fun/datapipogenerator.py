
import numpy as np
import random



class DataPipoGenerator(object):
    def __init__(self):

        # PIPO EXPRESSIONS
        self.pipo1 = ["Face à","Relativement à","Pour optimiser","Pour accentuer","Afin de maîtriser","Au moyen d#","Depuis l'émergence d#","Pour challenger", "Pour défier","Pour résoudre","En termes de redynamisation d#","Concernant l'implémentation d#","À travers","En s'orientant vers","En termes de process, concernant","En rebondissant sur","Pour intégrer","Une fois internalisée","Pour externaliser","Dans la lignée d#","En synergie avec","Là où les benchmarks désignent","Au cœur d#","En auditant","Une fois evaluée","Partout où domine","Pour réagir à","En jouant","Parallèlement à","Malgré","En réponse à","En réaction à","Répliquant à","En phase de montée en charge d#", "En réponse à","En phase de montée en charge d#", "Grâce à", "Perpendiculairement à", "Indépendamment d#", "Corrélativement à", "Tangentiellement à", "Concomitamment à","Par l'implémentation d#"]
        self.pipo2 = ["la problématique","l'opportunité","la mondialisation","une globalisation","la bulle","la culture","la synergie","l'efficience","la compétitivité","une dynamique","une flexibilité","la revalorisation","la crise","la stagflation","la convergence","une réactivité","une forte croissance","la gouvernance","la prestation","l'offre","l'expertise","une forte suppléance","une proposition de valeur","une supply chain","la démarche", "une plate-forme", "une approche", "la mutation","l'adaptabilité", "la pluralité", "une solution", "la multiplicité","la transversalité","la mutualisation"]
        self.pipo3 = ["opérationnelle,","quantitative,","des expertises,","porteuse,","autoporteuse,","collaborative,","accélérationnelle,","durable,","conjoncturelle,","institutionnelle,","managériale,","multi-directionnelle,","communicationnelle,","organisationnelle,","entrepreneuriale,","motivationnelle,","soutenable,","qualitative,","stratégique,","interne / externe,","online / offline,","situationnelle,","référentielle,","institutionnelle,","globalisante,","solutionnelle,","opérationnelle,","compétitionnelle,","gagnant-gagnant,","interventionnelle,","sectorielle,","transversale,","des prestations,","ambitionnelle,","des sous-traitances,", "corporate,", "asymétrique,", "budget", "référentielle"]
        self.pipo4 = ["les cadres doivent ","les personnels concernés doivent ","les personnels concernés doivent ","les N+1 doivent ","le challenge consiste à","le défi est d#","il faut","on doit","il faut","on doit","il faut","on doit","il faut","on doit","chacun doit","les fournisseurs vont","les managers décident d#","les acteurs du secteur vont","les responsables peuvent","la conjecture peut","il est impératif d#","un meilleur relationnel permet d#","une ambition s'impose :","mieux vaut","le marché exige d#","le marché impose d#","il s'agit d#","voici notre ambition :","une réaction s'impose :","voici notre conviction :","les bonnes pratiques consistent à","chaque entité peut","les décideurs doivent","il est requis d#","les sociétés s'engagent à","les décisionnaires veulent","les experts doivent","la conjecture pousse les analystes à","les structures vont","il faut un signal fort :","la réponse est simple :","il faut créer des occasions :","la réponse est simple :","l'objectif est d#","l'objectif est évident :","l'ambition est claire :","chaque entité doit","une seule solution :","il y a nécessité d#","il est porteur d#","il faut rapidement","il faut muscler son jeu : ","la réponse client permet d#","la connaissance des paramètres permet d#", "les éléments moteurs vont"]
        self.pipo4data = ["les data scientists doivent","les data managers doivent","les data engineers doivent","les web developpers doivent","les full stack developpers doivent","les chief data officers doivent","les services de protection des données doivent","les startups doivent"]
        self.pipo5 = ["optimiser","faire interagir","capitaliser sur","prendre en considération","anticiper ","intervenir dans","imaginer","solutionner","piloter","dématerialiser","délocaliser","coacher","investir sur","valoriser","flexibiliser","externaliser","auditer","sous-traiter","revaloriser","habiliter","requalifier","revitaliser","solutionner","démarcher","budgetiser","performer","incentiver","monitorer","segmenter","désenclaver",  "décloisonner", "déployer", "réinventer", "flexibiliser", "optimiser", "piloter","révolutionner", "gagner", "réussir", "connecter", "faire converger", "planifier", "innover sur", "monétiser", "concrétiser","impacter", "transformer", "prioriser", "chiffrer", "initiativer", "budgetiser", "rénover", "dominer"]
        self.pipo6 = ["solutions","issues","axes mobilisateurs","problématiques","cultures","alternatives","interactions","issues","expertises","focus","démarches","alternatives","thématiques","atouts","ressources","applications","applicatifs","architectures","prestations","process","performances","bénéfices","facteurs","paramètres","capitaux","sourcing","émergences","kick-off","recapitalisations","produits","frameworks","focus", "challenges","décisionnels","ouvertures","fonctionnels","opportunités","potentiels","territoires","leaderships","applicatifs","prestations","plans sociaux","wordings","harcèlements","monitorings","montées en puissance","montées en régime","facteurs","harcèlements","référents", "éléments", "nécessités", "partenariats", "retours d'expérience", "dispositifs", "potentiels", "intervenants","directives","directives","perspectives","contenus","implications","kilo-instructions","supports","potentiels","mind mappings", "thématiques","workshops","cœurs de mission", "managements", "orientations", "cibles"]
        self.pipo7 = ["métier","prospect","customer","back-office","client","envisageables","à l'international","secteur","client","vente","projet","partenaires","durables","à forte valeur ajoutée","soutenables","chiffrables","évaluables","force de vente","corporate","fournisseurs","bénéfices","convivialité","compétitivité","investissement","achat","performance","à forte valeur ajoutée","dès l'horizon 2020","à fort rendement","qualité", "logistiques", "développement", "risque", "terrain", "mobilité", "praticables", "infrastructures", "organisation", "projet", "recevables", "investissement", "conseil", "conseil", "sources", "imputables", "intermédiaires", "leadership","pragmatiques","framework","coordination","d'excellence","stratégie","de confiance", "crédibilité", "compétitivité", "méthodologie", "mobilité", "efficacité", "efficacité"]
        self.pipo7data = ["data","machine learning","deep learning","économétriques","statistiques","mathématiques","des données CRM","régressions","clustering"]

    def replace_hash(self,x):
        pass




    def generate(self):

        generation = []
        pipo_list = [
                [*["pipo{}".format(i) for i in range(1,4)],"pipo4data",*["pipo{}".format(i) for i in range(5,7)],"pipo7data"],
                # [*["pipo{}".format(i) for i in range(1,4)],"pipo4data",*["pipo{}".format(i) for i in range(5,7)],"pipo7"],
                # [*["pipo{}".format(i) for i in range(1,4)],"pipo4",*["pipo{}".format(i) for i in range(5,7)],"pipo7data"],
                ]

        pipo_choice = random.choice(pipo_list)

        for pipo in pipo_choice:
            generation.append(random.choice(getattr(self,pipo)))



        # ADD "LES"
        generation[5] = "les " + generation[5]


        # HASHTAGS
        for i,expression in enumerate(generation):
            if expression[-1] == "#":
                if i +1 < len(generation) and generation[i+1][0] in "aeiouyhéà":
                    generation[i] = generation[i].replace("#","'") + generation[i+1]
                    generation[i+1] = ""
                else:
                    generation[i] = generation[i].replace("#","e")

        generation = [expression for expression in generation if expression != ""]


        return " ".join(generation)
