import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import seaborn as sns
import lightgbm as lgb
import collections
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.externals import joblib
from operator import itemgetter


class BPCE_Data(object):
    """ Module for BPCE data
        collect and automate transforms of BPCE data
    """
    def __init__(self, network):
        """
        initiator for BPCE data
        """
        print('Loading raw data for {} network'.format(network)) if network in ['MySys','Equinoxe'] else print('Unknown network')
        self.address = u'C:/Users/dbettebghor/Documents/MISSIONS_ML/BPCE/RAWDATA/dataLatencySatisfaction' + network + u'ModelNew.csv'
        # self.address = u'C:/Users/rromanos/workspace-python/notebook/bpce/dataLatencySatisfaction' + network + u'ModelNew.csv'
        try:
            self.data = pd.read_csv(self.address)
            self.network  = network
            self.features = list(self.data)
            self.heatmap  = self.data.corr()
        except FileNotFoundError:
            print('No such file') 

    def __repr__(self):
        """
        print metainfo 
        """
        return "{} network observed latency over {} employees with {} Interact features".format(self.network, self.data.shape[0], self.data.shape[1])

    def __del__(self):
        """
        remove object
        """
        return "Deleting BPCE object"

    def plot_overall_satisfaction(self):
        """
        display satisfaction
        """
        sat = self.data['AppLatency'].values
        cpsat = np.copy(sat)
        cpsat[sat == 'YES'] = True
        cpsat[sat == 'NO'] = False       
        slices = [np.sum(cpsat == True),np.sum(cpsat == False)]
        activities = ['Satisfied','Not Satisfied']
        cols = ['#66CC99','#FF6666']
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10,10)
        patches, texts, autotexts = ax1.pie(slices, explode=[0.1,0.], colors = cols, labels=activities, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        texts[0].set_fontsize(18)
        texts[1].set_fontsize(18)
        texts[1].set_fontweight('bold')
        autotexts[0].set_fontsize(18)
        autotexts[1].set_fontweight('bold')
        autotexts[1].set_fontsize(24)
        plt.title('Observed satisfaction of ' + self.network + ' network',fontsize=24, fontweight='bold')
        fig1.savefig('observed_satisfaction_' + self.network + '_METEO_2_0.png',bbox_inches='tight')
        print('Chart saved as ' + 'observed_satisfaction_' + self.network + '_METEO_2_0.png')
        return plt;
   



    def addapps(self, typ):
        """
        lists and treat all apps per type per user
        """
        listapp = App_List(self, typ)
        listappsuser = listapp.apps_user(self)        
        unique_data = [list(x) for x in set(tuple(x) for x in listappsuser)]
        res = [[listappsuser.count(ens),ens] for ens in unique_data]
        apps = sorted(res, key=itemgetter(0), reverse=True)
        return apps, listappsuser

    def addtotalapps(self):
        """
        lists and treat all apps per user
        """
        listappweb = App_List(self, 'web')
        listappsuserweb = listappweb.apps_user(self) 
        listappbur = App_List(self, 'bur')
        listappsuserbur = listappbur.apps_user(self) 
        listappuser = []
        for ensweb,ensbur in zip(listappsuserweb,listappsuserbur):
            listappuser.append(ensweb+ensbur)
        unique_data = [list(x) for x in set(tuple(x) for x in listappuser)]
        res = [[listappuser.count(ens),ens] for ens in unique_data]
        apps = sorted(res, key=itemgetter(0), reverse=True)
        return apps, listappuser


    def plotapps(self, typ, ind):
        """
        plot satisfaction distribution for certain combination of apps
        """
        apps, listappsuser = self.addapps(typ)
        sat = []
        yn = self.data['AppLatency'].values
        for user, userapps in zip(range(len(listappsuser)), listappsuser):
            if userapps == apps[ind][1]:
                sat.append(yn[user])
        cpsat = np.array(sat)      
        slices = [np.sum(cpsat == 'YES'),np.sum(cpsat == 'NO')]
        percen = np.sum(cpsat == 'YES')/(np.sum(cpsat == 'YES')+np.sum(cpsat == 'NO'))
        print('{:.3} %'.format(100*percen) + ' of ' + str(apps[ind][1]) + ' users are satisfied')
        activities = ['Satisfied','Not Satisfied']
        cols = ['#66CC99','#FF6666']
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10,10)
        patches, texts, autotexts = ax1.pie(slices, explode=[0.1,0.], colors = cols, labels=activities, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        texts[0].set_fontsize(18)
        texts[1].set_fontsize(18)
        texts[1].set_fontweight('bold')
        autotexts[0].set_fontsize(18)
        autotexts[1].set_fontweight('bold')
        autotexts[1].set_fontsize(24)
        plt.title('Group ' + str(ind) + ' observed satisfaction of ' + self.network + ' network',fontsize=24, fontweight='bold')
        fig1.savefig('group_' + str(ind) + 'observed_satisfaction_' + self.network + '_METEO_2_0.png',bbox_inches='tight')
        return plt

    def totalapps(self, ind):
        """
        certain combination of apps, satisfaction, userlist
        """
        apps, listappsuser = self.addtotalapps()
        sat, un = [], []
        yn = self.data['AppLatency'].values
        for user, userapps in zip(range(len(listappsuser)), listappsuser):
            if userapps == apps[ind][1]:
                sat.append(yn[user])
                un.append(self.data['UserName'].values[user])
        #cpsat = np.array(sat)      
        return apps[ind][1], sat, un
           
    def appcloud(self, ind, ratio):
        """
        writes a tag cloud of apps for a given group
        """
        # gets combination of apps, satisfaction and usernames
        a, b, c = self.totalapps(ind)
        # a list de d'applis
        listappweb = App_List(self, 'web')
        # as we work with non fixed size sets, we will use
        # lists and not fixed size arrays, this complicates the
        # mean process
        if len(c) == 0:
            print('No user in this group')
        else:
            # get the data for the first user
            temp = self.data[self.data['UserName'] == c[0]].iloc[0]
            # initialize lists
            cnt_tran, cnt_tranwithlatency = [], []
            cnt_nbstart = []
            appstr, burstr = [], []
            # loop over apps to get the number of transactions
            # we consider the first user outside the
            # loop over users to set the righ lenght of list
            # are we add at each loop two lists
            for app in a:
                if app in listappweb.appname:
                    appstr.append(app)
                    cnt_tran.append(int(temp[app + '_NbrOfTransactions']))    
                    cnt_tranwithlatency.append(int(temp[app + '_NbrOfTransactionsWithLatency'])) 
                else:
                    burstr.append(app)
                    cnt_nbstart.append(int(temp[app + '_NbStart']))
            for user in c[1:]:
                temp2 = self.data[self.data['UserName'] == user].iloc[0]
                dnt_tran, dnt_tranwithlatency = [], []
                dnt_nbstart = []
                for app in a:
                    if app in listappweb.appname:
                        dnt_tran.append(int(temp2[app + '_NbrOfTransactions']))    
                        dnt_tranwithlatency.append(int(temp2[app + '_NbrOfTransactionsWithLatency'])) 
                    else:
                        dnt_nbstart.append(int(temp2[app + '_NbStart']))
                cnt_tran = [sum(i) for i in zip(cnt_tran,dnt_tran)]
                cnt_tranwithlatency = [sum(i) for i in zip(cnt_tranwithlatency,dnt_tranwithlatency)]
                cnt_nbstart = [sum(i) for i in zip(cnt_nbstart,dnt_nbstart)]
            indi = np.argsort(-np.array(cnt_tranwithlatency))            
            od = collections.OrderedDict()
            od2 = dict()
            # create an ordered dict, ranked by latency
            for i in indi:
                od[np.array(appstr)[i]] = cnt_tran[i]
            for i in range(len(dnt_nbstart)):
                od2[burstr[i]] = dnt_nbstart[i]
            #wordcloud = WordCloud(font_path='C:\Windows\Fonts\Metric-Regular_1.otf',background_color='white',
            #width = 800, height = 800, relative_scaling = ratio).generate_from_frequencies(od)
            #wordcloud.to_file(self.network + '_Group_' + str(ind) + '_web_applications.png')
            wordcloud2 = WordCloud(font_path='C:\Windows\Fonts\Metric-Regular_1.otf',background_color='white',
            width = 800, height = 800, relative_scaling = ratio).generate_from_frequencies(od2)
            wordcloud2.to_file(self.network + '_Group_' + str(ind) + '_bur_applications.png')            
            fig, ax = plt.subplots()
            fig.set_size_inches(25,25)
            ax.imshow(wordcloud2)
            ax.axis('off')
            plt.show()
            return plt


    def plottotalapps(self, ind):
        """
        plot satisfaction distribution for certain combination of apps
        """
        apps, listappsuser = self.addtotalapps()
        sat = []
        yn = self.data['AppLatency'].values
        for user, userapps in zip(range(len(listappsuser)), listappsuser):
            if userapps == apps[ind][1]:
                sat.append(yn[user])
        cpsat = np.array(sat)      
        slices = [np.sum(cpsat == 'YES'),np.sum(cpsat == 'NO')]
        percen = np.sum(cpsat == 'YES')/(np.sum(cpsat == 'YES')+np.sum(cpsat == 'NO'))
        print('{:.3} %'.format(100*percen) + ' of the {} users of '.format(len(sat)) + str(apps[ind][1]) + ' users are satisfied')
        activities = ['Satisfied','Not Satisfied']
        cols = ['#66CC99','#FF6666']
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10,10)
        patches, texts, autotexts = ax1.pie(slices, explode=[0.1,0.], colors = cols, labels=activities, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        texts[0].set_fontsize(18)
        texts[1].set_fontsize(18)
        texts[1].set_fontweight('bold')
        autotexts[0].set_fontsize(18)
        autotexts[1].set_fontweight('bold')
        autotexts[1].set_fontsize(24)
        plt.title('Group ' + str(ind) + ' observed satisfaction of ' + self.network + ' network',fontsize=24, fontweight='bold')
        fig1.savefig('group_' + str(ind) + 'observed_satisfaction_' + self.network + '_METEO_2_0.png',bbox_inches='tight')
        return plt

    def nb_apps_total_users(self):
        """
        computes the total number of apps by user
        """
        listapp = App_List(self, 'web')        
        nbappsweb = listapp.nb_apps_users(self) 
        listapp = App_List(self, 'bur')        
        nbappsbur = listapp.nb_apps_users(self)        
        return nbappsweb + nbappsbur

    def addntotalappusers(self):
        """
        adds a number of total apps  by user
        """
        nbapps = self.nb_apps_total_users()      
        newlabel = 'NbTotalAppUser'  
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 


    
    def addnappusers(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_apps_users(self)      
        if typ == 'bur':
            newlabel = 'NbBurAppUser'
        else:
            newlabel = 'NbWebAppUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self   

    def addtran(self, typ):
        """
        adds a number of total transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_transac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbTransacUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmediantran(self, typ):
        """
        adds the median of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_medtransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMedianTransacUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmaxtran(self, typ):
        """
        adds the max of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_maxtransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMaxTransacUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self     

    def addmmtran(self, typ):
        """
        adds the max of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_mmtransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMaxMeanTransacUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmaxmintran(self, typ):
        """
        adds the max-min of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_maxmintransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMaxMinTransacUser'   
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmadtran(self, typ):
        """
        adds the mad of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_madtransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMadTransacUser'   
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmeanadtran(self, typ):
        """
        adds the mean ad of transactions by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.nb_meanadtransac_users(self)      
        if typ == 'bur':
            print('No transactions for bureautique')
        else:
            newlabel = 'NbMeanadTransacUser'   
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addnctime(self, typ):
        """
        adds the total nc time per user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'TotalNonCompliantTimeBurUser'            
        else:
            newlabel = 'TotalNonCompliantTimeWebUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmediannctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.median_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MedianNonCompliantTimeBurUser'            
        else:
            newlabel = 'MedianNonCompliantTimeWebUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmaxnctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.max_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MaxNonCompliantTimeBurUser'            
        else:
            newlabel = 'MaxNonCompliantTimeWebUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmmnctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.mm_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MaxMeanNonCompliantTimeBurUser'            
        else:
            newlabel = 'MaxMeanNonCompliantTimeWebUser'	
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self 

    def addmaxminnctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.maxmin_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MaxMinNonCompliantTimeBurUser'            
        else:
            newlabel = 'MaxMinNonCompliantTimeWebUser' 
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self    

    def addmadnctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.mad_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MadNonCompliantTimeBurUser'            
        else:
            newlabel = 'MadNonCompliantTimeWebUser' 
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self     

    def addmeanadnctime(self, typ):
        """
        adds a number of app per type by user
        """
        listapp = App_List(self, typ)
        nbapps = listapp.meanad_noncomplianttime_users(self)      
        if typ == 'bur':
            newlabel = 'MeanadNonCompliantTimeBurUser'            
        else:
            newlabel = 'MeanadNonCompliantTimeWebUser' 
        self.data[newlabel] = nbapps
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self  

    def addwhole(self):
        """
        adds a whole set of variables
        """
        self.addntotalappusers()
        self.addnappusers('web')
        self.addnappusers('bur')
        self.addmaxtran('web')
        self.addmmtran('web')
        self.addtran('web')
        self.addmediantran('web')
        self.addmadtran('web')
        self.addmeanadtran('web')
        self.addnctime('web')
        self.addnctime('bur')
        self.addmaxnctime('web')
        self.addmaxnctime('bur')
        self.addmediannctime('web')
        self.addmediannctime('bur')
        self.addmmnctime('web')
        self.addmmnctime('bur')
        self.addmadnctime('web')
        self.addmeanadnctime('web')
        self.addmadnctime('bur')
        self.addmeanadnctime('bur')


    def convert_score(self):
        """
        convers 1-4 score into YES/NO
        """
        score = self.data.AppLatency.values
        yn    = convert(score)
        self.data['AppLatency'] = yn
        return self


    def remove_null_users(self):
        """
        removes set of users that have no transactions of office time
        """
        nullwebusers = self.data.TotalNbrOfTransactionsWithLatency == 0
        nullburusers = self.data.TotalOfficeFreezeTime == 0
        self.data = self.data.drop(self.data[nullwebusers & nullburusers].index)
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        return self

    def remove(self, typ):
        """
        remove the entire set of typ features
        """
        listapp       = App_List(self, typ)
        self.data     = self.data.drop(listapp.extendedapplist, axis = 1)   
        self.features =  list(self.data)
        self.heatmap  = self.data.corr()     
        return self

    def remove_spec(self, livar):
        """
        removes a specific list of variables
        """
        self.data     = self.data.drop(livar, axis = 1)   
        self.features =  list(self.data)
        self.heatmap  = self.data.corr()     
        return self         
  
    def get_latency(self, address):
        """
        changes self
        """
        y = self.data['AppLatency'].values
        self.applatency = y
        self.data.drop(['UserName','AppLatency'], axis = 1, inplace=True)
        self.features = list(self.data)
        self.heatmap = self.data.corr()
        colnames = pd.DataFrame({'cols' : self.data.columns.tolist()})
        colnames.to_csv(address)
        return self


    def split_latency(self):
        """
        split data between features and latency
        """
        y = self.data['AppLatency'].values
        pwd = self.data.drop(['UserName','AppLatency'], axis = 1)
        X = pwd.values   
        return X, y 

    def satisfaction(self):
    	"""
    	method that returns satisfaction
    	"""
    	ind_sat = self.data['AppLatency'] == 'YES'
    	return ind_sat

    def plotnbapp(self, ind, typ):
        """
        methods that plots distribution of satisfaction
        Parameters
        ----------
        ind : str
        'tapp'       : total number of apps
        'app'        : number of apps per user
        'tran'       : total transactions numbers (for web app only)
        'mtran'      : median of transactions per user (for web app only)
        'madtran'    : Median Absolute Deviation of transactions per user (for web app only) 
        'meanadtran' : Mean Absolute Deviation of transactions per user (for web app only)        
        'mmtran'     : max-mean of transactions per user (for web app only)
        'maxtran'    : maximum of transactions per user (for web app only)
        'maxmintran' : range of transactions per user (for web app only)
        'nct'        : total non compliant time per user
        'mnct'       : median of non compliant time over apps
        'mmnct'      : max-mean of non compliant time over apps
        'maxnct'     : maximum of non compliant time over apps
        'maxminnct'  : range of non compliant time over apps  
        'madnct'     : mad of non compliant time over apps
        'meanadnct'  : mean ad of non compliant time over apps              
        typ : str
        'bur' or 'web'
        """
        ind_sat = self.satisfaction()
        if ind != 'tapp':
            listapp = App_List(self, typ)
        if ind == 'tappwl':
            nbapp = self.data['TotalNbrOfTransactionsWithLatency'].values
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions avec latence'
                xlab = 'Transactions avec latence'             
        if ind == 'app':
            nbapp = listapp.nb_apps_users(self)
            if typ == 'bur':
                end  = 'applications bureautiques'
                xlab = 'Applications bureautiques différentes'
            elif typ == 'web':
                end  = 'applications web'
                xlab = 'Applications web différentes'   
        if ind == 'tapp':
            nbapp = self.nb_apps_total_users()
            if typ == 'bur':
                end  = 'applications bureautiques et web'
                xlab = 'Applications différentes'
            elif typ == 'web':
                end  = 'applications bureautiques et web'
                xlab = 'Applications différentes'                     
        elif ind == 'tran':
            nbapp = listapp.nb_transac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions totales'
                xlab = 'Transactions totales'             
        elif ind == 'mtran':
            nbapp = listapp.nb_medtransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions méd'
                xlab = 'Transactions méd' 
        elif ind == 'mmtran':
            nbapp = listapp.nb_mmtransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max-mean'
                xlab = 'Transactions max-mean' 
        elif ind == 'maxtran':
            nbapp = listapp.nb_maxtransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max'
                xlab = 'Transactions max' 
        elif ind == 'maxmintran':
            nbapp = listapp.nb_maxmintransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max-min'
                xlab = 'Transactions max-min'   
        elif ind == 'madtran':
            nbapp = listapp.nb_madtransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions median absolute deviation'
                xlab = 'Transactions median absolute deviation'    
        elif ind == 'meanadtran':
            nbapp = listapp.nb_meanadtransac_users(self)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions mean absolute deviation'
                xlab = 'Transactions mean absolute deviation'             
        elif ind == 'nct':
            nbapp = listapp.noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur total'
                xlab = 'Temps NC bur total'
            elif typ == 'web':
                end  = '% de temps NC web total'
                xlab = 'Temps NC web total'                                          
        elif ind == 'mnct':
            nbapp = listapp.median_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur médian'
                xlab = 'Temps NC bur médian'
            elif typ == 'web':
                end  = '% de temps NC web médian'
                xlab = 'Temps NC web médian'
        elif ind == 'maxnct':
            nbapp = listapp.max_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur max'
                xlab = 'Temps NC bur max'
            elif typ == 'web':
                end  = '% de temps NC web max'
                xlab = 'Temps NC web max'   
        elif ind == 'mmnct':
            nbapp = listapp.mm_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur max-mean'
                xlab = 'Temps NC bur max-mean'
            elif typ == 'web':
                end  = '% de temps NC web max-mean'
                xlab = 'Temps NC web max-mean'   
        elif ind == 'maxminnct':
            nbapp = listapp.maxmin_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur max-min'
                xlab = 'Temps NC bur max-min'
            elif typ == 'web':
                end  = '% de temps NC web max-min'
                xlab = 'Temps NC web max-min'
        elif ind == 'madnct':
            nbapp = listapp.mad_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur mad'
                xlab = 'Temps NC bur mad'
            elif typ == 'web':
                end  = '% de temps NC web mad'
                xlab = 'Temps NC web mad'
        elif ind == 'meanadnct':
            nbapp = listapp.meanad_noncomplianttime_users(self)
            if typ == 'bur':
                end  = 'ms de temps NC bur mean ad'
                xlab = 'Temps NC bur mean ad'
            elif typ == 'web':
                end  = '% de temps NC web mean ad'
                xlab = 'Temps NC web mean ad'
        else:
        	print('{} unknown indicator'.format(ind))
        users   = nbapp != 0
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        sns.set_style('ticks')
        sns.set(font_scale=1.6)
        sns.distplot(nbapp[ind_sat & users], kde= False,label = 
        'Satisfait, méd. = {:.3} '.format(np.median(nbapp[ind_sat & users]))+end)
        sns.distplot(nbapp[~ind_sat & users], kde =False, label = 
        'Non satisfait, méd. = {:.3} '.format(np.median(nbapp[~ind_sat & users]))+end)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Utilisateurs')
        plt.legend()
        plt.title(xlab + ' par utilisateur réseau ' + self.network)
        fig.savefig(xlab + self.network + '.png')
        return plt

class BPCE_Data_full(BPCE_Data): #inherits from BPCE_Data
    """
    class for extrapolation
    """
    def __init__(self, network):
        """
        initiator for BPCE data full
        """
        print('Loading raw data for {} network'.format(network)) if network in ['MySys','Equinoxe'] else print('Unknown network')
        self.address = u'C:/Users/dbettebghor/Documents/MISSIONS_ML/BPCE/RAWDATA/dataFullLatencySatisfaction' + network + u'Model.csv'
        # self.address = u'C:/Users/rromanos/workspace-python/notebook/bpce/dataFullLatencySatisfaction' + network + u'Model.csv'
        try:
            self.data = pd.read_csv(self.address)
            self.network  = network
            self.features = list(self.data)
            self.heatmap  = self.data.corr()
        except FileNotFoundError:
            print('No such file') 

    def __repr__(self):
        """
        print metainfo 
        """
        return "{} network over {} employees with {} Interact features".format(self.network, self.data.shape[0], self.data.shape[1])
    
    def split_latency(self):
        """
        split data between features and latency
        """
        pwd = self.data.drop(['UserName'], axis = 1)
        X = pwd.values   
        return X     

    def plotnbapp(self, ind, typ, bpcedata):
        """
        methods that plots distribution of users
        Parameters
        ----------
        ind : str
        'app'        : number of apps per user
        'tran'       : total transactions numbers (for web app only)
        'mtran'      : median of transactions per user (for web app only)
        'madtran'    : Median Absolute Deviation of transactions per user (for web app only) 
        'meanadtran' : Mean Absolute Deviation of transactions per user (for web app only)        
        'mmtran'     : max-mean of transactions per user (for web app only)
        'maxtran'    : maximum of transactions per user (for web app only)
        'maxmintran' : range of transactions per user (for web app only)
        'nct'        : total non compliant time per user
        'mnct'       : median of non compliant time over apps
        'mmnct'      : max-mean of non compliant time over apps
        'maxnct'     : maximum of non compliant time over apps
        'maxminnct'  : range of non compliant time over apps  
        'madnct'     : mad of non compliant time over apps
        'meanadnct'  : mean ad of non compliant time over apps              
        typ : str
        'bur' or 'web'
        """
        listapp = App_List(self, typ)
        listappMeteo = App_List(bpcedata, typ)
        if ind == 'app':
            nbapp = listapp.nb_apps_users(self)
            nbappMeteo = listappMeteo.nb_apps_users(bpcedata)
            if typ == 'bur':
                end  = 'applis bureautiques'
                xlab = 'Applis différentes'
            elif typ == 'web':
                end  = 'applis web'
                xlab = 'Applis différentes' 
        if ind == 'tapp':
            nbapp = self.nb_apps_total_users()
            nbappMeteo = bpcedata.nb_apps_total_users()
            if typ == 'bur':
                end  = 'applis bureautiques et web'
                xlab = 'Applis différentes'
            elif typ == 'web':
                end  = 'applis bureautiques et web'
                xlab = 'Applis différentes'    
        elif ind == 'tran':
            nbapp = listapp.nb_transac_users(self)
            nbappMeteo = listappMeteo.nb_transac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions totales'
                xlab = 'Transactions totales'             
        elif ind == 'mtran':
            nbapp = listapp.nb_medtransac_users(self)
            nbappMeteo = listappMeteo.nb_medtransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions méd'
                xlab = 'Transactions méd' 
        elif ind == 'mmtran':
            nbapp = listapp.nb_mmtransac_users(self)
            nbappMeteo = listappMeteo.nb_mmtransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max-mean'
                xlab = 'Transactions max-mean' 
        elif ind == 'maxtran':
            nbapp = listapp.nb_maxtransac_users(self)
            nbappMeteo = listappMeteo.nb_maxtransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max'
                xlab = 'Transactions max' 
        elif ind == 'maxmintran':
            nbapp = listapp.nb_maxmintransac_users(self)
            nbappMeteo = listappMeteo.nb_maxmintransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions max-min'
                xlab = 'Transactions max-min'   
        elif ind == 'madtran':
            nbapp = listapp.nb_madtransac_users(self)
            nbappMeteo = listappMeteo.nb_madtransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions median absolute deviation'
                xlab = 'Transactions median absolute deviation'    
        elif ind == 'meanadtran':
            nbapp = listapp.nb_meanadtransac_users(self)
            nbappMeteo = listappMeteo.nb_meanadtransac_users(bpcedata)
            if typ == 'bur':
                print('Transactions not available for bureautique')
            elif typ == 'web':
                end  = 'transactions mean absolute deviation'
                xlab = 'Transactions mean absolute deviation'             
        elif ind == 'nct':
            nbapp = listapp.noncomplianttime_users(self)
            nbappMeteo = listappMeteo.noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur total'
                xlab = 'Temps NC bur total'
            elif typ == 'web':
                end  = '% de temps NC web total'
                xlab = 'Temps NC web total'                                          
        elif ind == 'mnct':
            nbapp = listapp.median_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.median_noncomplianttime_users(bpcedata)            
            if typ == 'bur':
                end  = 'ms de temps NC bur médian'
                xlab = 'Temps NC bur médian'
            elif typ == 'web':
                end  = '% de temps NC web médian'
                xlab = 'Temps NC web médian'
        elif ind == 'maxnct':
            nbapp = listapp.max_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.max_noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur max'
                xlab = 'Temps NC bur max'
            elif typ == 'web':
                end  = '% de temps NC web max'
                xlab = 'Temps NC web max'   
        elif ind == 'mmnct':
            nbapp = listapp.mm_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.mm_noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur max-mean'
                xlab = 'Temps NC bur max-mean'
            elif typ == 'web':
                end  = '% de temps NC web max-mean'
                xlab = 'Temps NC web max-mean'   
        elif ind == 'maxminnct':
            nbapp = listapp.maxmin_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.maxmin_noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur max-min'
                xlab = 'Temps NC bur max-min'
            elif typ == 'web':
                end  = '% de temps NC web max-min'
                xlab = 'Temps NC web max-min'
        elif ind == 'madnct':
            nbapp = listapp.mad_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.mad_noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur mad'
                xlab = 'Temps NC bur mad'
            elif typ == 'web':
                end  = '% de temps NC web mad'
                xlab = 'Temps NC web mad'
        elif ind == 'meanadnct':
            nbapp = listapp.meanad_noncomplianttime_users(self)
            nbappMeteo = listappMeteo.meanad_noncomplianttime_users(bpcedata)
            if typ == 'bur':
                end  = 'ms de temps NC bur mean ad'
                xlab = 'Temps NC bur mean ad'
            elif typ == 'web':
                end  = '% de temps NC web mean ad'
                xlab = 'Temps NC web mean ad'
        else:
            print('{} unknown indicator'.format(ind))
        users   = nbapp != 0
        usersMeteo = nbappMeteo != 0
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        sns.set_style('ticks')
        sns.set(font_scale=1.6)
        sns.distplot(nbapp[users], kde= False,label = 
        'Hors METEO 2.0, méd. = {:.3} '.format(np.median(nbapp[users]))+end)
        sns.distplot(nbappMeteo[usersMeteo], kde =False, label = 
        'METEO 2.0, méd. = {:.3} '.format(np.median(nbappMeteo[usersMeteo]))+end)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Utilisateurs')
        plt.legend()
        plt.title('Comparaison_METEO_' + xlab + ' par utilisateur réseau ' + self.network)
        fig.savefig(xlab + self.network + '.png')
        return plt

    def extrapolate(self, ModelForBPCE):
        """
        extrapolate a given model to the new database
        returns AppLatency prediction
        """
        col = pd.read_csv(ModelForBPCE.cols)
        cols = col['cols']
        newdata = self.data[cols].values
        proba = ModelForBPCE.predict(newdata)
        return proba

    def extrapolate_satisfaction(self, ModelForBPCE):
        """
        extrapolate a given model to the new database
        returns AppLatency prediction
        """
        col = pd.read_csv(ModelForBPCE.cols)
        cols = col['cols']
        newdata = self.data[cols].values
        sat = ModelForBPCE.predict_satisfaction(newdata)
        return sat



class  App_List(object): 

    def __init__(self, bpcedata, typ):
        """
        initiator for App_List
        """
        self.typ             = typ
        self.appname         = self.get_app_list(bpcedata)
        self.extendedapplist = self.extend_app_list(bpcedata.network)
        self.len             = len(self.appname)
        self.network         = bpcedata.network

    def __repr__(self):
        """
        print metainfo
        """
        print("The names of the {} {} apps of {} network are: \n".format(self.len, self.typ, self.network))
        for app in self.appname:
            print("     - {} \n".format(app))
        return "Contains names of the {} {} apps of {} network".format(self.len, self.typ, self.network)

    def __str__(self):
        """
        print app list
        """
        return "The names of the {} {} apps of {} network are: \n".format(self.len, self.typ, self.network)
        #for app in self.appname:
        #    print("     - {} \n".format(app))

    def get_app_list(self, bpcedata):
        """
        method that lists all web app names
        """
        app_list = []
        if self.typ == 'bur':
            app_suffix = '_SLA_FrustratedTimeByStart'
        elif self.typ == 'web':
            app_suffix = '_AvgRT'
        else:
            print('Please provide bur or web')    
        for feat in bpcedata.features:
            if app_suffix in feat: app_list.append(feat[:-len(app_suffix)])
        print('{} network has {} {} apps'.format(bpcedata.network, 
            len(app_list),self.typ)) 
        return app_list


    def extend_app_list(self, network):
        """
        method that extends app list with Interact endings
        """
        if self.typ == 'bur':
            endings = ['_NbStart','_FreezeTime','_PeakPhysicalMemoryByStart',
            '_PeakPhysicalMemoryMax','_PhysicalMemoryByStart',
            '_PhysicalMemoryMax','_SLA_CompliancyTimeByStart',
            '_SLA_CompliancyTimeMax','_SLA_ToleratingTimeByStart',
            '_SLA_ToleratingTimeMax','_SLA_FrustratedTimeByStart',
            '_SLA_FrustratedTimeMax','_NbMemTicketsByStart',
            '_NbMemTicketsMax','_NbCPUTicketByStart','_NbCPUTicketMax']
        elif self.typ == 'web':
            if network == 'Equinoxe':
                endings = ['_AvgRT','_NbrOfTransactions','_SLACompliancyPercent',
                '_SLAToleratingPercent','_SLAFrustratedPercent','_NbrOfSystemErrorPercent',
                '_NbrOfOutOfRangePercent','_NbrOfTransactionsWithLatency','_AvgTrnWithLatencyRT']
            elif network == 'MySys':
                endings = ['_AvgRT','_NbrOfTransactions','_SLACompliancyPercent',
                '_SLAToleratingPercent','_SLAFrustratedPercent','_NbrOfSystemErrorPercent',
                '_NbrOfOutOfRangePercent','_NbrOfTransactionsWithLatency']                	
        extendedlist = []
        for app in self.appname:
            for end in endings:
                extendedlist.append(app + end)
        return extendedlist 
 

    def apps_user(self, bpcedata):
        """
        method that gives the list of apps for each users
        """
        count = []
        if self.typ == 'bur':
            ending = '_NbStart'
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt = []
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(app)
            count.append(cnt)
        return count        

    def nb_apps_users(self, bpcedata):
        """
        method that computes the numbers of apps per user"
        """
        count = []
        if self.typ == 'bur':
            ending = '_NbStart'
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt = 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt += 1
            count.append(cnt)
        return np.array(count)


    def nb_transac_users(self, bpcedata):
        """
        method that computes the numbers of apps per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = 0, 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt += rows[app + ending]
                    napp += 1
            if napp != 0:
                cnt = cnt
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_medtransac_users(self, bpcedata):
        """
        method that computes the numbers of apps per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = np.median(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_maxtransac_users(self, bpcedata):
        """
        method that computes the max of transactions per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_mmtransac_users(self, bpcedata):
        """
        method that computes the max of transactions per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))-np.mean(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_maxmintransac_users(self, bpcedata):
        """
        method that computes the max of transactions per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))-np.min(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_madtransac_users(self, bpcedata):
        """
        method that computes the mad of transactions per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = medad(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def nb_meanadtransac_users(self, bpcedata):
        """
        method that computes the mean ad of transactions per user"
        """
        count = []
        if self.typ == 'bur':
            print('No transactions for bur exe')
        else:
            ending = '_NbrOfTransactions'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    cnt.append(rows[app + ending])
                    napp += 1
            if napp != 0:
                cnt = meanad(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def noncomplianttime_users(self, bpcedata):
        """
        method that computes the total non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = np.sum(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)


    def median_noncomplianttime_users(self, bpcedata):
        """
        method that computes the median non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = np.median(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def max_noncomplianttime_users(self, bpcedata):
        """
        method that computes the non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def mm_noncomplianttime_users(self, bpcedata):
        """
        method that computes the difference between max and mean of non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))-np.mean(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def maxmin_noncomplianttime_users(self, bpcedata):
        """
        method that computes the difference between max and mean of non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = np.max(np.array(cnt))-np.min(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def mad_noncomplianttime_users(self, bpcedata):
        """
        method that computes the difference between max and mean of non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = medad(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)

    def meanad_noncomplianttime_users(self, bpcedata):
        """
        method that computes the difference between max and mean of non compliant time of apps per user"
        """
        count = []
        if self.typ == 'bur':
            end1 = '_SLA_ToleratingTimeByStart'
            end2 = '_SLA_FrustratedTimeByStart'
            ending =  '_SLA_CompliancyTimeByStart'           
        else:
            ending = '_SLACompliancyPercent'
        for index, rows in bpcedata.data.iterrows():
            cnt, napp = [], 0
            for app in self.appname:
                if rows[app+ending] > 0:
                    if self.typ == 'web':
                        cnt.append(100-rows[app + ending])
                    elif self.typ == 'bur':
                        cnt.append(rows[app + end1]+rows[app + end2])
                    napp += 1
            if napp != 0:
                cnt = meanad(np.array(cnt))
            else:
                cnt = 0
            count.append(cnt)
        return np.array(count)


class ModelForBPCE(object):

    def __init__(self, algo, ncv, nsamples, sca):
        """
        initiator for ModelForBPCE a quick
        class for objecting the metamodel
        """
        self.algo     = algo
        self.ncv      = ncv 
        self.nsamples = nsamples
        self.scaling  = sca
        self.ver      = self.version()

    def __repr__(self):
        """
        print metainfo
        """
        return "{} algorithm with {} hyperparameters values".format(self.algo,self.nsamples)

    def version(self):
        """
        stores current version of the model creation package
        """
        if self.algo == 'lr':
            from sklearn import __version__
            ver = __version__
        elif self.algo == 'lgbm':
            from lightgbm import __version__
            ver = __version__
        return ver

    
    def set_model(self, address):
        """
        once a final model has been found and written either as .pkl or .txt (for gbm)
        this function creates a link 
        """
        self.address = address
        return self

    def set_cols(self, address):
        """
        once a final model has been found and validated store the *.csv that contains
        cols names for extrapolation
        """
        self.cols = address
        return self

    def set_threshold(self, thr):
        """
        set the optimal threshold
        """
        self.threshold = thr
        return self

    def get_weights(self, bpcedata):
        """
        give importance plots for the given model
        returns a Pandas dataframe
        """
        if self.algo == 'lgbm':
            print('Load model to get weights')
            bst = lgb.Booster(model_file=self.address)
            weig = pd.DataFrame({'feat' : np.array(bpcedata.features)[bst.feature_importance() > 0],
            'weight' : bst.feature_importance()[bst.feature_importance() > 0]})
        elif self.algo == 'lr':
            print('Load model to get weights')
            clf = joblib.load(self.address)
            weiglr = pd.DataFrame({'feat' : np.array(bpcedata.features),
            'weight' : np.squeeze(clf.coef_)})
        return weiglr


    def predict(self, newdata):
        """
        predicts the probability for new data
        """
        if self.algo == 'lgbm':
            print('Load model to predict for the new data')
            import lightgbm as lgb
            bst = lgb.Booster(model_file=self.address)
            proba = bst.predict(newdata)
        elif self.algo == 'lr':
            print('Load model to predict for the new data')
            clf = joblib.load(self.address) 
            X = newdata
            if self.scaling == 'standard':
                sca = StandardScaler()
                X_sc = sca.fit_transform(X)
            pred  = clf.predict(X_sc)
            proba = clf.predict_proba(X_sc)[:,1]
        return proba       

    def predict_satisfaction(self, newdata):
        """
        predicts the probability for new data
        """
        if self.algo == 'lgbm':
            print('Load model to predict')
            import lightgbm as lgb
            bst = lgb.Booster(model_file=self.address)
            proba = bst.predict(newdata)
            thr = self.threshold
        elif self.algo == 'lr':
            print('Load model to get weights')
            clf = joblib.load(self.address) 
            X = newdata
            if self.scaling == 'standard':
                sca = StandardScaler()
                X_sc = sca.fit_transform(X)
            pred  = clf.predict(X_sc)
        return pred 
        

def baseline_model(X, y, ModelForBPCE):
    if ModelForBPCE.scaling == 'standard':
        sca = StandardScaler()
        X_sca = sca.fit_transform(X)
    elif ModelForBPCE.scaling == 'minmax':
    	sca = MinMaxScaler()
    	X_sca = sca.fit_transform(X)
    else:
        print('{} scaling should be implemented'.format(ModelForBPCE.scaling))
    
    if ModelForBPCE.algo == 'lr':
        clf_baseline = LogisticRegression()
        param_grid = { 'C' : np.logspace(-4, -1., ModelForBPCE.nsamples),
                   'solver' : ['newton-cg'],
                   'penalty' : ['l2']}
    elif ModelForBPCE.algo == 'svm':
        clf_baseline = SVC()  
        param_grid = { 'C' : np.logspace(2.8,3.2,ModelForBPCE.nsamples),
                   'gamma' : np.logspace(-4.2,-3.8,ModelForBPCE.nsamples)}  	
    else:
        print('{} algo should be implemented'.format(ModelForBPCE.algo))
    # Start training
    grid = GridSearchCV(clf_baseline, cv = ModelForBPCE.ncv,  param_grid= param_grid)  
    grid.fit(X_sca, y)
    return grid


def cv_auc(X, y, ModelForBPCE,params):
    """
    predicts satisfaction latency score over all basis through CV
    """
    rskf = RepeatedStratifiedKFold(n_splits = ModelForBPCE.ncv, n_repeats=5)
    y_tot = np.zeros_like(y)
    auc_cv = []
    i = 1
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        if ModelForBPCE.scaling == 'standard':
            sca = StandardScaler()
            X_sca = sca.fit_transform(X_train)
        elif ModelForBPCE.scaling == 'minmax':
            sca = MinMaxScaler()
            X_sca = sca.fit_transform(X_train)
        else:
            print('{} scaling should be implemented'.format(ModelForBPCE.scaling))
        if ModelForBPCE.algo == 'lr':
            clf = LogisticRegression(**params)
            clf.fit(X_sca, y_train)
            print('Training for {} K-fold out of {} done'.format(i,ModelForBPCE.ncv))
            X_t_sc = sca.fit_transform(X_test)
            y_pred = clf.predict_proba(X_t_sc)[:,1]
        elif ModelForBPCE.algo == 'lgbm':
            y_train_float = conv_YN_to_float(y_train)
            y_test_float = conv_YN_to_float(y_test)            
            lgb_train = lgb.Dataset(X_train, y_train_float, free_raw_data=False)
            lgb_eval = lgb.Dataset(X_test, y_test_float, reference=lgb_train, free_raw_data=False)            
            num_train, num_feature = X_train.shape
            feature_name = ['feature_' + str(col) for col in range(num_feature)]
            gbm = lgb.train(params,
                lgb_train,
                num_boost_round=80,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name)
            gbm.save_model('EQUINOXE_MODELS/model_LGBM_{}_{}' + '.txt'.format(i,ModelForBPCE.ncv))
            bst = lgb.Booster(model_file='EQUINOXE_MODELS/model_LGBM_{}_{}' + '.txt'.format(i,ModelForBPCE.ncv))
            y_pred = bst.predict(X_test)
            fpr, tpr, thresh = roc_curve(y_test_float, y_pred)
            auc_cv.append(auc(fpr, tpr))
            print('auc for {} K-fold is {}'.format(i,auc(fpr, tpr)))
        y_tot[test_index] = y_pred
        i += 1
    
    return auc_cv






def convert(arr):
    """
    conversion of scores into YES/NO
    """
    new_arr = np.copy(arr)
    for i in range(1,5):
        sat = 'Score' + str(i)
        ind = arr == sat
        if (i == 1) | (i == 2):
            new_arr[ind] = 'NO'
        else:
            new_arr[ind] = 'YES'
    return new_arr
    



def medad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def meanad(arr):
    """ Mean Absolute Deviation: another version of standard deviation. 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.mean(np.abs(arr - med))

def sat(arr, thr):
    """Translate probability into satisfaction
    """
    sat = np.zeros_like(arr)
    sat[arr < thr] = False
    sat[arr > thr] = True
    return sat

def plot_predicted_satisfaction(proba, ModelForBPCE):
    """plot satisfaction
    """
    satisfaction = sat(proba, ModelForBPCE.threshold)
    slices = [np.sum(satisfaction == True),np.sum(satisfaction == False)]
    activities = ['Satisfied','Not Satisfied']
    cols = ['#66CC99','#FF6666']
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(10,10)
    patches, texts, autotexts = ax1.pie(slices, explode=[0.05,0.], colors = cols, labels=activities, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    texts[0].set_fontsize(18)
    texts[1].set_fontsize(18)
    texts[1].set_fontweight('bold')
    autotexts[0].set_fontsize(18)
    autotexts[1].set_fontweight('bold')
    autotexts[1].set_fontsize(24)
    plt.title('Predicted distribution of satisfaction',fontsize=24)
    fig1.savefig(ModelForBPCE.address[:-4] + '.png',bbox_inches='tight')
    return plt

def conv_YN_to_float(pred):
    pred_f = np.copy(pred)
    IND_Y = pred == 'YES'
    IND_N = pred == 'NO'
    pred_f[IND_Y] = 1
    pred_f[IND_N] = 0
    return pred_f

def conv_TF_to_float(pred):
    pred_f = np.copy(pred)
    IND_Y = pred == True
    IND_N = pred == False
    pred_f[IND_Y] = 1
    pred_f[IND_N] = 0
    return pred_f
    















   

