
rm(list=ls())
cat('\014')

#-----------------------------------------------------------------------------------------------
#----------------------------------------- 1. MODULE -------------------------------------------

# Checker le path
source('S:/139. Richemont/1. Cupidon/Algorithmes/dedoublonnage/example.R/ultimate.R')

#-----------------------------------------------------------------------------------------------
#----------------------------------------- 2. DATA ---------------------------------------------

path = ''
data = read.csv(file = paste0(path, 'base_client.csv'), sep = ';', na.strings = '', stringsAsFactors = FALSE)
data = data.table(data)

#-----------------------------------------------------------------------------------------------
#----------------------------------------- 3. PARAMETERS ---------------------------------------

# Variables
global_id = 'Business.Partner'
last_name = 'Last.name'
first_name = 'First.name'
email_adress = 'E.Mail.Address'
phone_number = 'Telephone'
other = c('Street', 'City')  # Variables which will be used to compute the score
weights = c(10,5,3,1)  # Weigths used to compute the final score: should be in the same order as the variables !

# Parameters
loop_number = 1  # Implement a loop over data to prevent from overfitting RAM

#-----------------------------------------------------------------------------------------------
#----------------------------------------- 4. FUNCTIONS ---------------------------------------

bestpairs = deduplication(data, global_id, last_name, first_name, email_adress, phone_number, other, weights, loop_number, path)

pairs = pairs.identification(bestpairs[final_score > 0.6, .(ID, i.ID)])

pair1 = pairs[, .(ID1, pair_id)]
colnames(pair1) = c('ID', 'pair_id')
pair2 = pairs[, .(ID2, pair_id)]
colnames(pair2) = c('ID', 'pair_id')

pairs = rbind(pair1, pair2)

setkey(data, Business.Partner)
setkey(pairs, ID)

data = pairs[data]

