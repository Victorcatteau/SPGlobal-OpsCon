#!/usr/bin/env R
# -*- coding: utf-8 -*- 

"
--------------------------------------------------------------------
DEDUPLICATION TOOLS - CUPIDON PROJECT

Set all desired parameters before launching process !
--------------------------------------------------------------------
"

#-----------------------------------------------------------------------------------------------
#----------------------------------------- 1. PACKAGES -----------------------------------------


library(RecordLinkage)
library(stringdist)
library(readxl)
library(data.table)
library(dplyr)
library(devtools)
library(stats)


#-----------------------------------------------------------------------------------------------
#----------------------------------------- 2. DEDUPLICATION ------------------------------------


deduplication <- function(data, global_id, last_name, first_name, email_adress, phone_number, other, weights, loop_number, path)
{
  
  names = c(last_name, first_name)
  partial_var_list = c(email_adress, phone_number, other)
  all_var_list = c(last_name, first_name, partial_var_list)
  
  ####-----------------------------------------------------------------------------------
  ####-----------------------------------------------------------------------------------
  #### First case : same lastname + same firstname + score on other variables
  
  ##-------------------------
  ## DATA -------------------
  data = data[, c(global_id, all_var_list), with=FALSE]
  
  ##-------------------------
  ## FILTER -----------------
  
  eval(parse(text=paste0('data = data[(is.na(',first_name,') == FALSE) & (is.na(',last_name,') == FALSE)]')))
  
  row = nrow(data)
  
  data_reversed = data.table(data)
  colnames(data_reversed) = c(global_id, first_name, last_name, partial_var_list)
  
  big_data = rbind(data, data_reversed)
  big_data = big_data[, c(global_id, all_var_list), with=FALSE]
  
  for (var in partial_var_list)
  {
    eval(parse(text=paste0('big_data[(row+1):nrow(big_data), ',var,' := NA]')))
  }
  
  # If memory issue save 'data', delete it and load it after pair creation.
  rm(data_reversed)
  
  
  ##-------------------------
  ## CREATING POTENTIAL PAIRS
  
  time = Sys.time()
  
  # Here the condition is email OR phone OR (first name AND last name) identic
  # Since we have doubled the rows with different colnames there also is a reversed name similarity.
  
  potential_pairs = data.table(compare.dedup(big_data, blockfld = list(4,
                                                                       5,
                                                                       c(2, 3)))$pairs)[, .(id1, id2)]
  
  # potential_pairs = data.table(get.exact.pairs(data, blockfld = list(email_adress_colnumber,
  #                                                                    phone_number_colnumber,
  #                                                                    c(last_name_colnumber, first_name_colnumber))))
  
  colnames(potential_pairs) = c('V1', 'V2')
  
  rm(big_data)
  
  cat('\nCreating pairs time : ', round(difftime(Sys.time(),time,units = c('sec')), digits = 1), 'secs\n')
  
  ## ------------------------
  ## FILTERING DUPLICATE PAIRS
  
  # If both member of a pair have a rank > row that means that this pair exists twice (one in correct order, one in reverse order)
  potential_pairs = potential_pairs[!((V1 > row) & (V2 > row))]
  
  # So as to use only data (and not big data) for the later merge we transform rank to rank-row when rank > row
  potential_pairs[V1 > row, V1 := V1 - row]
  potential_pairs[V2 > row, V2 := V2 - row]
  
  # Cases where first name = last name can be weird
  potential_pairs = potential_pairs[V1 != V2]
  
  # Some same pairs exists
  potential_pairs[V1 > V2, `:=` (V1 = V2,
                                 V2 = V1)]
  potential_pairs = potential_pairs[duplicated(paste0(V1, V2)) == FALSE]
  
  
  ## ------------------------
  ## SAVING DATA-------------
  
  time2 = Sys.time()
  
  if (loop_number > 1)
  {
    step = floor(nrow(potential_pairs)/loop_number)
    steps = vector()
    steps[1] = 0
    for (i in 2:(loop_number))
    {
      steps[i] = step * (i-1) 
    }
    steps[loop_number + 1] = nrow(potential_pairs)
    
    for (i in 1:(length(steps)-1))
    {
      begining = steps[i]
      end = steps[i+1]
      eval(parse(text = paste0('PotentialPairs',i,' = potential_pairs[',begining + 1,':',end,']')))
      eval(parse(text = paste0('save(PotentialPairs',i,', file = paste0("',path,'","PotentialPairs',i,'.RData"))')))
      eval(parse(text = paste0('rm(PotentialPairs',i,')')))
    }
    
    rm(potential_pairs)
    
  } else {
    
    PotentialPairs1 = potential_pairs
    eval(parse(text = paste0('save(PotentialPairs1, file = paste0("',path,'","PotentialPairs1.RData"))')))
    
    rm(potential_pairs)
  }
  
  
  ## ------------------------
  ## PROCESSING DATA SOURCE--
  
  data[, `:=` (V1 = row.names(data),
               V2 = row.names(data))]
  
  for (var in all_var_list)
  {
    eval(parse(text = paste0('data[, get_',var,' := +!(is.na(',var,'))]')))
  }
  
  data_V1 = data[, c('V1', global_id, all_var_list, paste0(rep('get_', length(all_var_list)), all_var_list)), with=FALSE]
  
  data_V2 = data[, c('V2', global_id, all_var_list, paste0(rep('get_', length(all_var_list)), all_var_list)), with=FALSE]
  
  
  ## ------------------------
  ## MAIN LOOP --------------
  
  for (i in 1:loop_number)
  {
    time1 = Sys.time()
    
    eval(parse(text = paste0('load(paste0("',path,'","PotentialPairs',i,'.RData"))')))
    
    eval(parse(text = paste0('potential_pairs = PotentialPairs',i)))
    
    eval(parse(text = paste0('rm(PotentialPairs',i,')')))
    
    #--------------------------------------------------------------------------------------------
    #-------------------------------------- GET INFORMATION -------------------------------------
    
    time = Sys.time()
    
    potential_pairs[, `:=` (V1 = as.character(V1),
                            V2 = as.character(V2))]
    
    setkey(potential_pairs, V1)
    setkey(data_V1, V1)
    potential_pairs = data_V1[potential_pairs]
    
    setkey(potential_pairs, V2)
    setkey(data_V2, V2)
    potential_pairs = data_V2[potential_pairs]
    
    cat("\n\nGet information time : ", round(difftime(Sys.time(),time,units = c("sec")), digits = 2), "secs")
    
    #--------------------------------------------------------------------------------------------
    #-------------------------------------- FILTER PAIRS ----------------------------------------
    
    time = Sys.time()
    
    for (var in all_var_list)
    {
      eval(parse(text=paste0('potential_pairs[, Common_',var,' := get_',var,' * i.get_',var,']')))
    }
    
    eval(parse(text=paste0('potential_pairs[, `:=` (ID = ',global_id,',
                           i.ID = i.',global_id,')]')))
    
    eval(parse(text=paste0('potential_pairs[, `:=` (',global_id,' = NULL,
                           i.',global_id,' = NULL)]')))
    
    for (var in all_var_list)
    {
      eval(parse(text = paste0('potential_pairs[, `:=` (get_',var,' = NULL,
                               i.get_',var,' = NULL)]')))
    }
    
    # DELETE ALL PAIRS THAT MATCH ON LESS THAN 3 VARIABLES
    
    conditions = paste0(rep('Common_', length(all_var_list)), all_var_list)
    final_condition = paste(conditions, collapse = " + ")
    
    eval(parse(text=paste0('best_pairs = potential_pairs[',final_condition,' > 2]')))
    
    rm(potential_pairs)
    
    for (var in all_var_list)
    {
      eval(parse(text = paste0('best_pairs[, Common_',var,' := NULL]')))
    }
    
    best_pairs[, `:=` (V1 = NULL,
                       V2 = NULL)]
    
    cat("\nFiltering pairs time : ", round(difftime(Sys.time(),time,units = c("sec")), digits = 2), "secs")
    
    #--------------------------------------------------------------------------------------------
    #-------------------------------------- COMPUTE SCORE ---------------------------------------
    
    time = Sys.time()
    
    for (var in all_var_list)
    {
      eval(parse(text=paste0('best_pairs[, ',var,'_score := levenshteinSim(as.character(',var,'), as.character(i.',var,'))]')))
    }
    
    eval(parse(text=paste0('best_pairs[, reverse_name_score1 := levenshteinSim(',last_name,', i.',first_name,')]')))
    eval(parse(text=paste0('best_pairs[, reverse_name_score2 := levenshteinSim(',first_name,', i.',last_name,')]')))
    
    score_variables1 = paste0(partial_var_list, rep('_score', length(partial_var_list)))
    score_variables2 = paste0(names, rep('_score', length(names)))
    score_variables3 = c('reverse_name_score1', 'reverse_name_score2')
    
    best_pairs = best_pairs[,  c('ID', 'i.ID', 
                                 score_variables1, 
                                 score_variables2, 
                                 score_variables3), with=FALSE]
    
    # Construction of adapted 'vectorial mean' function
    input1 = paste0(score_variables1, collapse = ',')
    input2 = paste0('rep(',score_variables1,'[n],', weights, ')', collapse = ',')
    
    eval(parse(text=paste0('vectorial.mean <- function(',input1,')
                           {
                           nrow = length(',email_adress,'_score)
                           return_vector = vector(length = nrow)
                           
                           for (n in 1:nrow)
                           {
                           return_vector[n] = mean(c(',input2,'),
                           na.rm = TRUE)
                           }
                           return(return_vector)
                           }')))
  
    eval(parse(text=paste0('best_pairs[, partial_score := vectorial.mean(',input1,')]')))
    
    formula2 = paste(score_variables2, collapse = '+') 
    eval(parse(text=paste0('best_pairs[, name_score := (',formula2,')/2]')))
    
    best_pairs[, reverse_name_score := (reverse_name_score1 + reverse_name_score2)/2]
    
    ## ------------------------
    ## FINAL SCORE ------------
    
    # Filter on 'really bad' couples
    best_pairs = best_pairs[partial_score > 0.5 | name_score > 0.5 | reverse_name_score > 0.5]
    
    # Computing final score
    eval(parse(text=paste0('best_pairs[, final_score := ifelse((name_score == 1), partial_score,
                                                                ifelse((reverse_name_score == 1), partial_score, 
                                                                        ifelse(name_score > reverse_name_score, name_score, reverse_name_score)))]')))
    
    eval(parse(text=paste0('best_pairs[((name_score == 1) | (reverse_name_score == 1)) & ((',email_adress,'_score == 1) | (',phone_number,'_score == 1)) & (!is.na(',email_adress,'_score) | !is.na(',phone_number,'_score)), final_score := 1]')))
    
    cat("\nComputing score time : ", round(difftime(Sys.time(),time,units = c("sec")), digits = 1), "secs")
    
    #--------------------------------------------------------------------------------------------
    #-------------------------------------- SAVE DATA -------------------------------------------
    
    eval(parse(text = paste0('BestPairs',i,' = best_pairs')))
    
    eval(parse(text = paste0('save(BestPairs',i,', file = paste0("',path,'","BestPairs',i,'.RData"))')))
    
    eval(parse(text = paste0('rm(best_pairs,BestPairs',i,')')))
    
    cat("\n\n >> Step ", i, " / 10 : ", round(difftime(Sys.time(),time1,units = c("sec")), digits = 2), "secs\n")
    }
  
  cat("\n\n >> Total computing time : ", round(difftime(Sys.time(),time2,units = c("sec")), digits = 1), "secs\n")
  
  
  #-----------------------------------------------------------------------------------------------
  #-------------------------------------- SAVE ALL SCORES ----------------------------------------
  
  eval(parse(text=paste0('load(paste0("',path,'","BestPairs1.RData"))')))
  
  BestPairs = BestPairs1
  
  rm(BestPairs1)
  
  if (loop_number > 1)
  {
    for (i in 2:loop_number)
    {
      eval(parse(text = paste0('load(paste0("',path,'", "BestPairs',i,'.RData"))')))
      eval(parse(text = paste0('BestPairs = rbind(BestPairs,BestPairs',i,')')))
      eval(parse(text = paste0('rm(BestPairs',i,')')))
    }
  }
  
  # eval(parse(text=paste0('save(BestPairs, file = paste0("',path,'", "BestPairs_Full.RData"))')))
  
  return(BestPairs)
}



#-----------------------------------------------------------------------------------------------
#----------------------------------------- 3. PAIRS IDENTIFICATION -----------------------------



pairs.identification <- function(pairs)
{
  # pairs must be a datatable with two columns names "ID1" and "ID2"
  colnames(pairs) = c('ID1', 'ID2')
  
  is.good.pair <- function(id)
  {
    return(length(unique(pairs[ID1 == id | ID2 == id, pair_id])) > 1)
  }
  
  time = Sys.time()
  
  ##-------------------------------------
  ## First step: get first try of pair_id
  
  cat('\n>> Processing step 1\n')
  
  number = 1
  pairs[1, pair_id := number]
  ids = pairs[, ID1]
  
  for (i in 2:nrow(pairs))
  {
    subsample = pairs[1:(i-1)]
    id1 = pairs[i, ID1]
    id2 = pairs[i, ID2]
    subsample[, is.duplicate := +((ID1 == id1) | (ID1 == id2) | (ID2 == id1) | (ID2 == id2)) * pair_id]
    
    subsample = subsample[is.duplicate > 0]
    
    if (nrow(subsample) == 0)
    {
      number = number + 1
      pairs[i, pair_id := number]
    } else {
      pairs[i, pair_id := subsample[, min(is.duplicate)]]
    }
    
    if (i %% 500 == 0) {cat('\n', i, 'ID treated')}
  }
  
  ##------------------------------------------------------------------------------
  ## Second step: while there are pair issues reorder by pair_id and process again
  
  ids_unicity = sapply(ids, FUN = is.good.pair)
  k = 2
  
  cat('\n\n', sum(ids_unicity), 'issues left to solve\n')
  
  while (sum(ids_unicity) > 0)
  {
    
    cat('\n\nProcessing step', k, '\n')
    k = k + 1
    pairs = pairs[order(pair_id)]
    
    number = 1
    pairs[, pair_id := NULL]
    pairs[1, pair_id := number]
    
    for (i in 2:nrow(pairs))
    {
      subsample = pairs[1:(i-1)]
      id1 = pairs[i, ID1]
      id2 = pairs[i, ID2]
      subsample[, is.duplicate := +((ID1 == id1) | (ID1 == id2) | (ID2 == id1) | (ID2 == id2)) * pair_id]
      
      subsample = subsample[is.duplicate > 0]
      
      if (nrow(subsample) == 0)
      {
        number = number + 1
        pairs[i, pair_id := number]
      } else {
        pairs[i, pair_id := subsample[, min(is.duplicate)]]
      }
      
      if (i %% 500 == 0) {cat('\n', i, 'ID treated')}
    }
    
    ids_unicity = sapply(ids, FUN = is.good.pair)
    
    cat('\n\n', sum(ids_unicity), 'issues left to solve\n')
  }
  
  cat('\nPairs identification done in', round(difftime(Sys.time(), time, units = 'sec'), digits = 1), 'secs\n')
  
  return(pairs)
}


####-----------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------


##-------------------------
## DATA -------------------

# data = data[, c(global_id, first_name, last_name), with=FALSE]


##-------------------------
## CREATING POTENTIAL PAIRS

# time = Sys.time()
# 
# eval(parse(text=paste0('potential_pairs = data.table(',global_id,' = NA, i.',global_id,' = NA)')))
# 
# for (i in 1:nrow(data))
# {
#   eval(parse(text=paste0('id = data[i, ',global_id,']')))
#   eval(parse(text=paste0('first = data[i, ',first_name,']')))
#   eval(parse(text=paste0('last = data[i, ',last_name,']')))
#   
#   # For each loop start is row i so as to avoid redundance
#   eval(parse(text=paste0('data[i:nrow(data), is.name_reverse := ((',first_name,' == last) & (',last_name,' == first))]')))
#   
#   inter = data[is.name_reverse == TRUE]
#   
#   if (nrow(inter) > 0)
#   {
#     eval(parse(text=paste0('inter = data.table(',global_id,' = rep(',id,', nrow(inter)), i.',global_id,' = inter[, ',global_id,'])')))
#   
#     data[, is.name_reverse := NULL]
#     
#     potential_pairs = rbind(potential_pairs, inter)
#   }
#   
#   rm(inter)
#   
#   if (i %% 1000 == 0) {cat('\n', i, 'customers treated in', round(difftime(Sys.time(), time, units = c("min")), digits = 1), "mins")}
# }
# 
# potential_pairs = potential_pairs[2:nrow(potential_pairs)]

