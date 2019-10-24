rm(list=ls())
options(scipen=999)
setwd("~/Documents/R Directory/digital vote revised/")
# load useful packages 
library(scales)
library(stargazer)
library(foreign)
library(ranger)
library(foreach)
library(data.table)
library(parallel)
library(doParallel)
# This paper uses voter registration files registration files provided by L2. We cannot provide that dataset to the 
# public, but it is available for purchase by L2. 
# Wherever the L2 file was used directly for the creation of variables or dataframes which were in turn used to 
# create plots or tables , we opt to crate a new 'derived' dataset and release that instead. 
# In some instances, even this is not possible as some of our models use so many covariates that we would run into 
# privacy issues if we reported all the covariates. 

# Code to calculate the full results, including FIGURE 2 and other vote-analysis plots, is provided below. 
# The replicating researcher can check the turnout simulations make sense by using his own L2 dataset; 

# NOTE: the results will be slightly different from published due to simulation variance. 

# The sample v. pop plots are given with their own datasets, directly derived from the L2 Data and hence not shareable.  
# The Turnout scatterplot comparisons are also provided separately. 

# # # # ANONYMOUS SAMPLE
dv_match_sample = read.csv("matched_digital_sample_anonymous.csv")

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # Training the Forests # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# The following code is quite heavy and will take a long time to run due to simulations over many voter-categories.
# load digital traces 'Y_R' - these are 1 for reps and 0 for dems
Y_R =as.numeric( as.character(unlist(read.csv('Y_R.csv'))))
# load characteristics of the voters from the digital sample
Z_R = read.csv('Z_R.csv')

# load turnout predictions for digital sample
load('rf_T_FBobs_preds.RData')

# # # Calculate random forest model for vote choice conditional on turnout 
rf_model_R = 
  ranger(formula =  Y_R~.,
         data = Z_R,
         case.weights = apply(rf_T_FBobs$predictions[,2,],1,mean),
         num.trees = 1000,# set n.sims to 1000 to replicate
         write.forest = TRUE,
         probability = TRUE,
         keep.inbag = TRUE,
         #quantreg = TRUE,
         importance = 'permutation'
  )

# Predict vote choice for every member of the sample
rf_model_R_pred =predict(
  object = rf_model_R, 
  data = Z_R,
  predict.all = TRUE,
  num.trees = rf_model_R$num.trees, 
  type = "response",
  #se.method = "infjack",
  seed = NULL, 
  num.threads = detectCores(),
  verbose = TRUE)

# load cateogry-level turnout prediction simulations 
rf_T_CAT_predictions = read.csv(file = 'rf_T_CAT_predictions.csv') 

#rf_T_CAT_predictions = cbind(rf_T_CAT_predictions,rf_T_CAT_predictions[,1:250])
rf_T_CAT_predictions = data.frame(rf_T_CAT_predictions)

# enlarge category predictions to include wte
d_R_CAT =  foreach(e = 0:1,.combine = 'rbind') %:% 
  foreach(d = min(as.numeric(as.character(unlist(Z_R$wte)))):max(as.numeric(as.character(unlist(Z_R$wte)))),
          .combine = 'rbind') %do%  data.frame(el_type = e, wte = d, rf_T_CAT_predictions)

# use estimated regression tree to forecast prediction probabilties of vote choice per category
rf_R_CAT =predict(
  object = rf_model_R, 
  data = d_R_CAT,
  predict.all = TRUE,
  num.trees = rf_model_R$num.trees, 
  type = "response",
  #se.method = "infjack",
  seed = NULL, 
  num.threads = detectCores(),
  verbose = TRUE)

## MSPE1 implementation - calculating global RMSE
oob_residual = foreach(i = 1:length(Y_R),.combine = 'c') %do%
  # trees for which obs i is out of the bag
  (Y_R[i] - mean(rf_model_R_pred$predictions[i,2,
                                             which(unlist(lapply(lapply(rf_model_R$inbag.counts,
                                                                        FUN = function(x){which(x==0)}),
                                                                 function(y){i %in% y})))]))
oob_rmse = sqrt(mean(oob_residual^2))

# Simulate 500 runs of vote choice by category with above standard deviation
n_sims = 500
# simulate using parallel cores 
cl <- makeCluster(4)
registerDoParallel(cl)
clusterExport(cl = cl,varlist = c("rf_R_CAT","n_sims","oob_rmse"))
temp = t(parSapply(cl = cl, 
                   X = apply(rf_R_CAT$predictions[,2,],1,mean),
                   FUN = function(x){
                     rnorm(n = n_sims, mean = x, sd = oob_rmse)
                   })
)
stopCluster(cl)
# data.table simulations as it helps with calculations later 
temp = data.table(temp)
# rename to indicate these are republican vote choice sims
names(temp) = paste("R_sim_",1:dim(temp)[2],sep="")
# bind with turnout sims
d_R_CAT_sims = cbind(d_R_CAT, temp)
# load category counts 
C = read.csv('C.csv')
# merge counts with the sims dataset so we have target counts on the same row
WEEK_CATEGORY_DT = merge(C,d_R_CAT_sims,
                         by = c(#"US_Congressional_District",
                           "Voters_Gender",
                           "age_cat",                       
                           "EthnicGroups_EthnicGroup1Desc",
                           #"CommercialData_OccupationGroup",
                           "CommercialData_Education",
                           #"Religions_Description",         
                           #"income_cat_000",
                           #"CommercialData_HHComposition",
                           "Parties_Description"),#allow.cartesian = TRUE,
                         all = TRUE)

# the WEEK_CATEGORY_DT still has those few categories for which we predicted turnout, 
# but we could not predict vote choice because we don't have them in our sample. 
# As such we need to get rid of them, and ensure the size of 
# WEEK_CATEGORY_DT is exactly 7 times the size of C
length(which(is.na(WEEK_CATEGORY_DT$count)))
dim(WEEK_CATEGORY_DT )[1]/(7*2)==dim(C)[1]
#CHECK: 
sum(C$count)==sum(WEEK_CATEGORY_DT$count)/(7*2)
# all good.

# Extract point estimates for category comparison with surveys
R_sim_names = names(WEEK_CATEGORY_DT)[grep("R_sim",names(WEEK_CATEGORY_DT))]
T_sim_names = names(WEEK_CATEGORY_DT)[grep("V",names(WEEK_CATEGORY_DT))][-1]

WEEK_CATEGORY_DT_dt = as.data.frame(WEEK_CATEGORY_DT)
WEEK_CATEGORY_DT_dt$US_Congressional_District=as.numeric(as.character(unlist(WEEK_CATEGORY_DT_dt$US_Congressional_District)))

temp_T = apply(WEEK_CATEGORY_DT_dt[,grep("V",names(WEEK_CATEGORY_DT_dt))[-1]],1,FUN = function(x){mean(x)})
temp_R = apply(WEEK_CATEGORY_DT_dt[,grep("R",names(WEEK_CATEGORY_DT_dt))],1,FUN = function(x){mean(x)})

temp_cat = cbind(WEEK_CATEGORY_DT_dt[,1:9],T = temp_T,R = temp_R)

write.csv(temp_cat,file = "category_preds.csv",row.names = FALSE)

# over the simulations, count the vote chocies and turnout counts by category, 
# by mutiplying according to the usual decomposition
n_sims = 500

R_weighted_forecast = 
  lapply(1:n_sims,function(j){
    WEEK_CATEGORY_DT_dt[,R_sim_names[j]]*
      WEEK_CATEGORY_DT_dt[,T_sim_names[j]]*
      WEEK_CATEGORY_DT_dt[,"count"]})
# 500 sims by cat - R
R_weighted_forecast = as.data.frame(do.call(cbind.data.frame, R_weighted_forecast))

T_weighted_forecast = 
  lapply(1:n_sims,function(j){
    WEEK_CATEGORY_DT_dt[,T_sim_names[j]]*
      WEEK_CATEGORY_DT_dt[,"count"]})
# 500 sims by cat - T
T_weighted_forecast = as.data.frame(do.call(cbind.data.frame,T_weighted_forecast))


# # # Now crate aggregated datasets for CONGRESSIONAL DISTRICTS
# VOTE CHOCIE
R_f_counts_cd = data.frame()
for(d in min(WEEK_CATEGORY_DT_dt $wte):max(WEEK_CATEGORY_DT_dt $wte)){
  R_f_counts_cd_wte_temp = 
    lapply(1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),function(j){
      as.numeric(apply(R_weighted_forecast[which( WEEK_CATEGORY_DT_dt$el_type==0 & WEEK_CATEGORY_DT_dt$wte==d & WEEK_CATEGORY_DT_dt$US_Congressional_District==j),],2,sum)
      )})
  R_f_counts_cd_wte_temp = data.frame(cd = 1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),
                                      data.frame(wte = d, as.data.frame(do.call(rbind.data.frame,R_f_counts_cd_wte_temp)))
  )
  names(R_f_counts_cd_wte_temp)[-c(1,2)] = paste("R_sim_",1:n_sims,sep="")
  R_f_counts_cd = rbind(R_f_counts_cd,R_f_counts_cd_wte_temp)
}

# TURNOUT
counts_cd = data.frame()
for(d in min(WEEK_CATEGORY_DT_dt $wte):max(WEEK_CATEGORY_DT_dt $wte)){
  counts_cd_wte_temp = 
    lapply(1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),function(j){
      as.numeric(apply(T_weighted_forecast[which( WEEK_CATEGORY_DT_dt$el_type==0 & WEEK_CATEGORY_DT_dt$wte==d & WEEK_CATEGORY_DT_dt$US_Congressional_District==j),],2,sum)
      )})
  counts_cd_wte_temp = data.frame(cd = 1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),
                                  data.frame(wte = d, as.data.frame(do.call(rbind.data.frame,counts_cd_wte_temp)))
  )
  names(counts_cd_wte_temp)[-c(1,2)] = paste("T_sim_",1:n_sims,sep="")
  counts_cd = rbind(counts_cd,counts_cd_wte_temp)
}

# # # Now crate aggregated datasets for SENATE
# VOTE CHOCIE
R_f_counts_cd_senate = data.frame()
for(d in min(WEEK_CATEGORY_DT_dt $wte):max(WEEK_CATEGORY_DT_dt $wte)){
  R_f_counts_cd_senate_wte_temp = 
    lapply(1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),function(j){
      as.numeric(apply(R_weighted_forecast[which( WEEK_CATEGORY_DT_dt$el_type==1 & WEEK_CATEGORY_DT_dt$wte==d & WEEK_CATEGORY_DT_dt$US_Congressional_District==j),],2,sum)
      )})
  R_f_counts_cd_senate_wte_temp = data.frame(cd = 1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),
                                             data.frame(wte = d, as.data.frame(do.call(rbind.data.frame,R_f_counts_cd_senate_wte_temp)))
  )
  names(R_f_counts_cd_senate_wte_temp)[-c(1,2)] = paste("R_sim_",1:n_sims,sep="")
  R_f_counts_cd_senate = rbind(R_f_counts_cd_senate,R_f_counts_cd_senate_wte_temp)
}

# TURNOUT
counts_cd_senate = data.frame()
for(d in min(WEEK_CATEGORY_DT_dt $wte):max(WEEK_CATEGORY_DT_dt $wte)){
  counts_cd_senate_wte_temp = 
    lapply(1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),function(j){
      as.numeric(apply(T_weighted_forecast[which( WEEK_CATEGORY_DT_dt$el_type==1 & WEEK_CATEGORY_DT_dt$wte==d & WEEK_CATEGORY_DT_dt$US_Congressional_District==j),],2,sum)
      )})
  counts_cd_senate_wte_temp = data.frame(cd = 1:max(WEEK_CATEGORY_DT_dt$US_Congressional_District),
                                         data.frame(wte = d, as.data.frame(do.call(rbind.data.frame,counts_cd_senate_wte_temp)))
  )
  names(counts_cd_senate_wte_temp)[-c(1,2)] = paste("T_sim_",1:n_sims,sep="")
  counts_cd_senate = rbind(counts_cd_senate,counts_cd_senate_wte_temp)
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# PROBABILITY OF VOTING REPUBLICAN
CONGRESS_cd = merge(R_f_counts_cd,counts_cd,by = c("wte","cd"), all = TRUE)

CONGRESS_cd_P = cbind(wte = CONGRESS_cd$wte, 
                      cd = CONGRESS_cd$cd,
                      CONGRESS_cd[,grep("R_sim",names(CONGRESS_cd))]/CONGRESS_cd[,grep("T_sim",names(CONGRESS_cd))]
)
CONGRESS_cd_P = cbind(CONGRESS_cd_P,
                      prob_win = rowMeans(apply(CONGRESS_cd_P[,grep("R_sim",names(CONGRESS_cd_P))],2,function(x){ifelse(x>0.5,1,0)}))
)
CONGRESS_cd_P_quant = cbind(CONGRESS_cd_P[,c("wte","cd","prob_win")],
                            t(apply(CONGRESS_cd_P[,grep("R_sim",names(CONGRESS_cd_P))],1,function(x){quantile(x,c(0.1,0.5,0.9))}))
)

# now bring in 538 data and merge with our results
f38_forecast_house = read.csv("house_district_forecast.csv")
f38_forecast_house = f38_forecast_house[f38_forecast_house$state=="TX" & f38_forecast_house$model =="lite",-which(names(f38_forecast_house)=="state"|names(f38_forecast_house)=="model")]
f38_forecast_house$wte = as.numeric(
  floor(
    difftime(strptime("2018-11-06", format = "%Y-%m-%d"),
             strptime(f38_forecast_house$forecastdate,format = "%Y-%m-%d"),
             units="weeks")
  ))
f38_forecast_house = f38_forecast_house[which(f38_forecast_house$wte<=6),]
# choose central 2p sum 
f38_forecast_house$sum2p_central = NA
for (w in 6:0){ for(d in 36:1){
  f38_forecast_house[f38_forecast_house$district==d & f38_forecast_house$wte ==w,"sum2p_central"] = 
    
    ifelse(length(f38_forecast_house[f38_forecast_house$district==d & f38_forecast_house$wte ==w & f38_forecast_house$party=="R","voteshare"])==0,
           0,f38_forecast_house[f38_forecast_house$district==d & f38_forecast_house$wte ==w & f38_forecast_house$party=="R","voteshare"])+
    ifelse(length(f38_forecast_house[f38_forecast_house$district==d & f38_forecast_house$wte ==w & f38_forecast_house$party=="D","voteshare"])==0,
           0,f38_forecast_house[f38_forecast_house$district==d & f38_forecast_house$wte ==w & f38_forecast_house$party=="D","voteshare"])
}  }

f38_forecast_house = f38_forecast_house[f38_forecast_house$party=="R",]

f38_forecast_house_weekly = 
  aggregate(data.frame(
    f38_central = f38_forecast_house$voteshare/f38_forecast_house$sum2p_central,
    f38_10 = f38_forecast_house$p10_voteshare/f38_forecast_house$sum2p_central,
    f38_90 = f38_forecast_house$p90_voteshare/f38_forecast_house$sum2p_central,
    f38_PRWIN = f38_forecast_house$win_probability
  ),
  by = list(
    cd = f38_forecast_house$district,
    wte = f38_forecast_house$wte
  ),
  FUN = "mean")


CONGRESS_cd_P_f38 = merge(f38_forecast_house_weekly,CONGRESS_cd_P_quant ,by = c("wte","cd"),all=TRUE)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # ## # # ## # # # # #
# repeat process for Senate, and stack up 
SENATE_cd = merge(R_f_counts_cd_senate,counts_cd_senate,by = c("wte","cd"), all = TRUE)

SENATE = 
  aggregate(data.frame(SENATE_cd[,c(grep("R_sim",names(SENATE_cd)),grep("T_sim",names(SENATE_cd)))]),
            by = list(wte = SENATE_cd$wte),
            FUN = sum)
SENATE_P = cbind(wte = SENATE$wte, 
                 SENATE[,grep("R_sim",names(SENATE))]/SENATE[,grep("T_sim",names(SENATE))]
)
SENATE_P = cbind(SENATE_P,
                 prob_win = rowMeans(apply(SENATE_P[,grep("R_sim",names(SENATE_P))],2,function(x){ifelse(x>0.5,1,0)}))
)

SENATE_P_quant = cbind(SENATE_P[,c("wte","prob_win")],
                       t(apply(SENATE_P[,grep("R_sim",names(SENATE_P))],1,function(x){quantile(x,c(0.1,0.5,0.9))}))
)


f38_forecast_senate = read.csv("senate_seat_forecast.csv")
f38_forecast_senate = f38_forecast_senate[f38_forecast_senate$state=="TX" & f38_forecast_senate$model =="lite",-which(names(f38_forecast_senate)=="state"|names(f38_forecast_senate)=="model")]
f38_forecast_senate$wte = as.numeric(
  floor(
    difftime(strptime("2018-11-06", format = "%Y-%m-%d"),
             strptime(f38_forecast_senate$forecastdate,format = "%Y-%m-%d"),
             units="weeks")
  ))
f38_forecast_senate = f38_forecast_senate[which(f38_forecast_senate$wte<=6),]
# choose central 2p sum 
f38_forecast_senate$sum2p_central = NA
for (w in 6:0){ 
  f38_forecast_senate[f38_forecast_senate$wte ==w,"sum2p_central"] = 
    
    ifelse(length(f38_forecast_senate[f38_forecast_senate$wte ==w & f38_forecast_senate$party=="R","voteshare"])==0,
           0,f38_forecast_senate[f38_forecast_senate$wte ==w & f38_forecast_senate$party=="R","voteshare"])+
    ifelse(length(f38_forecast_senate[f38_forecast_senate$wte ==w & f38_forecast_senate$party=="D","voteshare"])==0,
           0,f38_forecast_senate[f38_forecast_senate$wte ==w & f38_forecast_senate$party=="D","voteshare"])
} 

f38_forecast_senate = f38_forecast_senate[f38_forecast_senate$party=="R",]

f38_forecast_senate_weekly = 
  aggregate(data.frame(
    f38_central = f38_forecast_senate$voteshare/f38_forecast_senate$sum2p_central,
    f38_10 = f38_forecast_senate$p10_voteshare/f38_forecast_senate$sum2p_central,
    f38_90 = f38_forecast_senate$p90_voteshare/f38_forecast_senate$sum2p_central,
    f38_PRWIN = f38_forecast_senate$win_probability
  ),
  by = list(
    wte = f38_forecast_senate$wte
  ),
  FUN = "mean")

SENATE_P_f38 = merge(f38_forecast_senate_weekly,SENATE_P_quant ,by = c("wte"),all=TRUE)
SENATE_cd_P_f38 = data.frame(SENATE_P_f38,cd = 0)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# clean names and stack up senate and congress
library(dplyr)
names(CONGRESS_cd_P_f38) =  gsub("\\%","",names(CONGRESS_cd_P_f38))
names(SENATE_cd_P_f38) = gsub('^\\.|\\.$', '',gsub("X","",names(SENATE_cd_P_f38)))
RESULTS_f38 = bind_rows(CONGRESS_cd_P_f38,SENATE_cd_P_f38)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# finally, bring in true results
elday_results = read.csv("Texas_Results.csv")
elday_results$PERCENT = as.numeric(gsub("\\%","",elday_results $PERCENT))
elday_results$PERCENT = elday_results$PERCENT /100
elday_results$sum2p_res = NA
for(d in 0:36){
  elday_results[elday_results$cd==d , "sum2p_res"] = 
    ifelse(length(elday_results[elday_results$cd==d & elday_results$PARTY=="REP","PERCENT"])==0,0,
           elday_results[elday_results$cd==d & elday_results$PARTY=="REP","PERCENT"]) + 
    ifelse(length(elday_results[elday_results$cd==d & elday_results$PARTY=="DEM","PERCENT"])==0,0,
           elday_results[elday_results$cd==d & elday_results$PARTY=="DEM","PERCENT"])
}
elday_results = elday_results[elday_results$PARTY=="REP",c("cd","PERCENT","sum2p_res")]
elday_results$res = elday_results$PERCENT/elday_results $sum2p_res
elday_results = elday_results[,c("cd","res")]

RESULTS_f38_elday = merge(elday_results, RESULTS_f38,by = "cd",all=TRUE)
RESULTS_f38_elday = RESULTS_f38_elday[order(RESULTS_f38_elday $cd,RESULTS_f38_elday $wte),]

write.csv(RESULTS_f38_elday,"RESULTS.csv")


# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # week 0 results # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# In this section we create the comparison plot between the true results, our estimates, 538 and the 2016 Trump Vote
# upload district-day results from 538 
XO = read.csv("house_district_forecast.csv")

# only look at texas
XO = XO[XO$state=="TX",]

# only look at 'lite' model - which only uses polls as an input
XO = XO[XO$model=="lite",]

# average over weeks to have comparable estimates to ours
XO = 
  aggregate(data.frame(vote = XO$voteshare/100),
            by = list(weeks = floor(as.numeric(difftime(as.Date("6-11-2018","%d-%m-%Y"),as.Date(XO$forecastdate,"%Y-%m-%d"),units = 'days'))/7),
                      district = XO$district,
                      party = XO$party),
            FUN = function(x){mean(x,na.rm=TRUE)})

# want to calculate two party vote - expand dt first and then take rowsums to find two party sum
XO_temp = reshape(XO,idvar = c("district","weeks"),timevar ="party",direction = "wide")
XO_temp$pv2 = rowSums(cbind(XO_temp$vote.D,XO_temp$vote.R))
XO_temp$R2p = XO_temp$vote.R/ XO_temp$pv2 

# This conlcudes our treatment of fivethirtyeight data
# Now upload our own results 
temp = read.csv("RESULTS.csv")
# merge with the fivethirtyeight data 
# 538 data obtained here: https://projects.fivethirtyeight.com/2018-midterm-election-forecast/house/?ex_cid=midterms-header
SXO = merge(temp,XO_temp,by.x=c("cd","wte"),by.y = c("district","weeks"),all=TRUE)
# now load histtorical results by district
# these are obtained here: https://www.dailykos.com/stories/2018/2/21/1742660/-The-ultimate-Daily-Kos-Elections-guide-to-all-of-our-data-sets
temp_hist = read.csv("pres_vote_cd_dt.csv",skip = 1)
# again focus only on Texas 
temp_hist_TX = temp_hist[which(substr(temp_hist$CD,1,2)=="TX"),]
# clean district names to merge
temp_hist_TX$CD = as.numeric(substr(temp_hist_TX$CD,4,5))
# merge with ours and 538s data
TEMP = merge(temp, temp_hist_TX,by.x = "cd",by.y = "CD",all = TRUE)
# derive trump vote from two-party vote 
TEMP$TRUMP = TEMP$Trump/rowSums(cbind(TEMP$Trump,TEMP$Clinton))
# look only at week zero 
week_zero_res = TEMP[which(TEMP$wte==0),]
# remove predictions for two-party vote where is there are not two party running
week_zero_res = week_zero_res[which(!is.na(week_zero_res$f38_central)),]
# the state vote for trump is input by hand (trump/(trump + clinton)) in texas
week_zero_res[week_zero_res$cd==0,"TRUMP"]= 0.5223/(0.5223+0.4324)

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # Information test - does our forecast add value ? # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

delta = week_zero_res$res-week_zero_res$TRUMP
delta_f38 = week_zero_res$f38_central-week_zero_res$TRUMP
delta_dv = week_zero_res$X50-week_zero_res$TRUMP

mean(abs(delta - rowMeans(cbind(delta_f38,delta_dv))))
mean(abs(delta - rowMeans(cbind(delta_dv ))))
mean(abs(delta - rowMeans(cbind(delta_f38 ))))

output1 = lm(week_zero_res$res~week_zero_res$f38_central+ week_zero_res$X50)
output2 = lm(week_zero_res$res~week_zero_res$X50 + week_zero_res$TRUMP)
output3 = lm(delta~delta_f38 + delta_dv)

# TABLE APPENDIX D.1
stargazer(output1, output2,output3, type = "latex")
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # Mean Absolute Error Bar Plot # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# FIGURE 3
pdf(file=paste("mae_nores.pdf",sep=""),width = 8.50, height = 5.00)
plot(
  1:dim(week_zero_res)[1],
  rev(sort(abs(week_zero_res$res-week_zero_res$X50))),
  bty = "n",
  ylab = "absolute error",
  xlab = "",
  xaxt = "n",pch = NA,main = "",#Absolute Errors of Point Estimates",
  ylim = c(0,0.2)
)
segments(x0 = 1:dim(week_zero_res)[1]-0.1, 
         x1 = 1:dim(week_zero_res)[1]-0.1,
         y0 = rep(0,dim(week_zero_res)[1]),
         y1 = rev(sort(abs(week_zero_res$res-week_zero_res$X50))),
         lwd = 1.5,
         col = "black",lty = 1
)
segments(x0 = 1:dim(week_zero_res)[1]+0.15, 
         x1 = 1:dim(week_zero_res)[1]+0.15,
         y0 = rep(0,dim(week_zero_res)[1]),
         y1 = abs(week_zero_res$res-week_zero_res$f38_central)[rev(order(abs(week_zero_res$res-week_zero_res$X50)))],
         lwd = 1.5,
         col = "orange",lty = 1
)
segments(x0 = 1:dim(week_zero_res)[1]+0.3, 
         x1 = 1:dim(week_zero_res)[1]+0.3,
         y0 = rep(0,dim(week_zero_res)[1]),
         y1 = abs(week_zero_res$res-week_zero_res$TRUMP)[rev(order(abs(week_zero_res$res-week_zero_res$X50)))],
         lwd = 1.5,
         col = "purple",lty = 1
)
axis(side = 1,
     at = 1:dim(week_zero_res)[1],
     ifelse(week_zero_res$cd[rev(order(abs(week_zero_res$res-week_zero_res$X50)))]==0,"S",
            paste("CD-",week_zero_res$cd[rev(order(abs(week_zero_res$res-week_zero_res$X50)))],sep="")),
     las = 2,cex.axis = 0.75
)

abline(h = mean(abs(week_zero_res$res-week_zero_res$f38_central)),lty = 2, col = "orange",lwd = 2)
abline(h =  mean(abs(week_zero_res$res-week_zero_res$X50)),lty = 2,col = "black",lwd = 2)
abline(h =  mean(abs(week_zero_res$res-week_zero_res$TRUMP)),lty = 5,col = "purple",lwd = 2)

legend("topright",
       legend = c("Digital Vote","FiveThirtyEight","2016 Trump Vote","Mean Absolute Error"),
       lty = c(1,1,1,2),
       col = c("black","orange","purple","black"),#c(alpha("red",alpha = 0.5),alpha("blue",alpha = 0.5)),
       lwd = 1,
       bty="n")
dev.off()
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # Mean Absolute Error - ordered to show attenuation bias # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# FIGURE 4
pdf(file=paste("attenuation_nores.pdf",sep=""),width = 6.50, height = 6.50)
par(oma = c(0,0,0,0))
plot(week_zero_res$res,
     abs(week_zero_res$res-week_zero_res$f38_central),
     pch = NA,ylab = "absolute error",xlab = "R two-party vote share",bty = "n",xlim = c(0,1),main = "",ylim = c(0,0.2))

lines(week_zero_res$res[order(week_zero_res$res)],
      abs(week_zero_res$res-week_zero_res$f38_central)[order(week_zero_res$res)],
      col = alpha("orange",alpha = 0.25),lty = 1)
lines(week_zero_res$res[order(week_zero_res$res)],
      loess(abs(week_zero_res$res-week_zero_res$f38_central)[order(week_zero_res$res)]~
              week_zero_res$res[order(week_zero_res$res)])$fitted,
      col = "orange",lty = 1,lwd = 2)
abline(h = mean(abs(week_zero_res$res-week_zero_res$f38_central)[order(week_zero_res$res)]),
       lty =2,col = "orange",lwd = 2)


lines(week_zero_res$res[order(week_zero_res$res)],
      abs(week_zero_res$res-week_zero_res$TRUMP)[order(week_zero_res$res)],
      col = alpha("purple",alpha = 0.25),lty = 1)
lines(week_zero_res$res[order(week_zero_res$res)],
      loess(abs(week_zero_res$res-week_zero_res$TRUMP)[order(week_zero_res$res)]~
              week_zero_res$res[order(week_zero_res$res)])$fitted,
      col = "purple",lty = 1,lwd = 2)
abline(h = mean(abs(week_zero_res$res-week_zero_res$TRUMP)[order(week_zero_res$res)]),
       lty =2,col = "purple",lwd = 2)

lines(week_zero_res$res[order(week_zero_res$res)],
      abs(week_zero_res$res-week_zero_res$X50)[order(week_zero_res$res)],
      col = alpha("black",alpha = 0.25),lty = 1)
lines(week_zero_res$res[order(week_zero_res$res)],
      loess(abs(week_zero_res$res-week_zero_res$X50)[order(week_zero_res$res)]~
              week_zero_res$res[order(week_zero_res$res)])$fitted,
      col = "black",lty = 1,lwd = 2)
abline(h = mean(abs(week_zero_res$res-week_zero_res$X50)[order(week_zero_res$res)]),
       lty =2,col = "black",lwd = 2)

legend("topright",
       legend = c("Digital Vote","FiveThirtyEight","2016 Trump Vote","Absolute Error","Smoothed AE","Mean Absolute Error"),
       lty = c(1,1,1,1,1,2),
       col = c("black","orange","purple",alpha("black",alpha = 0.25),"black","black"),#c(alpha("red",alpha = 0.25),alpha("blue",alpha = 0.25)),
       lwd = 1,
       cex = 1,
       bty="n")
text(x =0.065+ 0,y = 0.005+
       c(mean(abs(week_zero_res$res-week_zero_res$f38_central)[order(week_zero_res$res)]),
         mean(abs(week_zero_res$res-week_zero_res$X50)[order(week_zero_res$res)]),
         mean(abs(week_zero_res$res-week_zero_res$TRUMP)[order(week_zero_res$res)])
         ),
     labels = c(
       paste("538 MAE:",round(mean(abs(week_zero_res$res-week_zero_res$f38_central)[order(week_zero_res$res)]),3)),
       paste("D.V. MAE:",round(mean(abs(week_zero_res$res-week_zero_res$X50)[order(week_zero_res$res)]),3)),
       paste("Trump MAE:",round(mean(abs(week_zero_res$res-week_zero_res$TRUMP)[order(week_zero_res$res)]),3))
       ),
     cex = 0.65)
dev.off()
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # estimates comparison plot # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
mean(
TEMP$res[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]-
TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],na.rm=TRUE)

mean(abs(
  TEMP$res[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]-
    TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]),na.rm=TRUE)



# FIGURE 2 
pdf(file=paste("forecast_coefplot_wte0_nores.pdf",sep=""),width = 8.50, height = 5.00)
par(xpd = TRUE, oma = c(0,0,0,0),mar = c(2,5,3,2))
plot( TEMP$cd[TEMP$wte==0],
      TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
      xlab = '',#'days to election',
      main = "",# paste("Election Week TEMP"),
      xaxt = "n",cex.lab = 0.9,
      yaxt = "n",
      ylab = "% R" , 
      pch = 16, 
      ylim = c(0.15,0.85),
      cex.main = 0.9,
      cex = 0.85,
      bty='n',
      col = ifelse(TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]>0.5,"red","blue")
)
segments(x0 = TEMP$cd[TEMP$wte==0],
         y0 = TEMP$X10[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
         x1 = TEMP$cd[TEMP$wte==0],
         y1 = TEMP$X90[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
         col = ifelse(TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]>0.5,"red","blue"),
         lty = 1)
par(new=TRUE)
plot( TEMP$cd[TEMP$wte==0],
      TEMP$f38_central[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
      xlab = '',#'days to election',
      main = "",
      yaxt = "n",
      xaxt = "n",
      ylab = "" ,
      pch = 17,
      col = alpha(ifelse(TEMP$f38_central[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]>0.5,"lightcoral","skyblue"),1),
      bty = "n",
      cex = 0.95,
      ylim = c(0.15,0.85))
segments(x0 = TEMP$cd[TEMP$wte==0],
         y0 = TEMP$f38_10[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
         x1 = TEMP$cd[TEMP$wte==0],
         y1 = TEMP$f38_90[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
         col = alpha(ifelse(TEMP$f38_central[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]>0.5,"lightcoral","skyblue"),1),lty = 1)

par(new=TRUE)
plot( TEMP$cd[TEMP$wte==0],
      TEMP$TRUMP[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
      xlab = '',#'days to election',
      main = "",
      yaxt = "n",
      xaxt = "n",
      ylab = "" ,
      pch = 4,
      col = alpha(ifelse(TEMP$TRUMP[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]>0.5,"lightcoral","skyblue"),1),
      bty = "n",
      cex = 0.95,
      ylim = c(0.15,0.85))

par(new=TRUE)
plot( TEMP$cd[TEMP$wte==0],
      TEMP$res[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
      xlab = '',#'days to election',
      main = "",
      yaxt = "n",
      xaxt = "n",
      ylab = "" ,
      pch = 15,
      col = "darkgreen",
      bty = "n",
      cex = 0.95,
      ylim = c(0.15,0.85))
abline(h = 0.5,col = 'black',lty = 2,lwd = 1,xpd = FALSE)
axis(side = 2,
     at =seq(from = 0.15,to = 0.85,by = 0.1),
     labels = seq(from = 0.15,to = 0.85,by = 0.1),
     cex.axis = 0.95,las = 2
)
text(x = TEMP$cd[TEMP$wte==0],
     y = 0.075 + 
       apply(
         data.frame(
           TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],
           TEMP$f38_central[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]
         ),1,function(x){max(x,na.rm=TRUE)}),
     labels =
       ifelse(
         TEMP$cd[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])]==0,"Senate",
         paste("CD",TEMP$cd[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])])),
     cex = 0.8,
     srt = 90)

legend("bottomright",#x = 0, y = 0.1,
       legend = c("Election-Day Result","Estimated Democrat Win","Estimated Republican Win","Digital Vote 90% Pred. Interval","FiveThirtyEight 80% Pred. Interval","Trump 2016 Vote Share"),
       col = c("darkgreen","blue","red","black","darkgrey","darkgrey"),
       pch = c(15,15,15,16,17,4),
       lty = c(0,0,0,1,1,NA),
       bty="n",
       xpd = NA,
       cex = 0.9,
       horiz = FALSE
)
dev.off()

# FIGURE C1 - Appendix
pdf(file=paste("forecast_wte_districts_congress_nores.pdf",sep=""),width = 12, height = 9.5)
par(xpd = TRUE, oma = c(0,0,0,0),mar = c(2,1,2,1),mfrow = c(6,6))
for(i in 1:max(TEMP$cd)){
plot( TEMP$wte[TEMP$cd==i],
      TEMP$X50[TEMP$cd==i],
      xlab = '',#'days to election',
      ylab = "% R" , 
      pch = NA, 
      ylim = c(0.15,0.85),
      cex = 0.95,
      main = paste("district:",i),
      bty='n',xlim = c(6,0)
)
lines(TEMP$wte[TEMP$cd==i],
      TEMP$X50[TEMP$cd==i],col = 'black',lwd = 2)
lines(TEMP$wte[TEMP$cd==i],
      TEMP$f38_central[TEMP$cd==i],col = 'orange',lwd = 2)
lines(TEMP$wte[TEMP$cd==i],
      TEMP$res[TEMP$cd==i],col = 'darkgreen',lwd = 2)
par(new=FALSE)
}
dev.off()

mean(
TEMP$X50[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])] - 
TEMP$res[TEMP$wte==0][order(TEMP$X50[TEMP$wte==0])],na.rm=TRUE
)
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # category comparison with polls  # # # ## # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# upload survey data 
survey <- read.dta("Texas_survey_201810.dta")
survey_june= read.spss("Texas_survey_201806.sav", to.data.frame=TRUE)
survey_feb <- read.dta("Texas_survey_201802.dta")

# selct common variables in the surveys
# use vote frequency for turnout - closest to measure from registry
survey_oct = survey[,c("Q16","gender","race","birthyr","PID3_utex","educ","Q3A","Q3C","endtime")]
survey_june = survey_june[,c("Q16","gender","race","birthyr","PID3_utex","educ","Q3","Q3A","endtime")]
survey_feb = survey_feb[,c("Q15","gender","race","birthyr","PID3_utex","educ","Q3","Q3A","endtime")]
# harmonize names - use only june and october
names(survey_june) = names(survey_oct)
survey_sub = rbind(survey_oct,survey_june)
# upload category perdictions

category_preds = read.csv("category_preds.csv")
# look at senate predictions (state wide) only, as this is what is available on the surveys
temp = unique(category_preds[category_preds$el_type==1,-which(names(category_preds)=="US_Congressional_District"|
                                                                names(category_preds)=="count"|
                                                                names(category_preds)=="el_type")])
# order by week 
temp = temp[order(temp$wte),]
# harmonize party id from surveys to match cat preds
survey_sub$PID3_utex = 
  gsub("Democrat","Democratic",
       gsub("Other",NA,gsub("Not sure",NA,
                            gsub("Independent","Non-Partisan",survey_sub$PID3_utex)))
  )
# harmonize gender from surveys to match cat preds
survey_sub$gender = gsub("Female","F",gsub("Male","M",survey_sub$gender))
# harmonize age from surveys to match cat preds
survey_sub$age = cut(x = (2018 - survey_sub$birthyr),breaks = c(17,24,34,44,54,64,max((2018 - survey_sub$birthyr))),labels = c("18-24","25-34","35-44","45-54","55-64","65_or_older"))
# harmonize ethnicity from surveys to match cat preds
survey_sub$race = 
  gsub("Native American","Other",
       gsub("Mixed","Other",
            gsub("Middle Eastern","Other",
                 gsub("Hispanic","Hispanic and Portuguese",
                      gsub("Asian","East and South Asian",
                           gsub("Black","Likely African-American",
                                gsub("White / Blanco","European",survey_sub$race)))))))

# harmonize education from surveys to match cat preds
survey_sub$educ = 
  gsub("4-year","Bach Degree",
       gsub("2-year","Some College",
            gsub("Some college","Some College",
                 gsub("No HS","Less than HS Diploma",
                      gsub("High school graduate","HS Diploma",
                           gsub("Post-grad","Grad Degree",survey_sub$educ))))))
# no vocational training in survey - do not compare 

# transform days of survey into weeks 
# days of the survey: mostly 3rd week before the election, some 4th.
difftime(as.Date("2018/11/06","%Y/%m/%d"),as.Date("2018/10/15","%Y/%m/%d"))/7
difftime(as.Date("2018/11/06","%Y/%m/%d"),as.Date("2018/10/21","%Y/%m/%d"))/7
# for june: mostly 22nd week, a bit of 21st
difftime(as.Date("2018/11/06","%Y/%m/%d"),as.Date("2018/06/08","%Y/%m/%d"))/7
difftime(as.Date("2018/11/06","%Y/%m/%d"),as.Date("2018/06/17","%Y/%m/%d"))/7

# compare to third week estimates from category preds and surveys - area of most overlap
temp_3w_nopart_agg = temp[temp$wte=="2",-which(names(temp)=="wte")]
# Turnout variable - voting history (most similar to registration file, hence closest comparison to preds)
survey_sub$voting_history_prob = 
  as.numeric(
    gsub("Don't know",NA,
         gsub("None",0,
              gsub("One or two",0.25,
                   gsub("About half",0.5,
                        gsub("Almost every election",0.75,
                             gsub("Every election",1,survey_sub$Q3A)))))))
# harmonize vote variable and create survey comparison dataset
RT_set = data.frame(
  R = as.numeric(
    gsub("Beto O'Rourke",0,gsub("Ted Cruz",1,gsub("Neal Dikeman",NA,
                                                  gsub("\\$spanish_flag","",
                                                       gsub("No opinion",NA,gsub("Someone else",NA,
                                                                                 gsub("Haven't thought about it enough to have an opinion",NA,survey_sub$Q16)))))))),
  T = survey_sub$voting_history_prob,
  Voters_Gender = survey_sub$gender,
  EthnicGroups_EthnicGroup1Desc = survey_sub$race ,
  CommercialData_Education = survey_sub$educ ,
  age_cat = survey_sub$age ,
  Parties_Description =survey_sub$PID3_utex,
  time = survey_sub$endtime)

# fill in NA Parties description to "indepedent"
RT_set$Parties_Description = as.character(unlist(RT_set$Parties_Description))
RT_set$Parties_Description[which(is.na(RT_set$Parties_Description))]="Non-Partisan"


# Now we smooth the survey estimates over the relevant categories using a random forest 
# create turnout dataset to feed to RF - needs to be complete 
T_set_complete = RT_set[which(!is.na(RT_set$T)),-which(names(RT_set)=="R")]
T_set_na = RT_set[which(is.na(RT_set$T)),-which(names(RT_set)=="R"|names(RT_set)=="T")]

library(ranger)
library(parallel)
rf_model_T = 
  ranger(formula =  T~.,
         data = T_set_complete,
         num.trees = 1000,
         probability = FALSE,
         classification = FALSE,
         importance = 'permutation'
  )

rev(sort(rf_model_T$variable.importance))
rf_model_T$prediction.error

rf_model_T_pred =predict(
  object = rf_model_T, 
  data = T_set_na,
  predict.all = TRUE,
  num.trees = rf_model_T$num.trees, 
  type = "response",
  seed = NULL, 
  num.threads = detectCores(),
  verbose = TRUE)

# get predictions back into dataset replacing turnout NAs
RT_set$T[which(is.na(RT_set$T))] = apply(rf_model_T_pred$predictions,1,mean)

# now remove missing R  to smoothen vote choice 
R_set_complete = RT_set[which(!is.na(RT_set$R)),]
R_set_na = RT_set[which(is.na(RT_set$R)),-which(names(RT_set)=="R")]

rf_model_R = 
  ranger(formula =  R~.,
         data = R_set_complete[,-which(names(R_set_complete)=="T")],
         case.weights = R_set_complete$T,
         num.trees = 1000,
         probability = TRUE,
         classification = FALSE,
         importance = 'permutation'
  )

rev(sort(rf_model_R$variable.importance))
rf_model_R$prediction.error

# set of all possible categories
cats = 
  expand.grid(Voters_Gender = levels(RT_set$Voters_Gender),
              EthnicGroups_EthnicGroup1Desc = levels(RT_set$EthnicGroups_EthnicGroup1Desc),
              CommercialData_Education = levels(RT_set$CommercialData_Education),
              age_cat = levels(RT_set$age_cat),
              Parties_Description = levels(as.factor(RT_set$Parties_Description)),
              time =max(RT_set$time)
  )

# predictions per category for turnout and vote choice 
# turnout
rf_model_T_pred =predict(
  object = rf_model_T, 
  data = cats,
  predict.all = TRUE,
  num.trees = rf_model_T$num.trees, 
  type = "response",
  #se.method = "infjack",
  seed = NULL, 
  num.threads = detectCores(),
  verbose = TRUE)

apply(rf_model_T_pred$predictions,1,mean)

# vote-choice
rf_model_R_pred =predict(
  object = rf_model_R, 
  data = cats,
  #predict.all = TRUE,
  num.trees = rf_model_R$num.trees, 
  type = "response",
  #se.method = "infjack",
  seed = NULL, 
  num.threads = detectCores(),
  verbose = TRUE)

survey_preds = cbind(R_survey = rf_model_R_pred$predictions[,2],T_survey = apply(rf_model_T_pred$predictions,1,mean),cats[,-which(names(cats)=="time")])
survey_v_dv = merge(temp_3w_nopart_agg,survey_preds,by = names(cats)[-which(names(cats)=="time")],all=TRUE)
survey_v_dv = survey_v_dv [-which(is.na(survey_v_dv $R_survey)),]

# category comparison scatter plot 
# FIGURE 5
pdf(file=paste("voter_sub_categories_nores.pdf",sep=""),width = 5.00, height = 5.00)
par(mar = c(4,4,1,1))
survey_v_dv = survey_v_dv[order(survey_v_dv$R),]
plot(survey_v_dv$R,survey_v_dv$R_survey,ylim = c(0,1),xlim = c(0,1),
     bty = "n",xlab = "Digital Vote Estimates",ylab="Survey Estimates",
     col = "grey"
)
abline(0,1,lty = 2,lwd = 2)
lo <- loess(survey_v_dv$R_survey~survey_v_dv$R)
lines(survey_v_dv$R,
      predict(lo),
      col = "darkgreen", 
      lwd=2)
dev.off()

mean(survey_v_dv$R - survey_v_dv$R_survey)

png(file=paste("voter_sub_categories_T.png",sep=""),width = 500, height = 500)
par(mar = c(4,4,1,1))
survey_v_dv = survey_v_dv[order(survey_v_dv$T),]
plot(survey_v_dv$T,survey_v_dv$T_survey,ylim = c(0,1),xlim = c(0,1),
     bty = "n",xlab = "Digital Vote Estimates",ylab="Survey Estimates",
     col = "grey"
)
abline(0,1,lty = 2,lwd = 2)
lo <- loess(survey_v_dv$T_survey~survey_v_dv$T)
lines(survey_v_dv$T,
      predict(lo),
      col = "darkgreen", 
      lwd=2)
dev.off()

# TURNOUT ANALYSIS
# load predictions 
Pred_T_CD_2018 = read.csv('CD_dt_T.csv')[,-1]
# load publicly released turnout by district 
T_CD_2018 = read.csv('TURNOUT_BY_DISTRICT.csv')
T_CD_2018 = T_CD_2018[which(T_CD_2018$PARTY=='Race Total'),]
T_CD_2018$RACE= 0:36 # senate is 0
T_CD_2018 = T_CD_2018[,-which(names(T_CD_2018)=='PARTY'|names(T_CD_2018)=='NAMES')]
T_CD_2018$CANVASS.VOTES/15793257 # number of registered voters from : https://www.sos.state.tx.us/elections/historical/70-92.shtml
# rescale predictions to account for registration file underestimating registration by about 2.5 million users 
# assume we can just rescale, i.e. the missing 2.5 are eavenly dispersed across the districts 
# get true registered
Pred_T_CD_2018$R_star = Pred_T_CD_2018$R+c(
  (15793257-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0]),
  rep(((15793257-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0])/36),36))
# calcylate adjusted turnout preds
Pred_T_CD_2018$T_star = Pred_T_CD_2018$T+c(
  (8371655-Pred_T_CD_2018$T[Pred_T_CD_2018$CD==0]),
  rep(((8371655-Pred_T_CD_2018$T[Pred_T_CD_2018$CD==0])/36),36))
# calcylate 2018 truth
T_2018_TRUTH = T_CD_2018$CANVASS.VOTES/Pred_T_CD_2018$R_star 
Pred_T_CD_2018$P_T_star = Pred_T_CD_2018$T_star/Pred_T_CD_2018$R_star
# upload 2016 turnout by district 
T_CD_2016 = read.csv('TURNOUT_BY_DISTRICT_2016.csv')
T_CD_2016 = T_CD_2016[which(T_CD_2016$PARTY=='Race Total'),]
T_CD_2016$RACE= 0:36 # senate is 0
T_CD_2016 = T_CD_2016[,-which(names(T_CD_2016)=='PARTY'|names(T_CD_2016)=='NAMES')]
T_CD_2016$CANVASS.VOTES
# agian get 2018 true district wise registry 
Pred_T_CD_2018$R_star_16 = Pred_T_CD_2018$R+c(
  (15101087-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0]),
  rep(((15101087-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0])/36),36))
# calculate turnout accoridng to 2016 results
T_2016_TRUTH = T_CD_2016$CANVASS.VOTES/Pred_T_CD_2018$R_star_16
# and repeat process with 2014
T_CD_2014 = read.csv('TURNOUT_BY_DISTRICT_2014.csv')
T_CD_2014 = T_CD_2014[which(T_CD_2014$PARTY=='Race Total'),]
T_CD_2014$RACE= 0:36 # senate is 0
T_CD_2014 = T_CD_2014[,-which(names(T_CD_2014)=='PARTY'|names(T_CD_2014)=='NAME')]

Pred_T_CD_2018$R_star_14 = Pred_T_CD_2018$R+c(
  (14025441-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0]),
  rep(((14025441-Pred_T_CD_2018$R[Pred_T_CD_2018$CD==0])/36),36))

T_2014_TRUTH = T_CD_2014$CANVASS.VOTES/Pred_T_CD_2018$R_star_14

# FIGURE 6
# turnout analysis 
png(file=paste("turnout_predictions.png",sep=""),width = 500, height = 500)
plot(Pred_T_CD_2018$P_T,T_2018_TRUTH,
     bty = 'n',ylab = '2018 Observed Turnout',xlab = 'Predicted Turnout',
     ylim = c(0.1,0.8),xlim = c(0.1,0.8),main = '',col = 'darkgrey')
lines(Pred_T_CD_2018$P_T[order(Pred_T_CD_2018$P_T)],
      loess(T_2018_TRUTH[order(Pred_T_CD_2018$P_T)]~Pred_T_CD_2018$P_T[order(Pred_T_CD_2018$P_T)])$fitted,
      col = "darkgreen",lty = 1,lwd = 2)
par(new=TRUE)
plot(T_2016_TRUTH,T_2018_TRUTH,
     bty = 'n',ylab = '',xlab = '',
     ylim = c(0.1,0.8),xlim = c(0.1,0.8),pch = 4,xaxt = 'n',yaxt = 'n',col = 'darkgrey')
lines(T_2016_TRUTH[order(T_2016_TRUTH)],
      loess(T_2018_TRUTH[order(T_2018_TRUTH)]~T_2016_TRUTH[order(T_2016_TRUTH)])$fitted,
      col = "darkgreen",lty = 5,lwd = 2)
par(new=TRUE)
plot(T_2014_TRUTH,T_2018_TRUTH,
     bty = 'n',ylab = '',xlab = '',
     ylim = c(0.1,0.8),xlim = c(0.1,0.8),pch = 3,xaxt = 'n',yaxt = 'n',col = 'darkgrey')
lines(T_2014_TRUTH[order(T_2014_TRUTH)],
      loess(T_2018_TRUTH[order(T_2018_TRUTH)]~T_2014_TRUTH[order(T_2014_TRUTH)])$fitted,
      col = "darkgreen",lty = 4,lwd = 2)
abline(a = 0, b = 1,lty = 2)
legend('bottomright',legend = c('2018 Predicted','2016 Observed','2014 Observed'),
       pch=c(1,4,3),bty ='n')
dev.off()
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # sample size cost comparison # # # # ## # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# upload fivethrityeight polls 
f38_polls = read.csv("f38_polls_count_monitoring_period.csv")
sum(f38_polls$number_of_polls)

f38_polls = read.csv("f38_polls.csv",header = FALSE)

length(which(!is.na(f38_polls[,7])))

f38_polls_senate = read.csv("f38_polls_senate.csv",header = FALSE)
sum(f38_polls_senate[,7],na.rm=TRUE)
length(which(!is.na(f38_polls_senate[,7])))

f38_polls_house_texas = read.csv("f38_polls_house_texas.csv",header = FALSE)
sum(f38_polls_house_texas[,7],na.rm=TRUE)
length(which(!is.na(f38_polls_house_texas[,7])))


f38_polls_senate_texas = read.csv("f38_polls_senate_texas.csv",header = FALSE)
sum(f38_polls_senate_texas[,7],na.rm=TRUE)
length(which(!is.na(f38_polls_senate_texas[,7])))

results = read.csv("RESULTS.csv")[,-1]

# alpha = number of times we expect the estimate of the proportion derived from our sample to be outside the specified interval
# d = the margin of error - the specified interval is the estimated proportion +/- the marging of error
# pi = proportion probability that leads to maximum variance (a priori complete ignorance ) 


sample_size_prop_2sided = function(alpha,pi,d) { (( qnorm(alpha) * sqrt(pi*(1-pi)) )/ d )^2 }
sample_size_prop_2sided(alpha = 0.025, pi = 0.5, d = 0.01)


sample_size_prop_2sided(alpha = 0.005,pi = 0.5, d = 0.05)

sample_size_prop_2sided(alpha = 0.025,pi = results[results$wte==0,"res"], 
                        d = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])
)


sample_size_prop_2sided(alpha = 0.025,pi = 0.5,d = 0.01)

# as-if-probability sample size for each election
# FiveThirtyEight
# use usual alpha 
cbind(results[results$wte==0,"cd"],sample_size_prop_2sided(alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])))
sum(sample_size_prop_2sided(alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])),na.rm=TRUE)
# DigitalVote
cbind(results[results$wte==0,"cd"],sample_size_prop_2sided(alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"])))
sum(sample_size_prop_2sided(alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"])),na.rm=TRUE)

# TABLE APPENDIX C.1
floor = function(x,f){ifelse(x<f,f,x)}

library(xtable)
print(
  xtable(x = data.frame( 
    district = c(paste("CD-",results[results$wte==0,"cd"],sep=""),"Total"),
    DigitalVote_AE_pct =  c(round(abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"])*100,2),""),
    # DigitalVoteN_Ignorant = c(as.integer(
    #    round(sample_size_prop_2sided(
    #      alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"])))),
    #    sum(as.integer(
    #      round(sample_size_prop_2sided(
    #        alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"])))),na.rm=TRUE)),
    DigitalVoteN_Informed = c(as.integer(
      round(sample_size_prop_2sided(
        alpha = 0.025, pi = results[results$wte==0,"X50"], 
        d = floor(x = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"]),f = 0.01)
        ))),
      sum(as.integer(
        round(sample_size_prop_2sided(
          alpha = 0.025, pi = results[results$wte==0,"X50"], 
          d = floor(x = abs(results[results$wte==0,"res"] - results[results$wte==0,"X50"]),f = 0.01)
          ))),na.rm=TRUE))
    ,
    FiveThirtyEight_AE_pct = c(round(abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])*100,2),""),
    #  FiveThirtyEight_N_Ignnorant = c(as.integer(
    #    round(sample_size_prop_2sided(
    #      alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])))),
    #    sum(as.integer(
    #      round(sample_size_prop_2sided(
    #        alpha = 0.025, pi = 0.5, d = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"])))),na.rm=TRUE)
    #    ),
    FiveThirtyEight_N_Informed = c(as.integer(
      round(sample_size_prop_2sided(
        alpha = 0.025, pi = results[results$wte==0,"f38_central"], 
        d = floor(x = abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"]),f = 0.01)
        ))),
      sum(as.integer(
        round(sample_size_prop_2sided(
          alpha = 0.025, pi = results[results$wte==0,"f38_central"], 
          d = floor(x =abs(results[results$wte==0,"res"] - results[results$wte==0,"f38_central"]),f = 0.01)
          ))),na.rm=TRUE)
    )),
    caption = "Table showing the hypothetical sample size needed in independenty probability surveys to obtain the levels of accuracy observed."),
  include.rownames=FALSE)

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # sample size evolution # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# new plot for sample size evolution
# FIGURE APPENDIX A.1
sample_ev = read.csv("DATA_FOR_FIGURE__EVOLUTION OF THE SAMPLE.csv")[,-1]

png(file=paste("sample_evolution.png",sep=""),width = 850, height = 500)
par(oma = c(0,0,0,5),xpd = TRUE)

plot(sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"],
     sample_ev$n[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"] + sample_ev$n[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US House"] ,
     xlim =c(13,0),
     pch = NA,
     bty = "n",
     ylab = "n",
     xlab = "weeks to election",
     ylim = c(0,2500),
     main = "",
     xaxt = "n")

lines(sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"],
      sample_ev$n[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"] + sample_ev$n[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US House"],
      col = "black"
)
lines(7 + sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US Senate"],
      sample_ev$n[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US Senate"] + sample_ev$n[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US Senate"],
      col = "black"
)

lines(sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"],
      sample_ev$n[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US House"] ,
      col = "red"
)
lines(7 + sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US Senate"],
      sample_ev$n[sample_ev$Party=="Rep" & sample_ev$Election.Type=="US Senate"] ,
      col = "red"
)


lines(sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US House"],
      sample_ev$n[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US House"] ,
      col = "blue"
)
lines(7 + sample_ev$Weeks.to.Election.Day[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US Senate"],
      sample_ev$n[sample_ev$Party=="Dem" & sample_ev$Election.Type=="US Senate"] ,
      col = "blue"
)

abline(v = 6.5, lty =2 ,xpd =FALSE)
axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))

text(x = 3, y = 2500,labels = "House",font = 2)
text(x = 10, y = 2500,labels = "Senate", font = 2)

legend(x = -0.5,y = 1500,
       legend = c("Total","Republicans","Democrats"),
       col =c("black","red","blue"),
       lty = 1,
       xpd = NA,
       cex = 0.8,
       bty = "n")

dev.off()


# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # plot sample v population plots # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # #
# data provided separately for each plot as derived directly from registration file 
partisanship_sample_pop = read.csv("PARTISANSHIP_SAMPLE_POP.csv")

race_sample_pop = read.csv("RACE_SAMPLE_POP.csv")
age_sample_pop = read.csv("AGE_SAMPLE_POP.csv")
gender_sample_pop = read.csv("GENDER_SAMPLE_POP.csv")
education_sample_pop = read.csv("EDU_SAMPLE_POP.csv")

# GENDER - APPENDIX A.3
png(file=paste("GENDER_SAMPLE_POP.png",sep=""),width = 550, height = 500)
par(oma = c(2,0,0,0),xpd = TRUE)

plot(gender_sample_pop$Week_to_Election[gender_sample_pop$from=="reg_pop"],
     gender_sample_pop$pop_prop[gender_sample_pop$from=="reg_pop"],
     xlim =c(13,0),
     pch = NA,
     bty = "n",
     ylab = "proportion",
     xlab = "weeks to election",
     ylim = c(0,0.8),
     main = "Gender Groups",
     xaxt = "n")

lines(gender_sample_pop$Week_to_Election[gender_sample_pop$from=="reg_pop" & gender_sample_pop$Gender=="M"],
      gender_sample_pop$pop_prop[gender_sample_pop$from=="reg_pop" & gender_sample_pop$Gender=="M"],
      col = "darkgreen")
lines(7+gender_sample_pop$Week_to_Election[gender_sample_pop$from=="reg_pop" & gender_sample_pop$Gender=="F"],
      gender_sample_pop$pop_prop[gender_sample_pop$from=="reg_pop" & gender_sample_pop$Gender=="F"],
      col = "darkgreen")


lines(gender_sample_pop$Week_to_Election[gender_sample_pop$from=="sample_pop" & gender_sample_pop$Gender=="M"],
      gender_sample_pop$pop_prop[gender_sample_pop$from=="sample_pop" & gender_sample_pop$Gender=="M"],
      col = "black")
lines(7+gender_sample_pop$Week_to_Election[gender_sample_pop$from=="sample_pop" & gender_sample_pop$Gender=="F"],
      gender_sample_pop$pop_prop[gender_sample_pop$from=="sample_pop" & gender_sample_pop$Gender=="F"],
      col = "black")

abline(v=c(6.5),lty = 2,xpd = FALSE)

axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))

text(x = 3, y = 0.8,labels = "Male",font = 1,cex = 0.9)
text(x = median( 7:13), y = 0.8,labels = "Female",font = 1,cex = 0.9)

legend(x =14.75,y = -0.175,
       legend = c("Population","Sample"),
       col =c("darkgreen","black"),
       lty = 1,
       xpd = NA,horiz = TRUE,
       cex = 1,
       bty = "n")

dev.off()

# PARTISANSHIP - FIGURE 1 
png(file=paste("PARTISANSHIP_SAMPLE_POP.png",sep=""),width = 950, height = 500)
par(oma = c(3,0,0,0),xpd = TRUE, mar = c(5,4,1,0))

plot(partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="reg_pop"],
     partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="reg_pop"],
     xlim =c(20,0),
     pch = NA,
     bty = "n",
     ylab = "proportion",
     xlab = "weeks to election",
     ylim = c(0,0.8),
     main = "",#"Registered Partisans",
     xaxt = "n")

lines(partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Democratic"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Democratic"],
      col = "darkgreen",lwd=2)
lines(7+partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Non-Partisan"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Non-Partisan"],
      col = "darkgreen",lwd=2)
lines(14 + partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Republican"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="reg_pop" & partisanship_sample_pop$Registered_Party=="Republican"],
      col = "darkgreen",lwd=2)



lines(partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Democratic"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Democratic"],
      col = "black",lwd=2)
lines(7+partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Non-Partisan"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Non-Partisan"],
      col = "black",lwd=2)
lines(14 + partisanship_sample_pop$Week_to_Election[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Republican"],
      partisanship_sample_pop$pop_prop[partisanship_sample_pop$from=="sample_pop" & partisanship_sample_pop$Registered_Party=="Republican"],
      col = "black",lwd=2)


abline(v=c(6.5,13.5),lty = 2,xpd = FALSE)

axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))
axis(side = 1,at = 14:20,labels = c(0:6))

text(x = 3, y = 0.8,labels = "Democratic",font = 1,cex = 1)
text(x = median( 7:13), y = 0.8,labels = "Non-Partisan",font = 1,cex = 1)
text(x = median(14:20), y = 0.8,labels = "Republican",font = 1,cex = 1)

legend(x =22,y = -0.225,
       legend = c("Population","Sample"),
       col =c("darkgreen","black"),
       lty = 1,
       lwd = 2,
       xpd = NA,horiz = TRUE,
       cex = 1,
       bty = "n")

dev.off()

# RACE - APPENDIX FIGURE A.4 
png(file=paste("RACE_SAMPLE_POP.png",sep=""),width = 1100, height = 500)
par(oma = c(2,0,0,0),xpd = TRUE)

plot(race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop"],
     race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop"],
     xlim =c(33,0),
     pch = NA,
     bty = "n",
     ylab = "proportion",
     xlab = "weeks to election",
     ylim = c(0,0.8),
     main = "Ethnic Groups",
     xaxt = "n")

lines(race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="European"],
      race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="European"],
      col = "darkgreen")
lines(7+race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Hispanic and Portuguese"],
      race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Hispanic and Portuguese"],
      col = "darkgreen")
lines(14 + race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Likely African-American"],
      race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Likely African-American"],
      col = "darkgreen")
lines(21 + race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="East and South Asian"],
      race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="East and South Asian"],
      col = "darkgreen")
lines(28 + race_sample_pop$Week_to_Election[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Other"],
      race_sample_pop$pop_prop[race_sample_pop$from=="reg_pop" & race_sample_pop$Ethnicity=="Other"],
      col = "darkgreen")


lines(race_sample_pop$Week_to_Election[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="European"],
      race_sample_pop$pop_prop[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="European"],
      col = "black")
lines(7+race_sample_pop$Week_to_Election[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Hispanic and Portuguese"],
      race_sample_pop$pop_prop[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Hispanic and Portuguese"],
      col = "black")
lines(14 + race_sample_pop$Week_to_Election[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Likely African-American"],
      race_sample_pop$pop_prop[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Likely African-American"],
      col = "black")
lines(21 + race_sample_pop$Week_to_Election[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="East and South Asian"],
      race_sample_pop$pop_prop[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="East and South Asian"],
      col = "black")
lines(28 + race_sample_pop$Week_to_Election[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Other"],
      race_sample_pop$pop_prop[race_sample_pop$from=="sample_pop" & race_sample_pop$Ethnicity=="Other"],
      col = "black")

abline(v=c(6.5,13.5,20.5,27.5),lty = 2,xpd = FALSE)

axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))
axis(side = 1,at = 14:20,labels = c(0:6))
axis(side = 1,at = 21:27,labels = c(0:6))
axis(side = 1,at = 28:34,labels = c(0:6))

text(x = 3, y = 0.8,labels = "European",font = 1,cex = 0.9)
text(x = median( 7:13), y = 0.8,labels = "Hispanic and Portuguese",font = 1,cex = 0.9)
text(x = median(14:20), y = 0.8,labels = "Likely African-American",font = 1,cex = 0.9)
text(x = median(21:27), y = 0.8,labels = "East and South Asian",font = 1,cex = 0.9)
text(x = median(28:34), y = 0.8,labels = "Other",font = 1,cex = 0.9)

legend(x =36,y = -0.175,
       legend = c("Population","Sample"),
       col =c("darkgreen","black"),
       lty = 1,
       xpd = NA,horiz = TRUE,
       cex = 1,
       bty = "n")

dev.off()


# EDUCATION - APPENDIX FIGURE A.5 
png(file=paste("EDU_SAMPLE_POP.png",sep=""),width = 1250, height = 500)
par(oma = c(2,0,0,0),xpd = TRUE)

plot(education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop"],
     education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop"],
     xlim =c(40,0),
     pch = NA,
     bty = "n",
     ylab = "proportion",
     xlab = "weeks to election",
     ylim = c(0,0.4),
     main = "Educational Groups",
     xaxt = "n")

lines(education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Vocational Technical Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Vocational Technical Degree"],
      col = "darkgreen")
lines(7+education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Grad Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Grad Degree"],
      col = "darkgreen")
lines(14 + education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Bach Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Bach Degree"],
      col = "darkgreen")
lines(21 + education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Some College"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Some College"],
      col = "darkgreen")
lines(28 + education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="HS Diploma"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="HS Diploma"],
      col = "darkgreen")
lines(35 + education_sample_pop$Week_to_Election[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Less than HS Diploma"],
      education_sample_pop$pop_prop[education_sample_pop$from=="reg_pop" & education_sample_pop$Edu=="Less than HS Diploma"],
      col = "darkgreen")


lines(education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Vocational Technical Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Vocational Technical Degree"],
      col = "black")
lines(7+education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Grad Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Grad Degree"],
      col = "black")
lines(14 + education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Bach Degree"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Bach Degree"],
      col = "black")
lines(21 + education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Some College"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Some College"],
      col = "black")
lines(28 + education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="HS Diploma"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="HS Diploma"],
      col = "black")
lines(35 + education_sample_pop$Week_to_Election[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Less than HS Diploma"],
      education_sample_pop$pop_prop[education_sample_pop$from=="sample_pop" & education_sample_pop$Edu=="Less than HS Diploma"],
      col = "black")

abline(v=c(6.5,13.5,20.5,27.5,34.5),lty = 2,xpd = FALSE)

axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))
axis(side = 1,at = 14:20,labels = c(0:6))
axis(side = 1,at = 21:27,labels = c(0:6))
axis(side = 1,at = 28:34,labels = c(0:6))
axis(side = 1,at = 35:41,labels = c(0:6))

text(x = 3, y = 0.4,labels = "Vocational/Technical",font = 1,cex = 0.9)
text(x = median( 7:13), y = 0.4,labels = "Grad Degree",font = 1,cex = 0.9)
text(x = median(14:20), y = 0.4,labels = "Bach Degree",font = 1,cex = 0.9)
text(x = median(21:27), y = 0.4,labels = "Some College",font = 1,cex = 0.9)
text(x = median(28:34), y = 0.4,labels = "HS Diploma",font = 1,cex = 0.9)
text(x = median(35:41), y = 0.4,labels = "Less than HS Diploma",font = 1,cex = 0.9)

legend(x =43.5,y = -0.085,
       legend = c("Population","Sample"),
       col =c("darkgreen","black"),
       lty = 1,
       xpd = NA,horiz = TRUE,
       cex = 1,
       bty = "n")

dev.off()

# AGE - APPENDIX FIGURE A.2
png(file=paste("AGE_SAMPLE_POP.png",sep=""),width = 1250, height = 500)
par(oma = c(2,0,0,0),xpd = TRUE)

plot(age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop"],
     age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop"],
     xlim =c(40,0),
     pch = NA,
     bty = "n",
     ylab = "proportion",
     xlab = "weeks to election",
     ylim = c(0,0.4),
     main = "Age Groups",
     xaxt = "n")

lines(age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="65_or_older"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="65_or_older"],
      col = "darkgreen")
lines(7+age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="55-64"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="55-64"],
      col = "darkgreen")
lines(14 + age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="45-54"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="45-54"],
      col = "darkgreen")
lines(21 + age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="35-44"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="35-44"],
      col = "darkgreen")
lines(28 + age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="25-34"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="25-34"],
      col = "darkgreen")
lines(35 + age_sample_pop$Week_to_Election[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="18-24"],
      age_sample_pop$pop_prop[age_sample_pop$from=="reg_pop" & age_sample_pop$Age=="18-24"],
      col = "darkgreen")


lines(age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="65_or_older"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="65_or_older"],
      col = "black")
lines(7+age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="55-64"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="55-64"],
      col = "black")
lines(14 + age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="45-54"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="45-54"],
      col = "black")
lines(21 + age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="35-44"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="35-44"],
      col = "black")
lines(28 + age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="25-34"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="25-34"],
      col = "black")
lines(35 + age_sample_pop$Week_to_Election[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="18-24"],
      age_sample_pop$pop_prop[age_sample_pop$from=="sample_pop" & age_sample_pop$Age=="18-24"],
      col = "black")

abline(v=c(6.5,13.5,20.5,27.5,34.5),lty = 2,xpd = FALSE)

axis(side = 1,at = 0:6,labels = c(0:6))
axis(side = 1,at = 7:13,labels = c(0:6))
axis(side = 1,at = 14:20,labels = c(0:6))
axis(side = 1,at = 21:27,labels = c(0:6))
axis(side = 1,at = 28:34,labels = c(0:6))
axis(side = 1,at = 35:41,labels = c(0:6))

text(x = 3, y = 0.4,labels = "65 or older",font = 1,cex = 0.9)
text(x = median( 7:13), y = 0.4,labels = "55-64",font = 1,cex = 0.9)
text(x = median(14:20), y = 0.4,labels = "45-54",font = 1,cex = 0.9)
text(x = median(21:27), y = 0.4,labels = "35-44",font = 1,cex = 0.9)
text(x = median(28:34), y = 0.4,labels = "25-34",font = 1,cex = 0.9)
text(x = median(35:41), y = 0.4,labels = "18-24",font = 1,cex = 0.9)

legend(x =43.5,y = -0.085,
       legend = c("Population","Sample"),
       col =c("darkgreen","black"),
       lty = 1,
       xpd = NA,horiz = TRUE,
       cex = 1,
       bty = "n")

dev.off()

