library(tidyverse)
library(cowplot)
library(lubridate)
library(mgcv)
library(splines)
library(MASS)
library(dplyr)
library(rstanarm)
library(rjags)
library(R2jags)

#### forecast date
last_date <- as.Date("2022-10-15")


#source("R/get_hhs.R")
hhs <- read.csv("hhs_data.csv") ##get_hhs()  ### to replaced with however you want to get the data
hhs <- hhs[hhs$date < last_date,]
hhs_sub <- hhs %>%
  dplyr::select(state, date, total_patients_hospitalized_confirmed_influenza)
hhs_sub <- hhs_sub %>% dplyr::select(state,date,total_patients_hospitalized_confirmed_influenza)
hhs_sub$flu <- hhs_sub$total_patients_hospitalized_confirmed_influenza
hhs_sub <- hhs_sub %>% dplyr::select(state,date,flu)


hhs_sub$date = as.Date(strftime(hhs_sub$date,"%Y-%m-%d"))

uspop <- read.csv("data/uspop.csv")
uspop$location <- state.abb[match(uspop$location,state.name)]


hhs_sub_us <- hhs_sub %>% dplyr::group_by(date) %>% dplyr::summarise(state="US",flu = sum(flu))
hhs_sub <- rbind(hhs_sub,hhs_sub_us)

hhs_sub <- hhs_sub %>% filter(date < last_date)


sim_list <- list()

for (state_ in setdiff(unique(hhs_sub$state),c('DC'))){

  last_season <- seq(as.Date("2022-03-01"),as.Date("2022-08-01"),by='day')
  last_season_flu <- hhs_sub[hhs_sub$state == state_ & hhs_sub$date %in% last_season,]$flu

  tmp_len <- length(hhs_sub[hhs_sub$state == state_ & hhs_sub$date > as.Date("2022-09-15"),]$flu)
  current_season_flu <- hhs_sub[hhs_sub$state == state_ & hhs_sub$date > as.Date("2022-09-15"),]$flu
  while (length(current_season_flu) < length(last_season)){
    current_season_flu <- c(current_season_flu,NA)
  }

  flu_mat <- rbind(last_season_flu,current_season_flu)

  ### fir current1wave
  model_code <- "
    model
    {
       s_total ~ dunif(.1,.99)
      r <- 10000

      for (j in 1:2){
        s[1,j] <- s_total

        i0_[j] ~ dunif(i0[j]/N,(i0[j]+100)/N)
        i[1,j] <- i0_[j]

        I[1,j] <- i0_[j]

        log_beta[1,j] ~ dnorm(log(.5),.1)
        for (t in 2:(T+30)){
          log_beta[t,j] ~ dnorm(log_beta[t-1,j],10000)
        }


        gamma[j] <- .25
        for (t in 2:(T+30)){
            i[t,j] <- exp(log_beta[t,j])*I[t-1,j]*s[t-1,j]
            I[t,j] <- I[t-1,j] - gamma[j]*I[t-1,j]+ i[t,j]
            s[t,j] <- s[t-1,j] - i[t,j]
        }


          for (t in 1:T){
                #log(lambda[t,j]) <- N*max(1,i[t,j])
                p[t,j] <- r/(r+N*i[t,j])

                y[t,j] ~ dnegbin(p[t,j],r)

          }
        }
    }

    "

  pops_local <- uspop[uspop$location==state_,]$Pop_Est_2019
  pops_local <- pops_local[which(!is.na(pops_local))][1]
  model_data =list(N=pops_local,i0=flu_mat[,1],
                   y = t(flu_mat) +1 ,
                   T = ncol(flu_mat))

  model_parameters <- c("i","s","log_beta")

  # Run the model
  model_run <- jags(
    data = model_data,
    parameters.to.save = model_parameters,
    model.file = textConnection(model_code),
    n.chains = 4, # Number of different starting positions
    n.iter = 1000, # Number of iterations
    n.burnin = 200, # Number of iterations to remove at start
    n.thin = 2
  ) # Amo
  sims_list <- model_run$BUGSoutput$sims.list$i[,,2]
  plot(colMeans(sims_list))
  sims_list <- sims_list[,seq(tmp_len+1,tmp_len+30)]*model_data$N
  plot(seq(1,tmp_len),model_data$y[1:tmp_len,2],col='red',type='l',xlim=c(1,tmp_len+30))
  lines(seq(tmp_len+1,tmp_len+30),colMeans(sims_list),type='l',ylim=c(0,200))
  sim_list[[state_]] <-sims_list
}

write.csv(sims_list,file=paste0("mech",last_date,".csv"))

