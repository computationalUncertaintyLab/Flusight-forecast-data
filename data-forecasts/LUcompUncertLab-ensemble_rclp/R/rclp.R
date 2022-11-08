# This should be run with the repository root as working directory, using a
# command like
#
# Rscript --vanilla R/rclp.R 2022-04-02 BetaMixtureRCLP 5 all path/to/output.csv
#
# Arguments are:
# * reference date: a Saturday, two days before the Monday forecast date
# * rclp_method: the name of a class provided by the rclp module. Currently, one
# of "EqualLP", "LP", or "BetaMixtureRCLP" (in increasing order of complexity)
# * K: integer number of components in beta mixture for recalibration. Only used
# if rclp_method is "BetaMixtureRCLP"
# * history_length: "all" or an integer. Fit using all history or only the most
# recent target end dates
# * results_path: a path to save the RCLP output


library(lubridate)

# tidyverse packages
library(dplyr)
library(tidyr)
library(purrr)

# covidcast for loading data
# https://cmu-delphi.github.io/covidcast/covidcastR/
library(covidcast)

# covidHubUtils for loading component model forecast files
# https://github.com/reichlab/covidHubUtils
library(covidHubUtils)

# distfromq for approximating a distribution's pdf and cdf from a collection
# of quantiles
# https://github.com/reichlab/distfromq
library(distfromq)

# reticulate for interacting with the python module rclp
# rclp should be installed in the version of python that reticulate is
# configured to work with. See reticulate_py_install.R for one way to do that
library(reticulate)

# function for loading truth data
source("./R/load_flu_hosp_data.R")

# function to do a grid search to get quantiles from a cdf
source("./R/get_rclp_quantiles.R")


# process command line arguments to this script
# args <- c("2022-04-02", "BetaMixtureRCLP", "5", "all", "rclp_forecasts.csv")
#args <- commandArgs(trailingOnly = TRUE)

# The reference_date is the date of the Saturday relative to which week-ahead
# targets are defined.
reference_date <- "2022-11-05"#args[1]

# rclp_method is the name of a class provided by the rclp module. Currently, one
# of "EqualLP", "LP", or "BetaMixtureRCLP" (in increasing order of complexity)
rclp_method <- "BetaMixtureRCLP" #args[2]

#rclp_method <- "EqualLP" #args[2]

# Number of components in beta mixture for recalibration
K <- as.integer(5) #as.integer(args[3])

# Fit using all history or only the most recent target end dates
# should be either "all" or an integer
history_length <- "all" # args[4]

# results_path is the location where the output should be saved
results_path <- "./2022-11-07-LUcompUncertLab-ensemble_rclp.csv" #args[5]

# Get the list of required locations
required_locations <-
  readr::read_csv(file = "./data/locations.csv") %>%
  dplyr::select("location", "abbreviation")

# The forecast_date is the Monday of forecast creation.
forecast_date <- as.character(as.Date(reference_date) + 2)
print(forecast_date)

# Load data
weekly_data <- load_flu_hosp_data(
    source = "covidcast",
    as_of = forecast_date,
    temporal_resolution = "weekly") %>%
  dplyr::select(location, date, data_value = value) %>%
  dplyr::filter(!is.na(data_value))

# Load component forecasts
component_forecasts <- covidHubUtils::load_forecasts_repo(
  file_path = paste0("./component-forecasts/"),
  forecast_dates = seq(ymd('2022-09-05'),ymd('2022-11-07'), by = '1 week'),
  locations = NULL,
  types = "quantile",
  targets = NULL,
  hub = "FluSight",
  verbose = TRUE
  )

print(component_forecasts)

# training set forecasts -- those with a forecast date before the
# current forecast date
train_forecasts <- component_forecasts %>%
  dplyr::filter(forecast_date < !!forecast_date)

# if requested, subset training set forecasts to those with a recent
# target end date
if (history_length != "all") {
  history_length <- as.integer(history_length)
  max_data_date <- max(weekly_data$date)
  train_target_end_dates <- seq(from = max_data_date,
                                by = -7,
                                length.out = history_length)
  train_forecasts <- train_forecasts %>%
    dplyr::filter(target_end_date %in% train_target_end_dates)
}

# forecasts for the current forecast date
current_forecasts <- component_forecasts %>%
  dplyr::filter(forecast_date == !!forecast_date)


# define a RCLP model
rclp <- reticulate::import("rclp", delay_load = TRUE)

M <- length(unique(component_forecasts$model))
if (rclp_method == "EqualLP") {
  rclp_model <- rclp$EqualLP(M = M)
} else if (rclp_method == "LP") {
  rclp_model <- rclp$LP(M = M)
} else if (rclp_method == "BetaMixtureRCLP") {
  rclp_model <- rclp$BetaMixtureRCLP(M = M, K = K)
} else {
  stop("Unsupported rclp_method")
}


# if there are parameters to estimate, fit the RCLP model
if (rclp_method != "EqualLP") {
  # For any method other than equal weighted linear pool,
  # the code below requires that the training set and current date forecasts
  # have the same set of models; check this here and error out if not
  train_models <- sort(unique(train_forecasts$model))
  current_models <- sort(unique(current_forecasts$model))

    print(train_models)
    print(current_models)

  if (!identical(train_models, current_models)) {
    stop("Require that the training set and current models are the same")
  }

  # extract log pdf and cdf values for training set forecasts
  # we add a little noise to the value column so that there is a density to
  # work with in case the forecaster had a point mass anywhere
  train_forecasts <- train_forecasts %>%
    dplyr::mutate(
      value = rnorm(n = nrow(train_forecasts), mean = value, sd = 0.1)
    ) %>%
    dplyr::inner_join(weekly_data,
                      by = c("target_end_date" = "date", "location")) %>%
    dplyr::group_by(model, forecast_date, location, horizon,
                    temporal_resolution, target_variable, target_end_date) %>%
    dplyr::summarize(
      log_d = distfromq::make_d_fn(
        ps = quantile,
        qs = value)(unique(data_value), log = TRUE),
      log_p = distfromq::make_p_fn(
        ps = quantile,
        qs = value)(unique(data_value), log = TRUE)
    )

  # convert log pdf and cdf values to N by M matrices
  component_log_prob_df <- train_forecasts %>%
    dplyr::ungroup() %>%
    dplyr::select(model, forecast_date, location, horizon, log_d) %>%
    tidyr::pivot_wider(id_cols = c("forecast_date", "location", "horizon"),
                       names_from = "model",
                       values_from = "log_d") %>%
    dplyr::select(-c("forecast_date", "location", "horizon"))
  component_log_prob <- as.matrix(component_log_prob_df)

  component_log_cdf_df <- train_forecasts %>%
    dplyr::ungroup() %>%
    dplyr::select(model, forecast_date, location, horizon, log_p) %>%
    tidyr::pivot_wider(id_cols = c("forecast_date", "location", "horizon"),
                       names_from = "model",
                       values_from = "log_p") %>%
    dplyr::select(-c("forecast_date", "location", "horizon"))
  component_log_cdf <- as.matrix(component_log_cdf_df)

  rclp_model$fit(
    component_log_prob = component_log_prob,
    component_log_cdf = component_log_cdf,
    num_iter = 2000L,
    verbose = FALSE)

  # in theory, we should plot the loss trace for the fit as a check that it
  # reached convergence. in practice, in every example I've looked at, 2000
  # iterations was enough

  #plot(rclp_model$loss_trace)
}


# get the quantiles that come out of the RCLP distribution for the forecasts for
# the current forecast date. Within each "forecast task" corresponding to a
# combination of forecast_date, location, and horizon, this entails a grid
# search across values of the response variable. This grid search is done by
# get_rclp_quantiles, and it's really slow.
current_forecasts <- current_forecasts %>%
  dplyr::mutate(
    value = rnorm(n = nrow(current_forecasts), mean = value, sd = 0.1)
  )

task_group_combos <- current_forecasts %>%
  dplyr::distinct(forecast_date, location, horizon, temporal_resolution,
                  target_variable, target_end_date)


rclp_forecasts <- task_group_combos %>%
  purrr::pmap_dfr(
    function(forecast_date, location, horizon, temporal_resolution,
                  target_variable, target_end_date) {
      print(paste(forecast_date, location, horizon, temporal_resolution,
                  target_variable, target_end_date))
      cf <- current_forecasts %>%
        dplyr::filter(
          forecast_date == !!forecast_date,
          location == !!location,
          horizon == !!horizon,
          temporal_resolution == !!temporal_resolution,
          target_variable == !!target_variable,
          target_end_date == !!target_end_date)
      return(
        get_rclp_quantiles(rclp_model = rclp_model,
                           component_model = cf$model,
                           ps = cf$quantile,
                           qs = cf$value,
                           target_ps = unique(cf$quantile)) %>%
          dplyr::mutate(
            forecast_date = forecast_date,
            location = location,
            horizon = horizon,
            temporal_resolution = temporal_resolution,
            target_variable = target_variable,
            target_end_date = target_end_date)
      )
    }
  )


rclp_forecasts <- rclp_forecasts %>%
  dplyr::ungroup() %>%
  dplyr::mutate(value = pmax(0, value)) %>%
  dplyr::transmute(
    forecast_date = forecast_date,
    target = paste(horizon, temporal_resolution, "ahead", target_variable),
    target_end_date = target_end_date,
    location = location,
    type = "quantile",
    quantile = quantile,
    value = value
  )

write.csv(rclp_forecasts, file = results_path, row.names = FALSE)
