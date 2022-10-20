#' Get approximate quantiles of a fitted RCLP distribution using a grid search
#' 
#' @param ps 
get_rclp_quantiles <- function(rclp_model,
                               component_model, ps, qs,
                               target_ps, tol = 0.000001, max_iter = 100,
                               verbose = FALSE) {
  # approximated p functions based on ps and qs from component models
  p_fn_per_model <- purrr::map(
    unique(component_model),
    function(cm) {
      cm_inds <- which(component_model == cm)
      return(distfromq::make_p_fn(ps = ps[cm_inds], qs = qs[cm_inds]))
    })

  # initial grid of points at which to calculate recalibrated cdf
  q_grid <- seq(from = 0.5 * min(qs), to = 2 * max(qs), length.out = 100)

  within_tol <- FALSE
  iter <- 0
  while (!within_tol && iter < max_iter) {
    # evaluate the cdfs along the q grid, assemble into a matrix with
    # entries of q_grid in rows, models in columns
    log_cdf <- matrix(NA_real_,
                      nrow = length(q_grid),
                      ncol = length(p_fn_per_model))
    for (m in seq_along(p_fn_per_model)) {
      log_cdf[, m] <- p_fn_per_model[[m]](q_grid, log.p = TRUE)
    }

    rclp_cdf <- rclp_model$cdf(log_cdf)$numpy()

    # collect information about how far off from the target ps we are
    search_status <- purrr::map_dfr(
      target_ps,
      function(target_p) {
        diffs <- rclp_cdf - target_p
        min_ind <- which.min(abs(diffs))
        data.frame(
          min_ind = min_ind,
          diff = diffs[min_ind]
        )
      }
    ) %>%
      dplyr::mutate(
        within_tol = (abs(diff) < tol)
      )

    if (verbose) {
      print(iter)
      print(search_status)
    }

    # do we need to keep searching?
    within_tol <- all(search_status$within_tol)
    iter <- iter + 1

    # as necessary, expand the q grid for another round of searching
    if (!within_tol && iter < max_iter) {
      new_q_grid <- c()
      for (i in seq_along(target_ps)) {
        min_ind <- search_status$min_ind[i]
        if (search_status$within_tol[i]) {
          # preserve the "good" q grid member in the next iteration
          new_q_grid <- c(new_q_grid, q_grid[min_ind])
        } else {
          # keep looking by expanding the grid
          if (search_status$diff[i] > 0) {
            # cdf value larger than target probability -- expand on left side to
            # find quantiles with smaller probabilities
            if (min_ind == 1) {
              # we're at the left edge of the grid -- expand the grid
              new_q_grid <- c(
                new_q_grid,
                seq(from = q_grid[1] - diff(range(q_grid)), length.out = 10))
            } else {
              # we're in the middle of the grid -- fill in more densely
              new_q_grid <- c(
                new_q_grid,
                seq(from = q_grid[min_ind], to = q_grid[min_ind - 1], length.out = 10))
            }
          } else {
            # cdf value less than target probability -- expand on right side to
            # find quantiles with larger probabilities
            if (min_ind == length(q_grid)) {
              # we're at the right edge of the grid -- expand the grid
              new_q_grid <- c(
                new_q_grid,
                seq(from = q_grid[min_ind] + diff(range(q_grid)), length.out = 10))
            } else {
              # we're in the middle of the grid -- fill in more densely
              new_q_grid <- c(
                new_q_grid,
                seq(from = q_grid[min_ind], to = q_grid[min_ind + 1], length.out = 10))
            }
          }
        }
      }
      q_grid <- sort(unique(new_q_grid))
    }
  }

  return(data.frame(
    quantile = target_ps,
    value = q_grid[search_status$min_ind]
  ))
}
