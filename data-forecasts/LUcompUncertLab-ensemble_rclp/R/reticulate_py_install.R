# this file has an R command that will install the rclp module in the
# version of python that is used by reticulate

# update the first argument of `reticulate::py_install` to the path of a local
# clone of https://github.com/reichlab/rclp
# there is probably also a way to directly reference the github repo without
# having to clone it

library(reticulate)

reticulate::py_install(
  "/Users/tommcandrew/science/rclp",
  method = "auto",
  conda = "auto",
  pip = TRUE,
  pip_options = "--force-reinstall")
