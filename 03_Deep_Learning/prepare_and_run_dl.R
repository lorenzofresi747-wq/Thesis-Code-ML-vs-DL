# ===============================================================
# MASTER'S THESIS: R TO PYTORCH (SEPARATE PYTHON PROCESS)
# Save this .R file and train_dl_33.py in the SAME folder.
# Your existing model_data object must be loaded in RStudio.
# ===============================================================

if (!exists("model_data")) {
  stop("Load your saved model_data RDS first: model_data <- readRDS(file.choose())")
}

predictor_cols <- c(
  paste0("ReturnLag", 1:5),
  paste0("SMADist", c(5, 10, 20)),
  paste0("EMADist", c(5, 10, 20)),
  paste0("Momentum", c(5, 10, 20)),
  "RSI14", "MACDRel", "MACDSignalRel",
  "BBPosition", "StochK", "StochD",
  "Vol5", "Vol20", "LogVolume", "RelVolume20",
  "DeltaOBV", "OvernightReturn", "DollarVolume", "ILLIQ20"
)

# Important: this is our reconstructed original ML feature list.
# Compare with the predictors vector from your frozen ML code before proceeding.
required <- c("Ticker", "sector", "Date", "TargetDate", "Sample", "Target", predictor_cols)
missing <- setdiff(required, names(model_data))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
stopifnot(length(unique(model_data$Ticker)) == 33, length(predictor_cols) == 28)

cols <- c("Ticker", "sector", "Date", "TargetDate", "Sample", "Target", predictor_cols)
data <- as.data.frame(model_data[, cols, drop = FALSE])
data$Sample <- as.character(data$Sample)
data$Date <- as.character(as.Date(data$Date))
data$TargetDate <- as.character(as.Date(data$TargetDate))
stopifnot(identical(sort(unique(data$Sample)), sort(c("Training", "Validation", "Test"))))
stopifnot(!anyNA(data), all(is.finite(as.matrix(data[, c("Target", predictor_cols)]))))
stopifnot(!any(duplicated(data[, c("Ticker", "Date")])))

cat("Stocks:", length(unique(data$Ticker)), "\n")
cat("Predictors:", length(predictor_cols), "\n")
print(table(data$Sample))

input_file <- file.path(getwd(), "dl_input_33.csv")
write.csv(data, input_file, row.names = FALSE)
cat("Exported:", input_file, "\n")

python_exe <- file.path(Sys.getenv("LOCALAPPDATA"),
                        "thesis-dl-torch", "Scripts", "python.exe")
python_script <- file.path(getwd(), "train_dl_33.py")
output_dir <- file.path(getwd(), "dl_output")
if (!file.exists(python_exe)) stop("Python not found at: ", python_exe)
if (!file.exists(python_script)) stop("Place train_dl_33.py in: ", getwd())

# Runs all 33 stocks. Approx. 6 candidate fits per stock, CPU-only.
# Can take a substantial amount of time; finished stocks are skipped on rerun.
status <- system2(
  command = python_exe,
  args = c(shQuote(python_script),
           "--input", shQuote(input_file),
           "--output", shQuote(output_dir),
           "--epochs", "80", "--patience", "10"),
  stdout = "", stderr = "", wait = TRUE
)
cat("Process exit status:", status, "\n")
if (!identical(as.integer(status), 0L)) {
  stop("Training ended with a Python error; previously saved stocks remain available.")
}

# Reimport results into R: same structure as the ML forecasts.
dl_all_results <- read.csv(file.path(output_dir, "dl_architecture_results.csv"))
dl_all_predictions <- read.csv(file.path(output_dir, "dl_all_predictions.csv"))
selected_dl_models <- read.csv(file.path(output_dir, "dl_selected_models.csv"))
selected_dl_predictions <- read.csv(file.path(output_dir, "dl_selected_predictions.csv"))
dl_validation_candidates <- read.csv(file.path(output_dir, "dl_validation_candidates.csv"))

stopifnot(length(unique(selected_dl_models$Ticker)) == 33,
          nrow(selected_dl_models) == 33,
          length(unique(dl_all_results$Ticker)) == 33,
          nrow(dl_all_results) == 99)

saveRDS(selected_dl_models, "FINAL_DL_selected_models.rds")
saveRDS(selected_dl_predictions, "FINAL_DL_selected_predictions.rds")
saveRDS(dl_all_results, "FINAL_DL_all_architecture_results.rds")
saveRDS(dl_all_predictions, "FINAL_DL_all_architecture_predictions.rds")
saveRDS(dl_validation_candidates, "FINAL_DL_validation_candidates.rds")

cat("DONE: DL results imported and saved in R.\n")
print(table(selected_dl_models$Model))
print(selected_dl_models[, c("Ticker", "Model", "ValidationRMSE", "RMSEGain")])
