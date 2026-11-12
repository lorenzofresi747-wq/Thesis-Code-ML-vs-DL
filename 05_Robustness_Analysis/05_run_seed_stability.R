# ========================================================================
# MASTER'S THESIS -- LAUNCH PREDECLARED MULTI-SEED ROBUSTNESS (EXTERNAL PY)
# Keep the original model selected using validation RMSE. NO retuning.
# Requires dl_input_33.csv, dl_output/dl_selected_models.csv and
# train_dl_33.py produced by the finished 33-stock run.
# Place 05_seed_stability.py in the project folder (or choose its location).
# ========================================================================
python_exe <- file.path(Sys.getenv('LOCALAPPDATA'),
                        'thesis-dl-torch', 'Scripts', 'python.exe')
if (!file.exists(python_exe)) stop('Could not find separate PyTorch Python environment.')

get_path <- function(expected, label) {
  if (file.exists(expected)) return(normalizePath(expected, mustWork = TRUE))
  cat('Select ', label, ' using the file dialog.\n', sep = '')
  normalizePath(file.choose(), mustWork = TRUE)
}

input_file <- get_path(file.path(getwd(), 'dl_input_33.csv'), 'dl_input_33.csv')
selected_file <- get_path(file.path(getwd(), 'dl_output', 'dl_selected_models.csv'),
                          'dl_selected_models.csv')
original_trainer <- get_path(file.path(getwd(), 'train_dl_33.py'), 'train_dl_33.py')
seed_script <- get_path(file.path(getwd(), '05_seed_stability.py'), '05_seed_stability.py')
output_folder <- file.path(getwd(), 'FINAL_DL_vs_ML', 'seed_stability')
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

# NEW seeds declared before evaluating new test predictions.
# 3 additional fits per stock + original fitted model = 4 realizations/stock.
extra_seeds <- c(101L, 202L, 303L)
cat('Executing a fixed-architecture seed stability experiment across 33 stocks.\n')
cat('Additional seeds:', paste(extra_seeds, collapse = ', '), '\n')

status <- system2(
  command = python_exe,
  args = c(shQuote(seed_script),
           '--input', shQuote(input_file),
           '--selected', shQuote(selected_file),
           '--training-script', shQuote(original_trainer),
           '--output', shQuote(output_folder),
           '--seeds', paste(extra_seeds, collapse = ','),
           '--epochs', '80', '--patience', '10', '--batch-size', '128'),
  stdout = '', stderr = '', wait = TRUE
)
if (!identical(as.integer(status), 0L)) {
  stop('PyTorch seed-stability process failed. Earlier saved runs remain available.')
}
cat('Multi-seed training finished. Now run 06_analyse_seed_stability.R.\n')
