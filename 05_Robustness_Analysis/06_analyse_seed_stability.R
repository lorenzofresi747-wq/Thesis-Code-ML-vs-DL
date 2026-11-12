# ========================================================================
# MASTER'S THESIS -- MULTI-SEED ROBUSTNESS ANALYSIS IN R
# Uses reruns from 05_seed_stability.py + original selected-model forecasts.
# Main outputs: stability per stock, per-seed overall summary, and
# HIGH-vs-LOW volatility regime contrast for each new random seed.
# ========================================================================
required_packages <- c('dplyr', 'tidyr')
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                     logical(1), quietly = TRUE)]
if (length(missing)) stop('Install these packages: ', paste(missing, collapse = ', '))
suppressPackageStartupMessages({library(dplyr);library(tidyr)})

root <- file.path(getwd(), 'FINAL_DL_vs_ML')
seed_dir <- file.path(root, 'seed_stability')
score_file <- file.path(seed_dir, 'dl_seed_stability_scores.csv')
pred_file <- file.path(seed_dir, 'dl_seed_stability_predictions.csv')
if (!file.exists(score_file) || !file.exists(pred_file)) {
  stop('First finish 05_run_seed_stability.R; seed-stability CSVs are missing.')
}
if (!exists('stock_scores') || !exists('regime_stock')) {
  saved <- file.path(root, 'FINAL_DL_vs_ML_tables.rds')
  if (!file.exists(saved)) stop('Missing main DL-vs-ML saved analysis tables.')
  old <- readRDS(saved)
  stock_scores <- old$stock_scores
  regime_stock <- old$regime_stock
}
if (!exists('regime_daily')) {
  dailypath <- file.path(root, 'FINAL_ML_DL_daily_matched.rds')
  if (!file.exists(dailypath) || !exists('model_data') || !exists('volatility_thresholds')) {
    stop('Restore regime_daily, or load model_data and volatility_thresholds plus the saved daily RDS.')
  }
  daily <- readRDS(dailypath)
  states <- model_data %>% filter(Sample == 'Test') %>%
    transmute(Ticker = as.character(Ticker),
              ForecastOrigin = as.Date(Date), TargetDate = as.Date(TargetDate),
              Vol20 = as.numeric(Vol20)) %>%
    left_join(volatility_thresholds %>%
                transmute(Ticker = as.character(Ticker),
                          Vol20_q33 = as.numeric(Vol20_q33), Vol20_q67 = as.numeric(Vol20_q67)),
              by = 'Ticker') %>%
    mutate(VolatilityState = case_when(
      Vol20 < Vol20_q33 ~ 'Low', Vol20 < Vol20_q67 ~ 'Medium', TRUE ~ 'High')) %>%
    select(Ticker, ForecastOrigin, TargetDate, VolatilityState)
  regime_daily <- daily %>% left_join(states, by = c('Ticker','ForecastOrigin','TargetDate'))
}

new_scores <- read.csv(score_file, stringsAsFactors = FALSE)
new_preds <- read.csv(pred_file, stringsAsFactors = FALSE)
stopifnot(length(unique(new_scores$Ticker)) == 33L,
          nrow(new_scores) == 33 * 3L,
          nrow(new_preds) == 33 * 3 * 1508L,
          !anyDuplicated(new_scores[, c('Ticker','Seed')]),
          !anyDuplicated(new_preds[, c('Ticker','Seed','ForecastOrigin','TargetDate')]))

baseline <- stock_scores %>%
  transmute(Ticker = as.character(Ticker), RMSE_ML = as.numeric(RMSE_ML),
            RMSE_DL = as.numeric(RMSE_DL),
            DLGain_RMSE = as.numeric(DLGain_RMSE))

new_score_comparison <- new_scores %>%
  transmute(Ticker = as.character(Ticker), SeedLabel = paste0('Seed_', Seed),
            Architecture = as.character(Model),
            RMSE_DL = as.numeric(RMSE_DL),
            ValidationRMSE = as.numeric(ValidationRMSE)) %>%
  left_join(baseline %>% select(Ticker, RMSE_ML), by = 'Ticker') %>%
  mutate(DLGain_RMSE = 100 * (1 - RMSE_DL / RMSE_ML))
original_score <- baseline %>%
  transmute(Ticker, SeedLabel = 'Original', Architecture = NA_character_,
            RMSE_DL, ValidationRMSE = NA_real_, RMSE_ML, DLGain_RMSE)
all_seed_scores <- bind_rows(original_score, new_score_comparison)
stopifnot(nrow(all_seed_scores) == 33L * 4L)

seed_stock_summary <- all_seed_scores %>%
  group_by(Ticker) %>%
  summarise(N_Runs = n(), Original_DLGain = DLGain_RMSE[SeedLabel == 'Original'],
            Mean_DLGain = mean(DLGain_RMSE),
            Median_DLGain = median(DLGain_RMSE),
            SD_DLGain = sd(DLGain_RMSE),
            Min_DLGain = min(DLGain_RMSE), Max_DLGain = max(DLGain_RMSE),
            Fraction_DL_Wins = mean(DLGain_RMSE > 0),
            All_Seeds_Agree_With_Original = all((DLGain_RMSE > 0) ==
                                               (DLGain_RMSE[SeedLabel == 'Original'] > 0)),
            .groups = 'drop') %>% arrange(desc(SD_DLGain))
seed_overall_summary <- all_seed_scores %>%
  group_by(SeedLabel) %>%
  summarise(N_Stocks = n(), Stocks_DL_Better = sum(DLGain_RMSE > 0),
            Mean_DLGain = mean(DLGain_RMSE),
            Median_DLGain = median(DLGain_RMSE),
            .groups = 'drop') %>%
  mutate(SeedLabel = factor(SeedLabel, levels = c('Original','Seed_101','Seed_202','Seed_303'))) %>%
  arrange(SeedLabel)

# ----- TEST ORIGINAL HIGH-vs-LOW REGIME CONTRAST PER NEW SEED -----------
new_preds <- new_preds %>%
  mutate(Ticker = as.character(Ticker),
         ForecastOrigin = as.Date(ForecastOrigin),
         TargetDate = as.Date(TargetDate), Seed = as.integer(Seed))
ref <- regime_daily %>%
  transmute(Ticker = as.character(Ticker),
            ForecastOrigin = as.Date(ForecastOrigin), TargetDate = as.Date(TargetDate),
            OriginalActual = as.numeric(ActualReturn),
            PredML = as.numeric(PredML),
            VolatilityState = as.character(VolatilityState))
joined <- new_preds %>%
  inner_join(ref, by = c('Ticker', 'ForecastOrigin', 'TargetDate'),
             relationship = 'many-to-one')
if (nrow(joined) != 33L * 3L * 1508L ||
    max(abs(joined$ActualReturn - joined$OriginalActual)) > 1e-6 ||
    anyNA(joined)) {
  stop('Seed predictions and original ML forecasts do not match exactly.')
}

new_regime <- joined %>%
  group_by(Seed, Ticker, VolatilityState) %>%
  summarise(SS_ML = sum((OriginalActual - PredML)^2),
            SS_DL = sum((OriginalActual - Prediction)^2),
            .groups = 'drop') %>%
  mutate(DLGain = 100 * (1 - sqrt(SS_DL / SS_ML))) %>%
  select(Seed, Ticker, VolatilityState, DLGain) %>%
  pivot_wider(names_from = VolatilityState, values_from = DLGain) %>%
  mutate(High_minus_Low = High - Low)
stopifnot(nrow(new_regime) == 99L, !anyNA(new_regime))

original_regime <- regime_stock %>%
  transmute(Ticker = as.character(Ticker),
            VolatilityState = as.character(VolatilityState),
            DLGain = as.numeric(DLGain_RMSE)) %>%
  pivot_wider(names_from = VolatilityState, values_from = DLGain) %>%
  mutate(High_minus_Low = High - Low)

seed_regime_summary <- bind_rows(
  original_regime %>%
    summarise(SeedLabel = 'Original', Mean_High_Low_pp = mean(High_minus_Low),
              Median_High_Low_pp = median(High_minus_Low),
              Stocks_Positive = sum(High_minus_Low > 0),
              Mean_High_DLGain = mean(High), Mean_Low_DLGain = mean(Low)),
  new_regime %>% group_by(Seed) %>%
    summarise(SeedLabel = paste0('Seed_', first(Seed)),
              Mean_High_Low_pp = mean(High_minus_Low),
              Median_High_Low_pp = median(High_minus_Low),
              Stocks_Positive = sum(High_minus_Low > 0),
              Mean_High_DLGain = mean(High), Mean_Low_DLGain = mean(Low),
              .groups = 'drop') %>% select(-Seed)
)

# Clear descriptive stability figures; do not use seed-to-seed t tests.
cat('\n=== OVERALL DL GAIN BY RANDOM SEED ===\n'); print(seed_overall_summary, width = Inf)
cat('\n=== HIGH VOLATILITY MINUS LOW VOLATILITY, BY RANDOM SEED ===\n')
print(seed_regime_summary, width = Inf)
cat('\n=== MOST SEED-SENSITIVE STOCKS ===\n')
print(head(seed_stock_summary, 10), width = Inf)

write.csv(seed_stock_summary, file.path(seed_dir, 'seed_01_stock_stability.csv'), row.names=FALSE)
write.csv(seed_overall_summary, file.path(seed_dir, 'seed_02_overall_stability.csv'), row.names=FALSE)
write.csv(new_regime, file.path(seed_dir, 'seed_03_stock_regime_deltas.csv'), row.names=FALSE)
write.csv(seed_regime_summary, file.path(seed_dir, 'seed_04_regime_stability.csv'), row.names=FALSE)
write.csv(all_seed_scores, file.path(seed_dir, 'seed_05_all_seed_scores.csv'), row.names=FALSE)
saveRDS(list(stock_stability=seed_stock_summary, overall=seed_overall_summary,
             regime_summary=seed_regime_summary, regime_stock=new_regime),
        file.path(seed_dir, 'FINAL_DL_seed_stability_tables.rds'))
cat('\nSaved seed stability tables in: ', seed_dir, '\n', sep='')
cat('Notes: same DL architecture/hyperparameters per stock across all seeds.\n')
