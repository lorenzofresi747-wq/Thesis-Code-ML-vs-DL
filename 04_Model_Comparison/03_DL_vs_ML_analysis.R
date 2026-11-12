# =====================================================================
# MASTER'S THESIS - FINAL DL VS ML EMPIRICAL ANALYSIS
# 33 US equities | 2010-2017 train | 2018-2019 validation | 2020-2025 test
#
# REQUIRES EXISTING RESULTS. DOES NOT RETRAIN OR MODIFY MODELS.
# Outputs: direct paired comparisons, DM/PT tests, structural groups,
# time-varying volatility regimes, HC3 regressions, Spearman correlations.
# =====================================================================

# ----------------------------- PACKAGES ------------------------------
needed_pkgs <- c("dplyr", "tibble", "forecast", "lmtest", "sandwich")
missing_pkgs <- needed_pkgs[!vapply(needed_pkgs, requireNamespace,
                                    logical(1), quietly = TRUE)]
if (length(missing_pkgs)) {
  stop("Install these R packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_pkgs), collapse = ", "), "))")
}
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

# ------------------------ IMPORT SAVED OBJECTS ------------------------
# The objects should already be loaded by 02_DL_run_all_33.R and your ML
# workspace. If one is missing, select its corresponding CSV or RDS file.
load_if_needed <- function(object_name) {
  if (exists(object_name, envir = .GlobalEnv, inherits = FALSE)) return(invisible(NULL))
  message("Select the saved CSV/RDS for object: ", object_name)
  f <- file.choose()
  ext <- tolower(tools::file_ext(f))
  obj <- if (ext == "rds") readRDS(f) else if (ext == "csv") {
    read.csv(f, stringsAsFactors = FALSE, check.names = FALSE)
  } else stop("Expected a .csv or .rds for: ", object_name)
  assign(object_name, obj, envir = .GlobalEnv)
}

for (nm in c("selected_ml_predictions", "selected_dl_predictions",
             "selected_ml_models", "selected_dl_models",
             "stock_characteristics", "model_data", "volatility_thresholds")) {
  load_if_needed(nm)
}

out_dir <- file.path(getwd(), "FINAL_DL_vs_ML")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------- STRICT MATCHING CHECKS -----------------------
keys <- c("Ticker", "ForecastOrigin", "TargetDate")
required_pred <- c(keys, "ActualReturn", "Prediction", "Model")
stopifnot(all(required_pred %in% names(selected_ml_predictions)),
          all(required_pred %in% names(selected_dl_predictions)))

ml <- selected_ml_predictions %>%
  transmute(Ticker = as.character(Ticker),
            ForecastOrigin = as.Date(ForecastOrigin),
            TargetDate = as.Date(TargetDate),
            Actual_ML = as.numeric(ActualReturn),
            PredML = as.numeric(Prediction),
            ML_Model = as.character(Model))

dl <- selected_dl_predictions %>%
  transmute(Ticker = as.character(Ticker),
            ForecastOrigin = as.Date(ForecastOrigin),
            TargetDate = as.Date(TargetDate),
            Actual_DL = as.numeric(ActualReturn),
            PredDL = as.numeric(Prediction),
            DL_Model = as.character(Model))

if (anyNA(ml) || anyNA(dl)) stop("Missing values in ML/DL forecasts.")
if (anyDuplicated(ml[, keys]) || anyDuplicated(dl[, keys])) {
  stop("Duplicated (Ticker, ForecastOrigin, TargetDate) detected.")
}
if (nrow(ml) != 49764L || nrow(dl) != 49764L ||
    length(unique(ml$Ticker)) != 33L || length(unique(dl$Ticker)) != 33L) {
  stop("Expected 49,764 predictions and 33 stocks in EACH model family.")
}
if (nrow(anti_join(ml, dl, by = keys)) ||
    nrow(anti_join(dl, ml, by = keys))) {
  stop("ML and DL have different forecast dates or stocks. No comparison made.")
}

daily <- inner_join(ml, dl, by = keys, relationship = "one-to-one")
if (nrow(daily) != 49764L) stop("Unexpected row count after matching.")

actual_diff <- max(abs(daily$Actual_ML - daily$Actual_DL))
cat("Maximum difference in realized returns:", signif(actual_diff, 5), "\n")
if (!is.finite(actual_diff) || actual_diff > 1e-6) {
  stop("Realized returns differ across ML/DL. Check the exported data.")
}
# Keep the original ML realized returns for BOTH loss functions.
daily <- daily %>%
  transmute(Ticker, ForecastOrigin, TargetDate,
            ActualReturn = Actual_ML, PredML, PredDL,
            ML_Model, DL_Model)

# Independent model-selection check: one preferred model per stock.
ml_specs <- selected_ml_models %>%
  transmute(Ticker = as.character(Ticker),
            ML_Model_Chosen = as.character(Model),
            ML_ValidationRMSE = as.numeric(ValidationRMSE))
dl_specs <- selected_dl_models %>%
  transmute(Ticker = as.character(Ticker),
            DL_Model_Chosen = as.character(Model),
            DL_ValidationRMSE = as.numeric(ValidationRMSE))
if (anyDuplicated(ml_specs$Ticker) || anyDuplicated(dl_specs$Ticker)) {
  stop("More than one selected ML/DL model per ticker.")
}
daily <- daily %>%
  left_join(ml_specs, by = "Ticker", relationship = "many-to-one") %>%
  left_join(dl_specs, by = "Ticker", relationship = "many-to-one")
if (anyNA(daily[, c("ML_Model_Chosen", "DL_Model_Chosen")]) ||
    any(daily$ML_Model != daily$ML_Model_Chosen) ||
    any(daily$DL_Model != daily$DL_Model_Chosen)) {
  stop("Daily model labels do not match the validation-selected models.")
}
if (any(table(daily$Ticker) != 1508L)) {
  stop("Expected exactly 1,508 paired test dates for each stock.")
}

char_required <- c("Ticker", "sector", "VolatilityGroup", "LiquidityGroup",
                   "TrainingVolatility", "TrainingILLIQ")
stopifnot(all(char_required %in% names(stock_characteristics)))
characteristics <- stock_characteristics %>%
  transmute(Ticker = as.character(Ticker),
            sector = as.character(sector),
            VolatilityGroup = as.character(VolatilityGroup),
            LiquidityGroup = as.character(LiquidityGroup),
            TrainingVolatility = as.numeric(TrainingVolatility),
            TrainingILLIQ = as.numeric(TrainingILLIQ))
if (anyDuplicated(characteristics$Ticker)) stop("Duplicate stock characteristics.")

# ------------------ EXACT PAIRED FORECAST METRICS ---------------------
metrics <- function(actual, pred_ml, pred_dl) {
  rmse <- function(x) sqrt(mean((actual - x)^2))
  mae  <- function(x) mean(abs(actual - x))
  rmse_m <- rmse(pred_ml); rmse_d <- rmse(pred_dl); rmse_z <- rmse(0)
  mae_m  <- mae(pred_ml);  mae_d  <- mae(pred_dl);  mae_z  <- mae(0)
  tibble(
    N = length(actual),
    RMSE_ML = rmse_m, RMSE_DL = rmse_d, RMSE_Zero = rmse_z,
    MAE_ML = mae_m, MAE_DL = mae_d, MAE_Zero = mae_z,
    RMSEGain_ML = 100 * (1 - rmse_m / rmse_z),
    RMSEGain_DL = 100 * (1 - rmse_d / rmse_z),
    MAEGain_ML = 100 * (1 - mae_m / mae_z),
    MAEGain_DL = 100 * (1 - mae_d / mae_z),
    DLGain_RMSE = 100 * (1 - rmse_d / rmse_m),
    DLGain_MAE = 100 * (1 - mae_d / mae_m),
    R2_ML = 1 - rmse_m^2 / rmse_z^2,
    R2_DL = 1 - rmse_d^2 / rmse_z^2,
    DA_ML = 100 * mean(actual * pred_ml > 0),
    DA_DL = 100 * mean(actual * pred_dl > 0)
  )
}

stock_scores <- daily %>%
  group_by(Ticker) %>%
  group_modify(~ metrics(.x$ActualReturn, .x$PredML, .x$PredDL)) %>%
  ungroup() %>%
  left_join(characteristics, by = "Ticker", relationship = "many-to-one") %>%
  left_join(ml_specs, by = "Ticker", relationship = "many-to-one") %>%
  left_join(dl_specs, by = "Ticker", relationship = "many-to-one") %>%
  mutate(Winner_RMSE = if_else(DLGain_RMSE > 0, "DL", "ML"),
         Winner_MAE = if_else(DLGain_MAE > 0, "DL", "ML")) %>%
  arrange(desc(DLGain_RMSE))
if (nrow(stock_scores) != 33L || anyNA(stock_scores)) {
  stop("Stock-level aggregation/join failed.")
}
# Reconcile both sides with their independently saved official test scores.
reference_ml <- selected_ml_models %>%
  transmute(Ticker = as.character(Ticker), Reference_RMSE_ML = as.numeric(RMSE))
reference_dl <- selected_dl_models %>%
  transmute(Ticker = as.character(Ticker), Reference_RMSE_DL = as.numeric(RMSE))
score_check <- stock_scores %>%
  left_join(reference_ml, by = "Ticker", relationship = "one-to-one") %>%
  left_join(reference_dl, by = "Ticker", relationship = "one-to-one")
if (anyNA(score_check[, c("Reference_RMSE_ML", "Reference_RMSE_DL")]) ||
    max(abs(score_check$RMSE_ML - score_check$Reference_RMSE_ML)) > 1e-6 ||
    max(abs(score_check$RMSE_DL - score_check$Reference_RMSE_DL)) > 1e-6) {
  stop("Recomputed RMSE differs from the frozen selected ML/DL reports.")
}

cat("\n=== DIRECT COMPARISON ===\n")
cat("Stocks favoring DL by RMSE:", sum(stock_scores$DLGain_RMSE > 0), "/ 33\n")
cat("Mean DL gain (%):", mean(stock_scores$DLGain_RMSE), "\n")
cat("Median DL gain (%):", median(stock_scores$DLGain_RMSE), "\n")
print(stock_scores %>%
        select(Ticker, sector, ML_Model_Chosen, DL_Model_Chosen,
               RMSEGain_ML, RMSEGain_DL, DLGain_RMSE, DLGain_MAE),
      n = 33)

# ----------------- PAIRED DIEBOLD-MARIANO TEST ------------------------
# Difference is LOSS_DL - LOSS_ML. Negative means DL has lower loss.
# Two-sided H0: equal expected loss. Horizon h=1 (one-day-ahead).
dm_one <- function(actual, pred_ml, pred_dl, power) {
  e_dl <- actual - pred_dl
  e_ml <- actual - pred_ml
  diff <- mean(abs(e_dl)^power - abs(e_ml)^power)
  fit <- suppressWarnings(tryCatch(
    forecast::dm.test(e1 = e_dl, e2 = e_ml, h = 1,
                      power = power, alternative = "two.sided"),
    error = function(e) NULL
  ))
  stat <- if (is.null(fit)) NA_real_ else as.numeric(fit$statistic)
  pval <- if (is.null(fit)) NA_real_ else as.numeric(fit$p.value)
  tibble(LossDifference = diff,
         DM_statistic = ifelse(is.finite(stat), stat, NA_real_),
         DM_pvalue = ifelse(is.finite(pval), pval, NA_real_))
}

dm_results <- daily %>%
  group_by(Ticker) %>%
  group_modify(~ {
    sq <- dm_one(.x$ActualReturn, .x$PredML, .x$PredDL, 2)
    ab <- dm_one(.x$ActualReturn, .x$PredML, .x$PredDL, 1)
    tibble(DM_Sq_LossDiff = sq$LossDifference,
           DM_Sq_Statistic = sq$DM_statistic,
           DM_Sq_pvalue = sq$DM_pvalue,
           DM_Abs_LossDiff = ab$LossDifference,
           DM_Abs_Statistic = ab$DM_statistic,
           DM_Abs_pvalue = ab$DM_pvalue)
  }) %>%
  ungroup() %>%
  mutate(
    DM_Sq_pFDR = p.adjust(DM_Sq_pvalue, method = "BH"),
    DM_Abs_pFDR = p.adjust(DM_Abs_pvalue, method = "BH"),
    DL_Better_Sq = DM_Sq_LossDiff < 0,
    DL_Better_Abs = DM_Abs_LossDiff < 0,
    DL_SigBetter_Sq_5pct = DL_Better_Sq & DM_Sq_pvalue < .05,
    DL_SigBetter_Abs_5pct = DL_Better_Abs & DM_Abs_pvalue < .05,
    DL_SigBetter_Sq_FDR = DL_Better_Sq & DM_Sq_pFDR < .05,
    DL_SigBetter_Abs_FDR = DL_Better_Abs & DM_Abs_pFDR < .05
  )

stock_scores <- stock_scores %>%
  left_join(dm_results, by = "Ticker", relationship = "one-to-one")

# Separate question: Does DL outperform the zero-return benchmark?
dm_dl_zero <- daily %>%
  group_by(Ticker) %>%
  group_modify(~ {
    zeros <- rep(0, nrow(.x))
    sq <- dm_one(.x$ActualReturn, zeros, .x$PredDL, 2)
    ab <- dm_one(.x$ActualReturn, zeros, .x$PredDL, 1)
    tibble(DLZero_Sq_LossDiff = sq$LossDifference,
           DLZero_Sq_Statistic = sq$DM_statistic,
           DLZero_Sq_pvalue = sq$DM_pvalue,
           DLZero_Abs_LossDiff = ab$LossDifference,
           DLZero_Abs_Statistic = ab$DM_statistic,
           DLZero_Abs_pvalue = ab$DM_pvalue)
  }) %>%
  ungroup() %>%
  mutate(DLZero_Sq_pFDR = p.adjust(DLZero_Sq_pvalue, "BH"),
         DLZero_Abs_pFDR = p.adjust(DLZero_Abs_pvalue, "BH"),
         DL_BeatsZero_Sq_FDR = DLZero_Sq_LossDiff < 0 &
           DLZero_Sq_pFDR < .05,
         DL_BeatsZero_Abs_FDR = DLZero_Abs_LossDiff < 0 &
           DLZero_Abs_pFDR < .05)

cat("\n=== DM: DL BETTER THAN ML (5% LEVEL) ===\n")
cat("Squared loss, nominal:", sum(stock_scores$DL_SigBetter_Sq_5pct, na.rm=TRUE),
    " | FDR:", sum(stock_scores$DL_SigBetter_Sq_FDR, na.rm=TRUE), "\n")
cat("Absolute loss, nominal:", sum(stock_scores$DL_SigBetter_Abs_5pct, na.rm=TRUE),
    " | FDR:", sum(stock_scores$DL_SigBetter_Abs_FDR, na.rm=TRUE), "\n")

# ---------------------- DL DIRECTIONAL TEST --------------------------
# Same one-sided Pesaran-Timmermann approximation used in the ML stage.
# Constant sign forecasts cannot be tested. This is supplementary to DM.
pt_one <- function(actual, pred) {
  ok <- is.finite(actual) & is.finite(pred) & actual != 0 & pred != 0
  actual <- actual[ok]; pred <- pred[ok]
  n <- length(actual)
  if (n < 10L) return(tibble(PT_N = n, ExpectedDA = NA_real_,
                            PT_stat = NA_real_, PT_pvalue = NA_real_))
  x <- actual > 0; y <- pred > 0
  p_x <- mean(x); p_y <- mean(y)
  p_hit <- mean(x == y)
  p_expected <- p_x * p_y + (1 - p_x) * (1 - p_y)
  if (length(unique(x)) < 2L || length(unique(y)) < 2L) {
    return(tibble(PT_N = n, ExpectedDA = 100*p_expected,
                  PT_stat = NA_real_, PT_pvalue = NA_real_))
  }
  v_hit <- p_expected * (1 - p_expected) / n
  v_expected <- (2*p_y - 1)^2 * p_x * (1-p_x) / n +
    (2*p_x - 1)^2 * p_y * (1-p_y) / n +
    4*p_x*p_y*(1-p_x)*(1-p_y)/n^2
  v_diff <- v_hit - v_expected
  if (!is.finite(v_diff) || v_diff <= 0) {
    return(tibble(PT_N = n, ExpectedDA = 100*p_expected,
                  PT_stat = NA_real_, PT_pvalue = NA_real_))
  }
  statistic <- (p_hit - p_expected) / sqrt(v_diff)
  tibble(PT_N = n, ExpectedDA = 100*p_expected,
         PT_stat = statistic, PT_pvalue = 1 - pnorm(statistic))
}

pt_dl <- daily %>%
  group_by(Ticker) %>%
  group_modify(~ pt_one(.x$ActualReturn, .x$PredDL)) %>%
  ungroup() %>%
  mutate(PT_pFDR = p.adjust(PT_pvalue, method = "BH"),
         PT_Significant_5pct = PT_pvalue < 0.05,
         PT_Significant_FDR = PT_pFDR < 0.05)

# ------------------------- STRUCTURAL GROUPS ------------------------
structural_summary <- function(df, column) {
  df %>%
    group_by(across(all_of(column))) %>%
    summarise(
      N_Stocks = n(),
      Mean_DLGain = mean(DLGain_RMSE),
      Median_DLGain = median(DLGain_RMSE),
      SD_DLGain = sd(DLGain_RMSE),
      DL_Better = sum(DLGain_RMSE > 0),
      Percent_DL_Better = 100 * mean(DLGain_RMSE > 0),
      Mean_DLGain_MAE = mean(DLGain_MAE),
      Mean_RMSEGain_ML = mean(RMSEGain_ML),
      Mean_RMSEGain_DL = mean(RMSEGain_DL),
      Mean_DA_ML = mean(DA_ML),
      Mean_DA_DL = mean(DA_DL),
      DL_SigBetter_DM_Sq = sum(DL_SigBetter_Sq_5pct, na.rm = TRUE),
      DL_SigBetter_DM_Sq_FDR = sum(DL_SigBetter_Sq_FDR, na.rm = TRUE),
      .groups = "drop")
}

table_volatility <- structural_summary(stock_scores, "VolatilityGroup") %>%
  mutate(VolatilityGroup = factor(VolatilityGroup,
                                   levels = c("Low", "Medium", "High"))) %>%
  arrange(VolatilityGroup)
table_liquidity <- structural_summary(stock_scores, "LiquidityGroup") %>%
  mutate(LiquidityGroup = factor(LiquidityGroup,
                                  levels = c("High", "Medium", "Low"))) %>%
  arrange(LiquidityGroup)
table_sector <- structural_summary(stock_scores, "sector") %>%
  arrange(desc(Mean_DLGain))

# ----------------------- WITHIN-STOCK VOL REGIMES --------------------
stopifnot(all(c("Ticker", "Date", "TargetDate", "Sample", "Vol20") %in%
              names(model_data)))
stopifnot(all(c("Ticker", "Vol20_q33", "Vol20_q67") %in%
              names(volatility_thresholds)))
regime_data <- model_data %>%
  filter(Sample == "Test") %>%
  transmute(Ticker = as.character(Ticker),
            ForecastOrigin = as.Date(Date), TargetDate = as.Date(TargetDate),
            Vol20 = as.numeric(Vol20)) %>%
  left_join(volatility_thresholds %>%
              transmute(Ticker = as.character(Ticker),
                        Vol20_q33 = as.numeric(Vol20_q33),
                        Vol20_q67 = as.numeric(Vol20_q67)),
            by = "Ticker", relationship = "many-to-one") %>%
  mutate(VolatilityState = case_when(
    Vol20 < Vol20_q33 ~ "Low",
    Vol20 < Vol20_q67 ~ "Medium",
    TRUE ~ "High"))
if (anyNA(regime_data) || anyDuplicated(regime_data[, keys])) {
  stop("Invalid volatility state data.")
}

regime_daily <- daily %>%
  left_join(regime_data %>% select(all_of(keys), VolatilityState),
            by = keys, relationship = "one-to-one")
if (anyNA(regime_daily$VolatilityState)) {
  stop("Some DL/ML test dates have no volatility regime.")
}
regime_stock <- regime_daily %>%
  group_by(Ticker, VolatilityState) %>%
  group_modify(~ metrics(.x$ActualReturn, .x$PredML, .x$PredDL)) %>%
  ungroup()
if (nrow(regime_stock) != 99L) {
  stop("Expected 33 stocks x 3 volatility states = 99 rows.")
}
table_regimes <- regime_stock %>%
  group_by(VolatilityState) %>%
  summarise(N_Stocks = n_distinct(Ticker),
            Mean_DLGain = mean(DLGain_RMSE),
            Median_DLGain = median(DLGain_RMSE),
            SD_DLGain = sd(DLGain_RMSE),
            Stocks_DL_Better = sum(DLGain_RMSE > 0),
            Percent_DL_Better = 100 * mean(DLGain_RMSE > 0),
            Mean_ML_RMSEGain = mean(RMSEGain_ML),
            Mean_DL_RMSEGain = mean(RMSEGain_DL),
            Mean_DA_ML = mean(DA_ML),
            Mean_DA_DL = mean(DA_DL),
            .groups = "drop") %>%
  mutate(VolatilityState = factor(VolatilityState,
                                   levels = c("Low", "Medium", "High"))) %>%
  arrange(VolatilityState)

# -------------------- CONTINUOUS HC3 + SPEARMAN ----------------------
empirical_dl <- stock_scores %>%
  filter(is.finite(TrainingVolatility), is.finite(TrainingILLIQ),
         TrainingILLIQ > 0, is.finite(DLGain_RMSE)) %>%
  mutate(LogILLIQ = log(TrainingILLIQ),
         Volatility_z = as.numeric(scale(TrainingVolatility)),
         LogILLIQ_z = as.numeric(scale(LogILLIQ)))
if (nrow(empirical_dl) != 33L) stop("Expected N=33 for the regression.")

reg_vol <- lm(DLGain_RMSE ~ Volatility_z, data = empirical_dl)
reg_illiq <- lm(DLGain_RMSE ~ LogILLIQ_z, data = empirical_dl)
reg_joint <- lm(DLGain_RMSE ~ Volatility_z + LogILLIQ_z,
                data = empirical_dl)

extract_hc3 <- function(model, model_label) {
  tab <- as.matrix(lmtest::coeftest(
    model, vcov. = sandwich::vcovHC(model, type = "HC3")))
  tibble(Model = model_label, Term = rownames(tab),
         Estimate = unname(tab[, 1]), Robust_SE = unname(tab[, 2]),
         t_value = unname(tab[, 3]), p_value = unname(tab[, 4]),
         N = nobs(model), R2 = summary(model)$r.squared,
         Adjusted_R2 = summary(model)$adj.r.squared)
}
table_hc3 <- bind_rows(
  extract_hc3(reg_vol, "Volatility only"),
  extract_hc3(reg_illiq, "Illiquidity only"),
  extract_hc3(reg_joint, "Volatility + Illiquidity")
) %>% mutate(Significant_5pct = p_value < .05,
             Significant_10pct = p_value < .10)

sp_vol <- suppressWarnings(cor.test(empirical_dl$TrainingVolatility,
                                    empirical_dl$DLGain_RMSE,
                                    method = "spearman", exact = FALSE))
sp_illiq <- suppressWarnings(cor.test(empirical_dl$LogILLIQ,
                                      empirical_dl$DLGain_RMSE,
                                      method = "spearman", exact = FALSE))
table_spearman <- tibble(
  Characteristic = c("Training volatility", "Log Amihud illiquidity"),
  Spearman_rho = c(unname(sp_vol$estimate), unname(sp_illiq$estimate)),
  p_value = c(sp_vol$p.value, sp_illiq$p.value)
)

# ----------- CHECK WHETHER THE CROSS-SECTION HAS OUTLIERS ------------
influence_dl <- empirical_dl %>%
  mutate(CooksDistance = cooks.distance(reg_joint),
         StudentizedResidual = rstudent(reg_joint)) %>%
  select(Ticker, sector, DLGain_RMSE, TrainingVolatility,
         TrainingILLIQ, CooksDistance, StudentizedResidual) %>%
  arrange(desc(CooksDistance))

# --------------------------- SAVE EVERYTHING -------------------------
write_csv_safe <- function(df, filename) {
  write.csv(df, file.path(out_dir, filename), row.names = FALSE, na = "")
}
write_csv_safe(stock_scores, "01_stock_level_DLGain_and_DM.csv")
write_csv_safe(dm_dl_zero, "02_DL_vs_zero_DM.csv")
write_csv_safe(pt_dl, "03_DL_PT_directional_tests.csv")
write_csv_safe(table_volatility, "04_DLGain_structural_volatility.csv")
write_csv_safe(table_liquidity, "05_DLGain_liquidity.csv")
write_csv_safe(table_sector, "06_DLGain_sector.csv")
write_csv_safe(regime_stock, "07_DLGain_regime_by_stock.csv")
write_csv_safe(table_regimes, "08_DLGain_regime_summary.csv")
write_csv_safe(table_hc3, "09_DLGain_HC3_regressions.csv")
write_csv_safe(table_spearman, "10_DLGain_Spearman.csv")
write_csv_safe(influence_dl, "11_DLGain_influence_diagnostics.csv")
# Optional figure for thesis: paired DL advantage for all 33 stocks.
if (requireNamespace("ggplot2", quietly = TRUE)) {
  fig <- ggplot2::ggplot(stock_scores,
                         ggplot2::aes(x = reorder(Ticker, DLGain_RMSE),
                                      y = DLGain_RMSE,
                                      fill = DLGain_RMSE > 0)) +
    ggplot2::geom_col(show.legend = FALSE) +
    ggplot2::coord_flip() +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.3) +
    ggplot2::labs(title = "Validation-selected DL versus ML",
                  subtitle = "Positive values indicate lower RMSE for DL",
                  x = "Stock", y = "DLGain RMSE (%)") +
    ggplot2::theme_minimal(base_size = 11)
  ggplot2::ggsave(file.path(out_dir, "DLGain_by_stock.pdf"),
                  fig, width = 8, height = 9)
}
# Save the matched daily predictions for exact paired checks and future work.
saveRDS(daily, file.path(out_dir, "FINAL_ML_DL_daily_matched.rds"))
saveRDS(list(stock_scores = stock_scores, dm = dm_results,
             dm_dl_zero = dm_dl_zero, pt_dl = pt_dl, volatility = table_volatility,
             liquidity = table_liquidity, sector = table_sector,
             regime_stock = regime_stock, regime_summary = table_regimes,
             hc3 = table_hc3, spearman = table_spearman,
             influence = influence_dl),
        file.path(out_dir, "FINAL_DL_vs_ML_tables.rds"))

cat("\n==========================================\n")
cat("FINAL DL vs ML ANALYSIS COMPLETED\n")
cat("Files saved in:", out_dir, "\n")
cat("==========================================\n")
cat("\n-- Structural volatility --\n"); print(table_volatility)
cat("\n-- Liquidity --\n"); print(table_liquidity)
cat("\n-- Sector --\n"); print(table_sector)
cat("\n-- Volatility states --\n"); print(table_regimes)
cat("\n-- HC3 robust regression --\n"); print(table_hc3)
cat("\n-- Spearman --\n"); print(table_spearman)
cat("\n-- Most influential stocks --\n"); print(head(influence_dl, 5))
cat("\nNOTE: Single-seed neural networks require a separate stability check\n",
    "before treating these findings as definitive.\n")
