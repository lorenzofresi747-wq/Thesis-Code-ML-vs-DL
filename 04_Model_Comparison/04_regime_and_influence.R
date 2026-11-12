# ============================================================================
# MASTER'S THESIS -- REGIME COMPARISON + INFLUENCE ROBUSTNESS
# NO RETRAINING, NO MODEL SELECTION, AND NO CHANGES TO FROZEN ML/DL RESULTS.
# Uses stock_scores, regime_stock, regime_daily from 03_DL_vs_ML_analysis.R.
# ============================================================================
required_packages <- c("dplyr", "tidyr", "lmtest", "sandwich")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop("Install these R packages: ", paste(missing_packages, collapse = ", "))
}
suppressPackageStartupMessages({ library(dplyr); library(tidyr) })

# Restore principal stock/regime tables from the previous analysis if R restarted.
outdir_previous <- file.path(getwd(), "FINAL_DL_vs_ML")
if (!exists("stock_scores") || !exists("regime_stock")) {
  saved <- file.path(outdir_previous, "FINAL_DL_vs_ML_tables.rds")
  if (!file.exists(saved)) stop("First run 03_DL_vs_ML_analysis.R, or restore FINAL_DL_vs_ML_tables.rds")
  old <- readRDS(saved)
  stock_scores <- old$stock_scores
  regime_stock <- old$regime_stock
}
stopifnot(nrow(stock_scores) == 33L, nrow(regime_stock) == 99L)

outdir <- file.path(getwd(), "FINAL_DL_vs_ML", "robustness")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --------- 1. PAIRED HIGH-VERSUS-LOW VOLATILITY STATE COMPARISON -----------
paired_regime <- regime_stock %>%
  transmute(Ticker = as.character(Ticker),
            VolatilityState = as.character(VolatilityState),
            DLGain_RMSE = as.numeric(DLGain_RMSE)) %>%
  pivot_wider(names_from = VolatilityState, values_from = DLGain_RMSE) %>%
  mutate(High_minus_Low = High - Low,
         High_minus_Medium = High - Medium,
         High_Better_Than_Low = High_minus_Low > 0) %>%
  arrange(desc(High_minus_Low))
if (nrow(paired_regime) != 33L || anyNA(paired_regime)) {
  stop("Every stock must have valid Low, Medium and High regime DLGain.")
}

summarise_paired <- function(df, label) {
  delta <- df$High_minus_Low
  n <- length(delta)
  t_res <- t.test(delta, mu = 0, alternative = "two.sided")
  wilcox_res <- suppressWarnings(wilcox.test(delta, mu = 0,
                                             alternative = "two.sided", exact = FALSE))
  nonzero <- delta[delta != 0]
  sign_res <- if (length(nonzero)) binom.test(sum(nonzero > 0), length(nonzero), p = .5) else NULL
  tibble(Sample = label, N_Stocks = n,
         Mean_High_Low_pp = mean(delta), Median_High_Low_pp = median(delta),
         SD_High_Low_pp = sd(delta), Stocks_High_Better = sum(delta > 0),
         T_pvalue = t_res$p.value, T_CI_low = t_res$conf.int[1],
         T_CI_high = t_res$conf.int[2],
         Wilcoxon_pvalue = wilcox_res$p.value,
         SignTest_pvalue = if (is.null(sign_res)) NA_real_ else sign_res$p.value)
}
paired_tests <- bind_rows(
  summarise_paired(paired_regime, "All 33 stocks"),
  summarise_paired(filter(paired_regime, Ticker != "AMT"), "Without AMT (sensitivity)")
)

# --------- 2. HC3 REGRESSION WITH/WITHOUT AMT ------------------------------
empirical <- stock_scores %>%
  mutate(LogILLIQ = log(TrainingILLIQ),
         Volatility_z = as.numeric(scale(TrainingVolatility)),
         LogILLIQ_z = as.numeric(scale(LogILLIQ)))
stopifnot(nrow(empirical) == 33L,
          all(is.finite(empirical$DLGain_RMSE)),
          all(is.finite(empirical$Volatility_z)),
          all(is.finite(empirical$LogILLIQ_z)))

hc3_table <- function(dat, label) {
  model <- lm(DLGain_RMSE ~ Volatility_z + LogILLIQ_z, data = dat)
  co <- as.matrix(lmtest::coeftest(model, vcov. = sandwich::vcovHC(model, type = "HC3")))
  tibble(Sample = label, Term = rownames(co),
         Estimate = unname(co[,1]), Robust_SE = unname(co[,2]),
         t_value = unname(co[,3]), p_value = unname(co[,4]),
         N = nobs(model), R2 = summary(model)$r.squared,
         Adjusted_R2 = summary(model)$adj.r.squared)
}
hc3_sensitivity <- bind_rows(
  hc3_table(empirical, "All 33 stocks"),
  hc3_table(filter(empirical, Ticker != "AMT"), "Without AMT (sensitivity)")
)

# Each omitted stock, standardized covariates keep full-sample reference scale.
hc3_leave_one_out <- bind_rows(lapply(empirical$Ticker, function(ticker) {
  tab <- hc3_table(filter(empirical, Ticker != ticker), ticker)
  tab %>% filter(Term != "(Intercept)") %>%
    transmute(RemovedTicker = ticker, Term, Estimate, p_value)
}))

# --------- 3. MONTH RESAMPLING WITH STOCKS SYNCHRONIZED -------------------
# This block bootstrap is optional but valuable: resample calendar months
# *together across all 33 stocks* so contemporaneous common market shocks
# remain in the same block. It is approximate (months are treated as blocks).
if (!exists("regime_daily")) {
  saved_daily <- file.path(outdir_previous, "FINAL_ML_DL_daily_matched.rds")
  if (!file.exists(saved_daily) || !exists("model_data") || !exists("volatility_thresholds")) {
    stop("To run the month bootstrap, restore regime_daily or load model_data + volatility_thresholds and the matched daily RDS.")
  }
  daily <- readRDS(saved_daily)
  states <- model_data %>%
    filter(Sample == "Test") %>%
    transmute(Ticker = as.character(Ticker),
              ForecastOrigin = as.Date(Date), TargetDate = as.Date(TargetDate),
              Vol20 = as.numeric(Vol20)) %>%
    left_join(volatility_thresholds %>%
                transmute(Ticker = as.character(Ticker),
                          Vol20_q33 = as.numeric(Vol20_q33),
                          Vol20_q67 = as.numeric(Vol20_q67)),
              by = "Ticker") %>%
    mutate(VolatilityState = case_when(
      Vol20 < Vol20_q33 ~ "Low",
      Vol20 < Vol20_q67 ~ "Medium",
      TRUE ~ "High")) %>%
    select(Ticker, ForecastOrigin, TargetDate, VolatilityState)
  regime_daily <- daily %>%
    left_join(states, by = c("Ticker", "ForecastOrigin", "TargetDate"))
}
needed <- c("Ticker", "TargetDate", "VolatilityState", "ActualReturn", "PredML", "PredDL")
stopifnot(all(needed %in% names(regime_daily)),
          nrow(regime_daily) == 49764L,
          !anyNA(regime_daily[, needed]))

month_losses <- regime_daily %>%
  mutate(Month = format(as.Date(TargetDate), "%Y-%m"),
         VolatilityState = as.character(VolatilityState)) %>%
  group_by(Month, Ticker, VolatilityState) %>%
  summarise(SS_ML = sum((ActualReturn - PredML)^2),
            SS_DL = sum((ActualReturn - PredDL)^2),
            .groups = "drop")
months <- sort(unique(month_losses$Month))

boot_once <- function() {
  weights <- as.data.frame(table(factor(sample(months, length(months), replace = TRUE), levels = months)),
                           stringsAsFactors = FALSE)
  names(weights) <- c("Month", "Weight")
  weights <- weights[weights$Weight > 0, , drop = FALSE]
  v <- month_losses %>%
    inner_join(weights, by = "Month") %>%
    group_by(Ticker, VolatilityState) %>%
    summarise(SS_ML = sum(SS_ML * Weight), SS_DL = sum(SS_DL * Weight),
              .groups = "drop") %>%
    mutate(Gain = 100 * (1 - sqrt(SS_DL / SS_ML))) %>%
    select(Ticker, VolatilityState, Gain) %>%
    pivot_wider(names_from = VolatilityState, values_from = Gain)
  if (nrow(v) != 33L || any(!is.finite(v$High)) || any(!is.finite(v$Low))) {
    return(c(MeanDelta = NA_real_, MedianDelta = NA_real_))
  }
  c(MeanDelta = mean(v$High - v$Low),
    MedianDelta = median(v$High - v$Low))
}

set.seed(20261013)
B <- 1000L
boot_matrix <- replicate(B, boot_once())
boot_df <- as.data.frame(t(boot_matrix))
boot_df$Iteration <- seq_len(nrow(boot_df))
boot_valid <- filter(boot_df, is.finite(MeanDelta), is.finite(MedianDelta))
if (nrow(boot_valid) < .95 * B) stop("Too many invalid bootstrap resamples; inspect regime coverage.")
month_bootstrap_summary <- tibble(
  B_Valid = nrow(boot_valid),
  Observed_MeanDelta_pp = mean(paired_regime$High_minus_Low),
  Bootstrap_CI95_Low = as.numeric(quantile(boot_valid$MeanDelta, .025)),
  Bootstrap_CI95_High = as.numeric(quantile(boot_valid$MeanDelta, .975)),
  Observed_MedianDelta_pp = median(paired_regime$High_minus_Low),
  Bootstrap_Median_CI95_Low = as.numeric(quantile(boot_valid$MedianDelta, .025)),
  Bootstrap_Median_CI95_High = as.numeric(quantile(boot_valid$MedianDelta, .975))
)

# --------- SAVE WITHOUT OVERWRITING ANY ORIGINAL TABLE -------------------
write.csv(paired_regime, file.path(outdir, "12_stock_paired_high_low.csv"), row.names = FALSE)
write.csv(paired_tests, file.path(outdir, "13_paired_tests_and_AMT_sensitivity.csv"), row.names = FALSE)
write.csv(hc3_sensitivity, file.path(outdir, "14_HC3_with_without_AMT.csv"), row.names = FALSE)
write.csv(hc3_leave_one_out, file.path(outdir, "15_HC3_leave_one_out.csv"), row.names = FALSE)
write.csv(month_bootstrap_summary, file.path(outdir, "16_month_bootstrap_summary.csv"), row.names = FALSE)
saveRDS(list(paired_stocks = paired_regime, paired_tests = paired_tests,
             hc3_sensitivity = hc3_sensitivity, hc3_leave_one_out = hc3_leave_one_out,
             bootstrap_summary = month_bootstrap_summary),
        file.path(outdir, "FINAL_regime_robustness_tables.rds"))

cat("\n=== HIGH MINUS LOW VOLATILITY REGIME: PAIRED TESTS ===\n")
print(paired_tests, width = Inf)
cat("\n=== JOINT HC3 REGRESSION: WITH AND WITHOUT AMT ===\n")
print(hc3_sensitivity, width = Inf)
cat("\n=== MONTH-BLOCK BOOTSTRAP: 95% INTERVALS ===\n")
print(month_bootstrap_summary, width = Inf)
cat("\nSaved robustness files to:", outdir, "\n")
cat("Interpret paired cross-stock tests cautiously: stocks share market shocks.\n")
