# ======================================================================
# MASTER'S THESIS — VISUAL RESULTS (TRADITIONAL ML VS DEEP LEARNING)
# Run in RStudio after the DL/ML analysis. NO neural networks are retrained.
# Exports thesis-ready PDF (vector) and PNG (300 dpi) into:
# FINAL_DL_vs_ML/figures/
# ======================================================================

needed <- c("dplyr", "ggplot2")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install missing R packages first: install.packages(c(",
       paste(shQuote(missing), collapse = ", "), "))")
}
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

# ---- Load prior results if they are not already in R's Environment ----
root <- file.path(getwd(), "FINAL_DL_vs_ML")
saved_tables <- file.path(root, "FINAL_DL_vs_ML_tables.rds")
if (!exists("stock_scores") || !exists("regime_stock")) {
  if (!file.exists(saved_tables)) {
    message("Select FINAL_DL_vs_ML_tables.rds from your previous analysis.")
    saved_tables <- file.choose()
    root <- dirname(saved_tables)
  }
  results <- readRDS(saved_tables)
  stock_scores <- results$stock_scores
  regime_stock <- results$regime_stock
}

stopifnot(
  is.data.frame(stock_scores), nrow(stock_scores) == 33,
  all(c("Ticker", "sector", "VolatilityGroup", "LiquidityGroup",
        "RMSEGain_ML", "RMSEGain_DL", "DLGain_RMSE") %in% names(stock_scores)),
  is.data.frame(regime_stock), nrow(regime_stock) == 99,
  all(c("Ticker", "VolatilityState", "DLGain_RMSE") %in% names(regime_stock))
)

out_dir <- file.path(root, "figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Consistent formatting, optimized for inclusion in a thesis.
ML_COLOR <- "#275D80"
DL_COLOR <- "#C9773B"
ZERO_COLOR <- "#777777"
base_theme <- theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "#EAEAEA", linewidth = 0.3),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, color = "#4A4A4A"),
    plot.caption = element_text(size = 8.5, color = "#555555", hjust = 0),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

save_fig <- function(plot, filename, width, height) {
  ggsave(file.path(out_dir, paste0(filename, ".pdf")), plot = plot,
         width = width, height = height, units = "in", bg = "white")
  ggsave(file.path(out_dir, paste0(filename, ".png")), plot = plot,
         width = width, height = height, units = "in", dpi = 300, bg = "white")
  message("Saved: ", filename, " (.pdf and .png)")
}

# ======================================================================
# FIGURE 0: TRADITIONAL ML BASELINE — WHERE DOES ML BEAT ZERO?
# Suggested placement: end of the ML results section (Section 4.2.1).
# ======================================================================
ml_order <- stock_scores$Ticker[order(stock_scores$RMSEGain_ML)]
ml_data <- stock_scores %>%
  mutate(Ticker_display = factor(Ticker, levels = ml_order),
         Better_zero = if_else(RMSEGain_ML > 0, "ML beats zero", "Zero wins"))
fig0 <- ggplot(ml_data, aes(x = Ticker_display, y = RMSEGain_ML,
                            fill = Better_zero)) +
  geom_hline(yintercept = 0, color = ZERO_COLOR, linewidth = 0.4) +
  geom_col(width = 0.78) +
  coord_flip() +
  scale_fill_manual(values = c("ML beats zero" = ML_COLOR,
                               "Zero wins" = "#B7C1C9"), name = NULL) +
  labs(
    title = "Traditional ML compared with the zero-return benchmark",
    subtitle = "One validation-selected ML model for each of the 33 stocks",
    x = NULL, y = "RMSEGain relative to zero (%)",
    caption = "Positive values indicate lower forecasting error than the zero-return benchmark."
  ) + base_theme +
  theme(axis.text.y = element_text(size = 8.5))
save_fig(fig0, "Figure_00_ML_vs_zero_by_stock", 8.5, 9)

# ======================================================================
# FIGURE 1: MODEL PERFORMANCE VS ZERO-RETURN BENCHMARK FOR ALL 33 STOCKS
# Connected points show if DL actually improves ML and whether either beats 0.
# ======================================================================
ordered_tickers <- stock_scores$Ticker[order(stock_scores$DLGain_RMSE)]
fig1_data <- stock_scores %>%
  mutate(Ticker_display = factor(Ticker, levels = ordered_tickers))

fig1 <- ggplot(fig1_data, aes(x = Ticker_display)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.55,
             color = ZERO_COLOR) +
  geom_segment(aes(xend = Ticker_display, y = RMSEGain_ML,
                   yend = RMSEGain_DL),
               linewidth = 0.65, color = "#A9B1B7") +
  geom_point(aes(y = RMSEGain_ML, color = "Traditional ML"), size = 2.2) +
  geom_point(aes(y = RMSEGain_DL, color = "Deep Learning"), size = 2.2) +
  coord_flip() +
  scale_color_manual(values = c("Traditional ML" = ML_COLOR,
                                "Deep Learning" = DL_COLOR), name = NULL) +
  labs(
    title = "Traditional ML and DL relative to the zero-return benchmark",
    subtitle = "One validation-selected model per family and stock | Test: 2020–2025",
    x = NULL, y = "RMSEGain relative to zero (%)",
    caption = "Positive values beat zero; negative values lose to zero. Each line joins ML and DL for the same stock."
  ) + base_theme +
  theme(axis.text.y = element_text(size = 8))
save_fig(fig1, "Figure_01_ML_DL_vs_zero_by_stock", 9, 9.4)

# ======================================================================
# FIGURE 2: RELATIVE DL ADVANTAGE ACROSS ECONOMIC SECTORS
# Individual-stock dots + sector medians (only 3 stocks per sector).
# ======================================================================
sector_order <- stock_scores %>%
  group_by(sector) %>%
  summarise(Median = median(DLGain_RMSE), .groups = "drop") %>%
  arrange(Median) %>% pull(sector)
fig2_data <- stock_scores %>%
  mutate(Sector_display = factor(sector, levels = sector_order))

fig2 <- ggplot(fig2_data, aes(x = Sector_display, y = DLGain_RMSE)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = ZERO_COLOR) +
  geom_jitter(width = 0.12, height = 0, size = 2.2,
              alpha = 0.7, color = ML_COLOR) +
  stat_summary(fun = median, geom = "point", shape = 18,
               size = 3.9, color = DL_COLOR) +
  coord_flip() +
  labs(
    title = "Relative DL advantage by sector",
    subtitle = "Each dot represents one stock; diamonds indicate sector medians",
    x = NULL, y = "DLGain in RMSE (%)",
    caption = "DLGain > 0 favors DL. Each sector includes only three stocks; sector differences are descriptive."
  ) + base_theme
save_fig(fig2, "Figure_02_DLGain_by_sector", 9, 6.4)

# ======================================================================
# FIGURE 3: VOLATILITY AND LIQUIDITY GROUPS
# Individual observations are shown to reveal skewness and AMT/SPG outliers.
# ======================================================================
vol_df <- stock_scores %>%
  transmute(Ticker, Dimension = "Structural volatility",
            Group = paste(VolatilityGroup, "volatility"),
            DLGain = DLGain_RMSE)
liq_df <- stock_scores %>%
  transmute(Ticker, Dimension = "Liquidity",
            Group = paste(LiquidityGroup, "liquidity"),
            DLGain = DLGain_RMSE)
fig3_data <- bind_rows(vol_df, liq_df) %>%
  mutate(
    Dimension = factor(Dimension, levels = c("Structural volatility", "Liquidity")),
    Group = factor(Group, levels = c(
      "Low volatility", "Medium volatility", "High volatility",
      "High liquidity", "Medium liquidity", "Low liquidity"
    ))
  )
fig3 <- ggplot(fig3_data, aes(x = Group, y = DLGain)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = ZERO_COLOR) +
  geom_jitter(width = 0.11, height = 0, size = 2.0,
              alpha = 0.75, color = ML_COLOR) +
  stat_summary(fun = median, geom = "point", shape = 18,
               size = 4, color = DL_COLOR) +
  facet_wrap(~ Dimension, scales = "free_x", nrow = 1) +
  labs(
    title = "Does market structure explain the advantage of DL?",
    subtitle = "Each point is a stock (11 per group); diamonds indicate group medians",
    x = NULL, y = "DLGain in RMSE (%)",
    caption = "Stocks are grouped using training-period characteristics. Large positive values are not representative of every stock."
  ) + base_theme +
  theme(axis.text.x = element_text(angle = 22, hjust = 1, size = 8))
save_fig(fig3, "Figure_03_DLGain_by_volatility_and_liquidity", 10.8, 5.3)

# ======================================================================
# FIGURE 4: DL ADVANTAGE ACROSS TIME-VARYING VOLATILITY STATES
# Shows all 33 stocks in each state; distributions matter more than means.
# ======================================================================
fig4_data <- regime_stock %>%
  mutate(VolatilityState = factor(VolatilityState,
                                  levels = c("Low", "Medium", "High")))
fig4 <- ggplot(fig4_data, aes(x = VolatilityState, y = DLGain_RMSE)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = ZERO_COLOR) +
  geom_boxplot(width = 0.50, outlier.shape = NA,
               fill = "#DEEAF0", color = ML_COLOR, alpha = 0.85) +
  geom_jitter(width = 0.12, height = 0, size = 1.8,
              alpha = 0.65, color = ML_COLOR) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               fill = DL_COLOR, color = "white", stroke = 0.7, size = 3.7) +
  labs(
    title = "Relative DL performance across changing volatility states",
    subtitle = "33 stocks per state | Orange diamonds indicate means",
    x = "Volatility state (training thresholds)", y = "DLGain in RMSE (%)",
    caption = "Descriptive pattern only. Paired high-versus-low t-test: p = 0.356; month bootstrap CI includes zero."
  ) + base_theme
save_fig(fig4, "Figure_04_DLGain_by_time_varying_volatility", 8.4, 5.8)

# ======================================================================
# FIGURE 5: SEED STABILITY — NUMBER OF STOCKS WHERE DL BEATS ML
# Plot if the multi-seed robustness run was completed.
# ======================================================================
if (!exists("seed_overall_summary")) {
  seed_csv <- file.path(root, "seed_stability", "seed_02_overall_stability.csv")
  if (file.exists(seed_csv)) {
    seed_overall_summary <- read.csv(seed_csv, stringsAsFactors = FALSE)
  }
}
if (exists("seed_overall_summary") &&
    all(c("SeedLabel", "Stocks_DL_Better", "N_Stocks") %in%
        names(seed_overall_summary))) {
  fig5_data <- seed_overall_summary %>%
    mutate(SeedLabel = factor(as.character(SeedLabel),
                              levels = c("Original", "Seed_101", "Seed_202", "Seed_303")))
  fig5 <- ggplot(fig5_data, aes(x = SeedLabel, y = Stocks_DL_Better)) +
    geom_hline(yintercept = 16.5, linetype = "dashed", color = ZERO_COLOR) +
    geom_col(fill = ML_COLOR, width = 0.58) +
    geom_text(aes(label = paste0(Stocks_DL_Better, "/", N_Stocks)),
              vjust = -0.7, size = 4) +
    scale_y_continuous(limits = c(0, 23), breaks = seq(0, 20, 5)) +
    labs(
      title = "How sensitive is the ML–DL comparison to random initialization?",
      subtitle = "Same selected architecture and hyperparameters, four neural-network initializations",
      x = "DL training run", y = "Number of stocks where DL has lower RMSE",
      caption = "Dashed line represents half the stock universe (16.5 out of 33). The four runs are not independent samples."
    ) + base_theme
  save_fig(fig5, "Figure_05_Seed_stability_DL_wins", 8, 5)
} else {
  message("Seed summary unavailable: skipping Figure 5 (Figures 0–4 were saved).")
}
