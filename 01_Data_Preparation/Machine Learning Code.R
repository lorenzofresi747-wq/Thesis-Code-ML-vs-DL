# ============================================================
# MASTER'S THESIS
# ML vs DL Stock Forecasting
#
# STEP 1: DOWNLOAD RAW STOCK DATA FROM YAHOO FINANCE
# Period: 2010-01-01 to 2025-12-31
# Universe: 33 stocks, 11 sectors
# ============================================================


# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------

packages <- c(
  "quantmod",
  "xts",
  "zoo",
  "dplyr",
  "purrr",
  "tibble"
)

# Install missing packages
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]

if(length(new_packages) > 0){
  install.packages(new_packages)
}

# Load packages
invisible(
  lapply(packages, library, character.only = TRUE)
)


# ------------------------------------------------------------
# 1. SAMPLE PERIOD
# ------------------------------------------------------------

start_date <- as.Date("2010-01-01")

# Using 2026-01-01 as the download boundary ensures that
# observations through 2025-12-31 are available.
# We will explicitly restrict the data afterward.
download_end_date <- as.Date("2026-01-01")

final_date <- as.Date("2025-12-31")


# ------------------------------------------------------------
# 2. STOCK UNIVERSE
# ------------------------------------------------------------

stock_info <- tibble(
  
  ticker = c(
    
    # Information Technology
    "MSFT", "NVDA", "IBM",
    
    # Communication Services
    "VZ", "DIS", "NFLX",
    
    # Consumer Discretionary
    "AMZN", "MCD", "F",
    
    # Consumer Staples
    "KO", "PG", "WMT",
    
    # Health Care
    "JNJ", "UNH", "GILD",
    
    # Financials
    "JPM", "BAC", "GS",
    
    # Industrials
    "CAT", "BA", "UNP",
    
    # Energy
    "XOM", "OXY", "SLB",
    
    # Utilities
    "NEE", "DUK", "ED",
    
    # Materials
    "APD", "FCX", "NEM",
    
    # Real Estate
    "AMT", "O", "SPG"
  ),
  
  sector = c(
    
    rep("Information Technology", 3),
    rep("Communication Services", 3),
    rep("Consumer Discretionary", 3),
    rep("Consumer Staples", 3),
    rep("Health Care", 3),
    rep("Financials", 3),
    rep("Industrials", 3),
    rep("Energy", 3),
    rep("Utilities", 3),
    rep("Materials", 3),
    rep("Real Estate", 3)
    
  )
)


# Check sample size
nrow(stock_info)

# Should return:
# 33


# Look at selected stocks
print(stock_info)


# ------------------------------------------------------------
# 3. VECTOR OF YAHOO FINANCE TICKERS
# ------------------------------------------------------------

tickers <- stock_info$ticker

print(tickers)


# ------------------------------------------------------------
# 4. FUNCTION TO DOWNLOAD ONE STOCK
# ------------------------------------------------------------

download_stock <- function(symbol,
                           from = start_date,
                           to = download_end_date) {
  
  cat("Downloading:", symbol, "...\n")
  
  tryCatch(
    
    {
      
      data <- getSymbols(
        Symbols = symbol,
        src = "yahoo",
        from = from,
        to = to,
        auto.assign = FALSE,
        warnings = FALSE
      )
      
      
      # Keep only observations through the thesis end date
      data <- data[paste0("/", final_date)]
      
      
      # Standardize column names
      colnames(data) <- c(
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Adjusted"
      )
      
      
      cat(
        "   Success:",
        NROW(data),
        "observations from",
        as.character(first(index(data))),
        "to",
        as.character(last(index(data))),
        "\n"
      )
      
      
      return(data)
      
    },
    
    
    error = function(e) {
      
      warning(
        paste(
          "Download failed for",
          symbol,
          "-",
          e$message
        )
      )
      
      return(NULL)
    }
    
  )
}


# ------------------------------------------------------------
# 5. DOWNLOAD ALL 33 STOCKS
# ------------------------------------------------------------

raw_stocks <- setNames(
  
  lapply(
    tickers,
    download_stock
  ),
  
  tickers
  
)


# ------------------------------------------------------------
# 6. CHECK DOWNLOAD RESULTS
# ------------------------------------------------------------

download_summary <- tibble(
  
  ticker = tickers,
  
  downloaded = sapply(
    raw_stocks,
    function(x) !is.null(x)
  ),
  
  observations = sapply(
    raw_stocks,
    function(x){
      if(is.null(x)) NA_integer_ else NROW(x)
    }
  ),
  
  first_date = as.Date(
    sapply(
      raw_stocks,
      function(x){
        if(is.null(x)){
          NA_character_
        } else {
          as.character(first(index(x)))
        }
      }
    )
  ),
  
  last_date = as.Date(
    sapply(
      raw_stocks,
      function(x){
        if(is.null(x)){
          NA_character_
        } else {
          as.character(last(index(x)))
        }
      }
    )
  )
  
) %>%
  
  left_join(
    stock_info,
    by = "ticker"
  )


print(download_summary)


# ------------------------------------------------------------
# 7. IDENTIFY FAILED DOWNLOADS
# ------------------------------------------------------------

failed_tickers <- download_summary %>%
  filter(!downloaded) %>%
  pull(ticker)


if(length(failed_tickers) == 0){
  
  cat("\nAll 33 stocks downloaded successfully.\n")
  
} else {
  
  cat(
    "\nFailed downloads:",
    paste(failed_tickers, collapse = ", "),
    "\n"
  )
  
}


# ------------------------------------------------------------
# 8. CHECK NUMBER OF OBSERVATIONS
# ------------------------------------------------------------

download_summary %>%
  
  arrange(observations) %>%
  
  select(
    ticker,
    sector,
    observations,
    first_date,
    last_date
  ) %>%
  
  print(n = 33)


# ------------------------------------------------------------
# 9. EXAMPLE: INSPECT ONE STOCK
# ------------------------------------------------------------

head(raw_stocks$KO)

tail(raw_stocks$KO)

head(raw_stocks$NVDA)

tail(raw_stocks$NVDA)


saveRDS(
  raw_stocks,
  file = "raw_stocks_2010_2025.rds"
)

saveRDS(
  stock_info,
  file = "stock_information.rds"
)

write.csv(
  download_summary,
  file = "download_summary.csv",
  row.names = FALSE
)


cat("\nRaw data successfully saved.\n")

# ============================================================
# MASTER'S THESIS
# ML vs DL Stock Forecasting
#
# STEP 2: CLEAN RAW YAHOO FINANCE DATA
#
# IMPORTANT:
# - No interpolation
# - No winsorization
# - No standardization
# - No outlier removal
# - No use of future information
# ============================================================


# ------------------------------------------------------------
# 1. CHECK THAT RAW DATA EXIST
# ------------------------------------------------------------

if(!exists("raw_stocks")){
  stop(
    "Object 'raw_stocks' not found. Run the download script first."
  )
}

if(!exists("stock_info")){
  stop(
    "Object 'stock_info' not found. Run the download script first."
  )
}


# ------------------------------------------------------------
# 2. DEFINE REQUIRED VARIABLES
# ------------------------------------------------------------

required_columns <- c(
  "Open",
  "High",
  "Low",
  "Close",
  "Volume",
  "Adjusted"
)


# ------------------------------------------------------------
# 3. FUNCTION TO AUDIT + CLEAN ONE STOCK
# ------------------------------------------------------------

clean_stock <- function(data, ticker){
  
  cat("\nCleaning:", ticker, "...\n")
  
  
  # ----------------------------------------------------------
  # Safety check
  # ----------------------------------------------------------
  
  if(is.null(data)){
    
    warning(
      paste("No raw data available for", ticker)
    )
    
    return(NULL)
  }
  
  
  # ----------------------------------------------------------
  # Convert xts object to data.frame
  # ----------------------------------------------------------
  
  df <- data.frame(
    Date = as.Date(index(data)),
    coredata(data)
  )
  
  
  # Standardize column names
  colnames(df) <- c(
    "Date",
    required_columns
  )
  
  
  # Store initial number of observations
  n_initial <- nrow(df)
  
  
  # ----------------------------------------------------------
  # 4. SORT CHRONOLOGICALLY
  # ----------------------------------------------------------
  
  df <- df %>%
    arrange(Date)
  
  
  # ----------------------------------------------------------
  # 5. REMOVE DUPLICATED DATES
  # ----------------------------------------------------------
  
  duplicated_dates <- sum(
    duplicated(df$Date)
  )
  
  df <- df %>%
    distinct(
      Date,
      .keep_all = TRUE
    )
  
  
  # ----------------------------------------------------------
  # 6. KEEP ONLY THESIS SAMPLE
  # ----------------------------------------------------------
  
  df <- df %>%
    filter(
      Date >= start_date,
      Date <= final_date
    )
  
  
  # ----------------------------------------------------------
  # 7. ENSURE NUMERIC VARIABLES
  # ----------------------------------------------------------
  
  df <- df %>%
    
    mutate(
      
      across(
        all_of(required_columns),
        as.numeric
      )
      
    )
  
  
  # ----------------------------------------------------------
  # 8. COUNT MISSING VALUES BEFORE REMOVAL
  # ----------------------------------------------------------
  
  missing_values <- colSums(
    is.na(df[, required_columns])
  )
  
  
  rows_with_missing <- sum(
    !complete.cases(
      df[, required_columns]
    )
  )
  
  
  # ----------------------------------------------------------
  # 9. IDENTIFY INVALID PRICE OBSERVATIONS
  #
  # Prices cannot be <= 0.
  # ----------------------------------------------------------
  
  invalid_price <- (
    
    df$Open <= 0 |
      df$High <= 0 |
      df$Low <= 0 |
      df$Close <= 0 |
      df$Adjusted <= 0
    
  )
  
  
  invalid_price[is.na(invalid_price)] <- TRUE
  
  
  n_invalid_price <- sum(
    invalid_price
  )
  
  
  # ----------------------------------------------------------
  # 10. IDENTIFY INVALID VOLUME
  #
  # Negative trading volume is impossible.
  #
  # Zero volume is NOT automatically removed:
  # we simply record it.
  # ----------------------------------------------------------
  
  invalid_volume <- df$Volume < 0
  
  invalid_volume[
    is.na(invalid_volume)
  ] <- TRUE
  
  
  n_invalid_volume <- sum(
    invalid_volume
  )
  
  
  zero_volume <- sum(
    df$Volume == 0,
    na.rm = TRUE
  )
  
  
  # ----------------------------------------------------------
  # 11. CHECK OHLC CONSISTENCY
  #
  # High should normally be >= Open, Low and Close.
  # Low should normally be <= Open, High and Close.
  # ----------------------------------------------------------
  
  inconsistent_ohlc <- (
    
    df$High < df$Open |
      df$High < df$Close |
      df$High < df$Low |
      
      df$Low > df$Open |
      df$Low > df$Close |
      df$Low > df$High
    
  )
  
  
  inconsistent_ohlc[
    is.na(inconsistent_ohlc)
  ] <- TRUE
  
  
  n_inconsistent_ohlc <- sum(
    inconsistent_ohlc
  )
  
  
  # ----------------------------------------------------------
  # 12. REMOVE UNUSABLE ROWS
  #
  # We remove only:
  #
  # 1. Missing essential data
  # 2. Non-positive prices
  # 3. Negative volume
  # 4. Internally inconsistent OHLC observations
  #
  # We DO NOT remove:
  #
  # - Large price movements
  # - Crashes
  # - Volatility spikes
  # - Large volume spikes
  # - Zero-volume observations automatically
  # ----------------------------------------------------------
  
  df_clean <- df %>%
    
    filter(
      
      complete.cases(
        across(
          all_of(required_columns)
        )
      ),
      
      Open > 0,
      High > 0,
      Low > 0,
      Close > 0,
      Adjusted > 0,
      
      Volume >= 0,
      
      High >= Open,
      High >= Close,
      High >= Low,
      
      Low <= Open,
      Low <= Close,
      Low <= High
      
    )
  
  
  # ----------------------------------------------------------
  # 13. ADD TICKER
  # ----------------------------------------------------------
  
  df_clean <- df_clean %>%
    
    mutate(
      
      Ticker = ticker,
      
      .before = Date
      
    )
  
  
  # ----------------------------------------------------------
  # 14. CREATE ADJUSTMENT FACTOR
  #
  # Yahoo provides:
  #
  # Adjusted / Close
  #
  # This captures the cumulative effect of corporate actions
  # such as stock splits and dividends.
  #
  # We preserve BOTH raw and adjusted information.
  # ----------------------------------------------------------
  
  df_clean <- df_clean %>%
    
    mutate(
      
      AdjustmentFactor =
        Adjusted / Close
      
    )
  
  
  # ----------------------------------------------------------
  # 15. CREATE ADJUSTED OHLC SERIES
  #
  # Useful later for technical indicators and price-derived
  # predictors while avoiding artificial split jumps.
  # ----------------------------------------------------------
  
  df_clean <- df_clean %>%
    
    mutate(
      
      AdjOpen =
        Open * AdjustmentFactor,
      
      AdjHigh =
        High * AdjustmentFactor,
      
      AdjLow =
        Low * AdjustmentFactor,
      
      AdjClose =
        Adjusted
      
    )
  
  
  # ----------------------------------------------------------
  # 16. FINAL SANITY CHECK
  # ----------------------------------------------------------
  
  n_final <- nrow(df_clean)
  
  rows_removed <- n_initial - n_final
  
  
  cat(
    "   Initial observations:",
    n_initial,
    "\n"
  )
  
  cat(
    "   Final observations:",
    n_final,
    "\n"
  )
  
  cat(
    "   Rows removed:",
    rows_removed,
    "\n"
  )
  
  
  # ----------------------------------------------------------
  # Return cleaned data + audit information
  # ----------------------------------------------------------
  
  list(
    
    data = df_clean,
    
    audit = tibble(
      
      ticker = ticker,
      
      initial_obs =
        n_initial,
      
      final_obs =
        n_final,
      
      removed_obs =
        rows_removed,
      
      duplicated_dates =
        duplicated_dates,
      
      rows_missing =
        rows_with_missing,
      
      invalid_prices =
        n_invalid_price,
      
      invalid_volume =
        n_invalid_volume,
      
      zero_volume =
        zero_volume,
      
      inconsistent_ohlc =
        n_inconsistent_ohlc,
      
      first_date =
        if(n_final > 0)
          min(df_clean$Date)
      else
        as.Date(NA),
      
      last_date =
        if(n_final > 0)
          max(df_clean$Date)
      else
        as.Date(NA)
      
    )
    
  )
  
}

cleaning_results <- mapply(
  
  FUN = clean_stock,
  
  data = raw_stocks,
  
  ticker = names(raw_stocks),
  
  SIMPLIFY = FALSE
  
)

clean_stocks <- lapply(
  
  cleaning_results,
  
  function(x){
    
    if(is.null(x)){
      return(NULL)
    }
    
    x$data
    
  }
  
)


# Preserve stock names

names(clean_stocks) <-
  names(raw_stocks)

cleaning_audit <- bind_rows(
  
  lapply(
    
    cleaning_results,
    
    function(x){
      
      if(is.null(x)){
        return(NULL)
      }
      
      x$audit
      
    }
    
  )
  
)


cleaning_audit <- cleaning_audit %>%
  
  left_join(
    stock_info,
    by = c("ticker" = "ticker")
  ) %>%
  
  select(
    ticker,
    sector,
    everything()
  )


print(
  cleaning_audit,
  n = 33
)


# ============================================================
# MASTER'S THESIS
# ML vs DL Stock Forecasting
#
# STEP 3: FEATURE ENGINEERING + TRADITIONAL MACHINE LEARNING
#
# Models:
#   1. OLS
#   2. Ridge
#   3. LASSO
#   4. Bagging
#   5. Random Forest
#   6. Gradient Boosting
#
# Output:
#   ml_results_table
#   ml_predictions
# ============================================================


# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------

packages_ml <- c(
  "dplyr",
  "tidyr",
  "purrr",
  "tibble",
  "zoo",
  "TTR",
  "glmnet",
  "ranger",
  "gbm"
)


new_packages <- packages_ml[
  !(packages_ml %in% installed.packages()[, "Package"])
]


if(length(new_packages) > 0){
  install.packages(new_packages)
}


invisible(
  lapply(
    packages_ml,
    library,
    character.only = TRUE
  )
)


# ------------------------------------------------------------
# 1. CHECK INPUT DATA
# ------------------------------------------------------------

if(!exists("stock_panel")){
  
  if(file.exists("stock_panel_clean_2010_2025.rds")){
    
    stock_panel <- readRDS(
      "stock_panel_clean_2010_2025.rds"
    )
    
  } else {
    
    stop(
      paste(
        "stock_panel was not found.",
        "Run the cleaning stage first."
      )
    )
    
  }
}


# ------------------------------------------------------------
# 2. FEATURE-ENGINEERING PARAMETERS
# ------------------------------------------------------------

RETURN_LAGS <- 1:5

TREND_WINDOWS <- c(
  5,
  10,
  20
)

MOMENTUM_WINDOWS <- c(
  5,
  10,
  20
)

VOL_WINDOWS <- c(
  5,
  20
)

VOLUME_WINDOW <- 20

ILLIQ_WINDOW <- 20


# ------------------------------------------------------------
# 3. FUNCTION TO CREATE FEATURES FOR ONE STOCK
# ------------------------------------------------------------

create_stock_features <- function(df){
  
  df <- df %>%
    arrange(Date)
  
  
  # ----------------------------------------------------------
  # DAILY LOG RETURN
  # ----------------------------------------------------------
  
  Return <- c(
    NA,
    diff(log(df$AdjClose))
  )
  
  
  # ----------------------------------------------------------
  # TARGET:
  # one-day-ahead log return
  # ----------------------------------------------------------
  
  Target <- dplyr::lead(
    Return,
    1
  )
  
  
  TargetDate <- dplyr::lead(
    df$Date,
    1
  )
  
  
  # ----------------------------------------------------------
  # LAGGED RETURNS
  # ----------------------------------------------------------
  
  lagged_returns <- lapply(
    
    RETURN_LAGS,
    
    function(k){
      
      dplyr::lag(
        Return,
        k
      )
      
    }
    
  )
  
  
  names(lagged_returns) <-
    paste0(
      "ReturnLag",
      RETURN_LAGS
    )
  
  
  # ----------------------------------------------------------
  # SIMPLE MOVING AVERAGES
  # ----------------------------------------------------------
  
  sma_list <- lapply(
    
    TREND_WINDOWS,
    
    function(h){
      
      TTR::SMA(
        df$AdjClose,
        n = h
      )
      
    }
    
  )
  
  
  sma_distance <- Map(
    
    function(sma, h){
      
      (
        df$AdjClose - sma
      ) / sma
      
    },
    
    sma_list,
    TREND_WINDOWS
    
  )
  
  
  names(sma_distance) <-
    paste0(
      "SMADist",
      TREND_WINDOWS
    )
  
  
  # ----------------------------------------------------------
  # EXPONENTIAL MOVING AVERAGES
  # ----------------------------------------------------------
  
  ema_list <- lapply(
    
    TREND_WINDOWS,
    
    function(h){
      
      TTR::EMA(
        df$AdjClose,
        n = h
      )
      
    }
    
  )
  
  
  ema_distance <- Map(
    
    function(ema, h){
      
      (
        df$AdjClose - ema
      ) / ema
      
    },
    
    ema_list,
    TREND_WINDOWS
    
  )
  
  
  names(ema_distance) <-
    paste0(
      "EMADist",
      TREND_WINDOWS
    )
  
  
  # ----------------------------------------------------------
  # MOMENTUM
  # ----------------------------------------------------------
  
  momentum_list <- lapply(
    
    MOMENTUM_WINDOWS,
    
    function(h){
      
      log(
        df$AdjClose /
          dplyr::lag(
            df$AdjClose,
            h
          )
      )
      
    }
    
  )
  
  
  names(momentum_list) <-
    paste0(
      "Momentum",
      MOMENTUM_WINDOWS
    )
  
  
  # ----------------------------------------------------------
  # RSI
  # ----------------------------------------------------------
  
  RSI14 <- TTR::RSI(
    df$AdjClose,
    n = 14
  )
  
  
  # ----------------------------------------------------------
  # MACD
  #
  # Thesis specification:
  # EMA(12) - EMA(26)
  # signal line = EMA(9)
  # ----------------------------------------------------------
  
  macd_result <- TTR::MACD(
    
    df$AdjClose,
    
    nFast = 12,
    nSlow = 26,
    nSig = 9,
    
    maType = "EMA",
    
    percent = FALSE
    
  )
  
  
  MACD <- as.numeric(
    macd_result[, 1]
  )
  
  
  MACDSignal <- as.numeric(
    macd_result[, 2]
  )
  
  
  # Normalize MACD by price.
  #
  # The raw MACD difference is still available conceptually,
  # but expressing it relative to price improves comparability.
  
  MACDRel <-
    MACD / df$AdjClose
  
  
  MACDSignalRel <-
    MACDSignal / df$AdjClose
  
  
  # ----------------------------------------------------------
  # BOLLINGER BAND POSITION
  # ----------------------------------------------------------
  
  bb_middle <- zoo::rollapply(
    
    df$AdjClose,
    
    width = 20,
    
    FUN = mean,
    
    align = "right",
    
    fill = NA
    
  )
  
  
  bb_sd <- zoo::rollapply(
    
    df$AdjClose,
    
    width = 20,
    
    FUN = sd,
    
    align = "right",
    
    fill = NA
    
  )
  
  
  bb_upper <-
    bb_middle + 2 * bb_sd
  
  
  bb_lower <-
    bb_middle - 2 * bb_sd
  
  
  BBPosition <- (
    df$AdjClose - bb_lower
  ) / (
    bb_upper - bb_lower
  )
  
  
  # ----------------------------------------------------------
  # STOCHASTIC OSCILLATOR
  # ----------------------------------------------------------
  
  HLC <- cbind(
    df$AdjHigh,
    df$AdjLow,
    df$AdjClose
  )
  
  
  stochastic_result <- TTR::stoch(
    
    HLC,
    
    nFastK = 14,
    nFastD = 3,
    nSlowD = 3
    
  )
  
  
  StochK <- as.numeric(
    stochastic_result[, 1]
  )
  
  
  StochD <- as.numeric(
    stochastic_result[, 2]
  )
  
  
  # ----------------------------------------------------------
  # ROLLING VOLATILITY
  # ----------------------------------------------------------
  
  volatility_list <- lapply(
    
    VOL_WINDOWS,
    
    function(h){
      
      zoo::rollapply(
        
        Return,
        
        width = h,
        
        FUN = sd,
        
        align = "right",
        
        fill = NA
        
      )
      
    }
    
  )
  
  
  names(volatility_list) <-
    paste0(
      "Vol",
      VOL_WINDOWS
    )
  
  
  # ----------------------------------------------------------
  # VOLUME VARIABLES
  # ----------------------------------------------------------
  
  LogVolume <- ifelse(
    
    df$Volume > 0,
    
    log(df$Volume),
    
    NA_real_
    
  )
  
  
  rolling_volume <- zoo::rollapply(
    
    df$Volume,
    
    width = VOLUME_WINDOW,
    
    FUN = mean,
    
    align = "right",
    
    fill = NA
    
  )
  
  
  RelVolume20 <-
    df$Volume /
    rolling_volume
  
  
  # ----------------------------------------------------------
  # ON-BALANCE VOLUME
  # ----------------------------------------------------------
  
  OBV <- TTR::OBV(
    df$AdjClose,
    df$Volume
  )
  
  
  DeltaOBV <-
    OBV -
    dplyr::lag(
      OBV,
      1
    )
  
  
  # ----------------------------------------------------------
  # OVERNIGHT RETURN
  #
  # Adjusted open/close are used here to avoid artificial
  # movements induced by corporate actions.
  # ----------------------------------------------------------
  
  OvernightReturn <- log(
    
    df$AdjOpen /
      dplyr::lag(
        df$AdjClose,
        1
      )
    
  )
  
  
  # ----------------------------------------------------------
  # DOLLAR TRADING VOLUME
  # ----------------------------------------------------------
  
  DollarVolume <-
    df$AdjClose *
    df$Volume
  
  
  # ----------------------------------------------------------
  # AMIHUD DAILY ILLIQUIDITY
  # ----------------------------------------------------------
  
  DailyILLIQ <- ifelse(
    
    DollarVolume > 0,
    
    abs(Return) /
      DollarVolume,
    
    NA_real_
    
  )
  
  
  ILLIQ20 <- zoo::rollapply(
    
    DailyILLIQ,
    
    width = ILLIQ_WINDOW,
    
    FUN = mean,
    
    align = "right",
    
    fill = NA
    
  )
  
  
  # ----------------------------------------------------------
  # BUILD FINAL FEATURE DATASET
  # ----------------------------------------------------------
  
  feature_df <- df %>%
    
    mutate(
      
      Return = Return,
      
      Target = Target,
      
      TargetDate = TargetDate,
      
      RSI14 = RSI14,
      
      MACDRel = MACDRel,
      
      MACDSignalRel =
        MACDSignalRel,
      
      BBPosition =
        BBPosition,
      
      StochK =
        StochK,
      
      StochD =
        StochD,
      
      LogVolume =
        LogVolume,
      
      RelVolume20 =
        RelVolume20,
      
      DeltaOBV =
        DeltaOBV,
      
      OvernightReturn =
        OvernightReturn,
      
      DollarVolume =
        DollarVolume,
      
      DailyILLIQ =
        DailyILLIQ,
      
      ILLIQ20 =
        ILLIQ20
      
    )
  
  
  # Add lagged returns
  
  for(name in names(lagged_returns)){
    
    feature_df[[name]] <-
      lagged_returns[[name]]
    
  }
  
  
  # Add SMA distances
  
  for(name in names(sma_distance)){
    
    feature_df[[name]] <-
      sma_distance[[name]]
    
  }
  
  
  # Add EMA distances
  
  for(name in names(ema_distance)){
    
    feature_df[[name]] <-
      ema_distance[[name]]
    
  }
  
  
  # Add momentum
  
  for(name in names(momentum_list)){
    
    feature_df[[name]] <-
      momentum_list[[name]]
    
  }
  
  
  # Add volatility
  
  for(name in names(volatility_list)){
    
    feature_df[[name]] <-
      volatility_list[[name]]
    
  }
  
  
  # ----------------------------------------------------------
  # SAMPLE ASSIGNMENT
  #
  # IMPORTANT:
  # use TARGET DATE rather than forecasting-origin date.
  # ----------------------------------------------------------
  
  feature_df <- feature_df %>%
    
    mutate(
      
      Sample = case_when(
        
        TargetDate <=
          as.Date("2017-12-31") ~
          "Training",
        
        TargetDate <=
          as.Date("2019-12-31") ~
          "Validation",
        
        TargetDate <=
          as.Date("2025-12-31") ~
          "Test",
        
        TRUE ~
          NA_character_
        
      )
      
    )
  
  
  return(
    feature_df
  )
  
}

# ============================================================
# 4. CREATE FEATURES FOR ALL 33 STOCKS
# ============================================================

stock_features <- stock_panel %>%
  
  group_split(
    Ticker
  ) %>%
  
  lapply(
    create_stock_features
  ) %>%
  
  bind_rows()

# ============================================================
# 5. COMMON PREDICTOR SET
# ============================================================

predictor_names <- c(
  
  # Lagged returns
  "ReturnLag1",
  "ReturnLag2",
  "ReturnLag3",
  "ReturnLag4",
  "ReturnLag5",
  
  
  # SMA distance
  "SMADist5",
  "SMADist10",
  "SMADist20",
  
  
  # EMA distance
  "EMADist5",
  "EMADist10",
  "EMADist20",
  
  
  # Momentum
  "Momentum5",
  "Momentum10",
  "Momentum20",
  
  
  # Technical indicators
  "RSI14",
  "MACDRel",
  "MACDSignalRel",
  "BBPosition",
  "StochK",
  "StochD",
  
  
  # Volatility
  "Vol5",
  "Vol20",
  
  
  # Trading activity
  "LogVolume",
  "RelVolume20",
  "DeltaOBV",
  
  
  # Overnight information
  "OvernightReturn",
  
  
  # Liquidity
  "ILLIQ20"
  
)

# ============================================================
# 6. REMOVE INITIAL WINDOW-INDUCED NAs
# ============================================================

model_data <- stock_features %>%
  
  filter(
    !is.na(Sample),
    !is.na(Target)
  ) %>%
  
  filter(
    if_all(
      all_of(predictor_names),
      ~ is.finite(.x)
    )
  ) %>%
  
  arrange(
    Ticker,
    Date
  )

# ============================================================
# 7. TRAINING-PERIOD STOCK CHARACTERISTICS
# ============================================================

stock_characteristics <- model_data %>%
  
  filter(
    Sample == "Training"
  ) %>%
  
  group_by(
    Ticker,
    sector
  ) %>%
  
  summarise(
    
    TrainingVolatility =
      sd(
        Return,
        na.rm = TRUE
      ) *
      sqrt(252),
    
    
    TrainingILLIQ =
      median(
        ILLIQ20,
        na.rm = TRUE
      ),
    
    
    TrainingDollarVolume =
      median(
        DollarVolume,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  )

# ============================================================
# 8. VOLATILITY AND LIQUIDITY GROUPS
# ============================================================

stock_characteristics <- stock_characteristics %>%
  
  mutate(
    
    VolatilityRank =
      ntile(
        TrainingVolatility,
        3
      ),
    
    
    VolatilityGroup =
      case_when(
        
        VolatilityRank == 1 ~
          "Low",
        
        VolatilityRank == 2 ~
          "Medium",
        
        VolatilityRank == 3 ~
          "High"
        
      ),
    
    
    # Larger Amihud = LESS liquid
    
    LiquidityRank =
      ntile(
        TrainingILLIQ,
        3
      ),
    
    
    LiquidityGroup =
      case_when(
        
        LiquidityRank == 1 ~
          "High",
        
        LiquidityRank == 2 ~
          "Medium",
        
        LiquidityRank == 3 ~
          "Low"
        
      )
    
  )

# ============================================================
# 9. TRAINING-ONLY STANDARDIZATION
# ============================================================

standardize_data <- function(
    train,
    validation,
    test,
    predictors
){
  
  means <- sapply(
    train[, predictors],
    mean
  )
  
  
  sds <- sapply(
    train[, predictors],
    sd
  )
  
  
  # Protect against zero-variance predictors
  
  sds[
    !is.finite(sds) |
      sds == 0
  ] <- 1
  
  
  scale_matrix <- function(df){
    
    x <- as.matrix(
      df[, predictors]
    )
    
    
    x <- sweep(
      x,
      2,
      means,
      "-"
    )
    
    
    x <- sweep(
      x,
      2,
      sds,
      "/"
    )
    
    
    return(x)
    
  }
  
  
  list(
    
    train =
      scale_matrix(train),
    
    validation =
      scale_matrix(validation),
    
    test =
      scale_matrix(test),
    
    means =
      means,
    
    sds =
      sds
    
  )
}

# ============================================================
# 10. FORECAST EVALUATION FUNCTION
# ============================================================

evaluate_forecast <- function(
    actual,
    predicted
){
  
  RMSE <- sqrt(
    mean(
      (actual - predicted)^2
    )
  )
  
  
  MAE <- mean(
    abs(
      actual - predicted
    )
  )
  
  
  # Zero-return benchmark
  
  BenchmarkRMSE <- sqrt(
    mean(
      actual^2
    )
  )
  
  
  BenchmarkMAE <- mean(
    abs(actual)
  )
  
  
  RelRMSE <-
    RMSE /
    BenchmarkRMSE
  
  
  RelMAE <-
    MAE /
    BenchmarkMAE
  
  
  RMSEGain <- 100 * (
    1 - RelRMSE
  )
  
  
  MAEGain <- 100 * (
    1 - RelMAE
  )
  
  
  R2_OOS <- 1 -
    (
      sum(
        (actual - predicted)^2
      ) /
        sum(
          actual^2
        )
    )
  
  
  DirectionalAccuracy <- mean(
    actual * predicted > 0
  )
  
  
  tibble(
    
    RMSE =
      RMSE,
    
    MAE =
      MAE,
    
    BenchmarkRMSE =
      BenchmarkRMSE,
    
    BenchmarkMAE =
      BenchmarkMAE,
    
    RelRMSE =
      RelRMSE,
    
    RelMAE =
      RelMAE,
    
    RMSEGain =
      RMSEGain,
    
    MAEGain =
      MAEGain,
    
    R2_OOS =
      R2_OOS,
    
    DirectionalAccuracy =
      DirectionalAccuracy
    
  )
}

# ============================================================
# TRAIN ALL TRADITIONAL ML MODELS FOR ONE STOCK
#
# Models:
# 1. OLS
# 2. Ridge
# 3. LASSO
# 4. Bagging
# 5. Random Forest
# 6. Gradient Boosting
# ============================================================

run_ml_stock <- function(stock_df, stock_number = 1) {
  
  # ----------------------------------------------------------
  # BASIC STOCK INFORMATION
  # ----------------------------------------------------------
  
  ticker <- unique(stock_df$Ticker)
  sector_name <- unique(stock_df$sector)
  
  if (length(ticker) != 1) {
    stop("stock_df must contain exactly one ticker.")
  }
  
  if (length(sector_name) != 1) {
    stop("stock_df must contain exactly one sector.")
  }
  
  cat(
    "\n========================================\n",
    "Running ML models for: ", ticker,
    " | ", sector_name,
    "\n========================================\n",
    sep = ""
  )
  
  
  # ----------------------------------------------------------
  # TRAIN / VALIDATION / TEST SPLIT
  # ----------------------------------------------------------
  
  train <- stock_df %>%
    dplyr::filter(Sample == "Training")
  
  validation <- stock_df %>%
    dplyr::filter(Sample == "Validation")
  
  test <- stock_df %>%
    dplyr::filter(Sample == "Test")
  
  
  # Safety checks
  
  if (nrow(train) == 0) {
    stop(paste("No training observations for", ticker))
  }
  
  if (nrow(validation) == 0) {
    stop(paste("No validation observations for", ticker))
  }
  
  if (nrow(test) == 0) {
    stop(paste("No test observations for", ticker))
  }
  
  
  # Targets
  
  y_train <- train$Target
  y_validation <- validation$Target
  y_test <- test$Target
  
  
  # ----------------------------------------------------------
  # SIMPLE RMSE FUNCTION FOR VALIDATION
  # ----------------------------------------------------------
  
  calc_rmse <- function(actual, predicted) {
    
    sqrt(
      mean(
        (actual - predicted)^2
      )
    )
    
  }
  
  
  # ----------------------------------------------------------
  # STANDARDIZATION
  #
  # Parameters are estimated using training data only.
  # ----------------------------------------------------------
  
  scaled <- standardize_data(
    train = train,
    validation = validation,
    test = test,
    predictors = predictor_names
  )
  
  x_train <- scaled$train
  x_validation <- scaled$validation
  x_test <- scaled$test
  
  
  # ----------------------------------------------------------
  # STORAGE OBJECT
  # ----------------------------------------------------------
  
  model_outputs <- list()
  
  
  # ==========================================================
  # MODEL 1: OLS
  # ==========================================================
  
  ols_train <- data.frame(
    Target = y_train,
    x_train
  )
  
  ols_validation <- data.frame(
    x_validation
  )
  
  ols_test <- data.frame(
    x_test
  )
  
  
  ols_model <- stats::lm(
    Target ~ .,
    data = ols_train
  )
  
  
  ols_val_pred <- as.numeric(
    stats::predict(
      ols_model,
      newdata = ols_validation
    )
  )
  
  
  ols_test_pred <- as.numeric(
    stats::predict(
      ols_model,
      newdata = ols_test
    )
  )
  
  
  ols_val_rmse <- calc_rmse(
    y_validation,
    ols_val_pred
  )
  
  
  model_outputs[["OLS"]] <- list(
    prediction = ols_test_pred,
    validation_rmse = ols_val_rmse,
    hyperparameters = "None"
  )
  
  
  cat("   OLS completed\n")
  
  
  # ==========================================================
  # MODEL 2: RIDGE
  # ==========================================================
  
  ridge_fit <- glmnet::glmnet(
    x = x_train,
    y = y_train,
    alpha = 0,
    standardize = FALSE,
    intercept = TRUE,
    nlambda = 100
  )
  
  
  ridge_val_predictions <- predict(
    ridge_fit,
    newx = x_validation,
    s = ridge_fit$lambda
  )
  
  
  ridge_val_rmse <- apply(
    ridge_val_predictions,
    MARGIN = 2,
    FUN = function(pred) {
      calc_rmse(
        y_validation,
        pred
      )
    }
  )
  
  
  ridge_best_index <- which.min(
    ridge_val_rmse
  )
  
  
  ridge_best_lambda <- ridge_fit$lambda[
    ridge_best_index
  ]
  
  
  ridge_test_pred <- as.numeric(
    predict(
      ridge_fit,
      newx = x_test,
      s = ridge_best_lambda
    )
  )
  
  
  model_outputs[["Ridge"]] <- list(
    prediction = ridge_test_pred,
    validation_rmse = ridge_val_rmse[ridge_best_index],
    hyperparameters = paste0(
      "lambda=",
      signif(ridge_best_lambda, 5)
    )
  )
  
  
  cat("   Ridge completed\n")
  
  
  # ==========================================================
  # MODEL 3: LASSO
  # ==========================================================
  
  lasso_fit <- glmnet::glmnet(
    x = x_train,
    y = y_train,
    alpha = 1,
    standardize = FALSE,
    intercept = TRUE,
    nlambda = 100
  )
  
  
  lasso_val_predictions <- predict(
    lasso_fit,
    newx = x_validation,
    s = lasso_fit$lambda
  )
  
  
  lasso_val_rmse <- apply(
    lasso_val_predictions,
    MARGIN = 2,
    FUN = function(pred) {
      calc_rmse(
        y_validation,
        pred
      )
    }
  )
  
  
  lasso_best_index <- which.min(
    lasso_val_rmse
  )
  
  
  lasso_best_lambda <- lasso_fit$lambda[
    lasso_best_index
  ]
  
  
  lasso_test_pred <- as.numeric(
    predict(
      lasso_fit,
      newx = x_test,
      s = lasso_best_lambda
    )
  )
  
  
  model_outputs[["LASSO"]] <- list(
    prediction = lasso_test_pred,
    validation_rmse = lasso_val_rmse[lasso_best_index],
    hyperparameters = paste0(
      "lambda=",
      signif(lasso_best_lambda, 5)
    )
  )
  
  
  cat("   LASSO completed\n")
  
  
  # ==========================================================
  # DATA FOR TREE-BASED MODELS
  # ==========================================================
  
  tree_train <- train %>%
    dplyr::select(
      Target,
      dplyr::all_of(predictor_names)
    ) %>%
    as.data.frame()
  
  
  tree_validation <- validation %>%
    dplyr::select(
      dplyr::all_of(predictor_names)
    ) %>%
    as.data.frame()
  
  
  tree_test <- test %>%
    dplyr::select(
      dplyr::all_of(predictor_names)
    ) %>%
    as.data.frame()
  
  
  # ==========================================================
  # MODEL 4: BAGGING
  # ==========================================================
  
  bagging_nodesize_grid <- c(
    5,
    10,
    20
  )
  
  
  bagging_best_rmse <- Inf
  bagging_best_model <- NULL
  bagging_best_nodesize <- NA_integer_
  
  
  for (nodesize in bagging_nodesize_grid) {
    
    seed_value <- 1000 +
      stock_number +
      nodesize
    
    
    set.seed(seed_value)
    
    
    current_model <- ranger::ranger(
      Target ~ .,
      data = tree_train,
      num.trees = 500,
      mtry = length(predictor_names),
      min.node.size = nodesize,
      replace = TRUE,
      sample.fraction = 1,
      seed = seed_value
    )
    
    
    current_prediction <- predict(
      current_model,
      data = tree_validation
    )$predictions
    
    
    current_rmse <- calc_rmse(
      y_validation,
      current_prediction
    )
    
    
    if (current_rmse < bagging_best_rmse) {
      
      bagging_best_rmse <- current_rmse
      bagging_best_model <- current_model
      bagging_best_nodesize <- nodesize
      
    }
    
  }
  
  
  bagging_test_pred <- predict(
    bagging_best_model,
    data = tree_test
  )$predictions
  
  
  model_outputs[["Bagging"]] <- list(
    prediction = bagging_test_pred,
    validation_rmse = bagging_best_rmse,
    hyperparameters = paste0(
      "trees=500; min.node.size=",
      bagging_best_nodesize
    )
  )
  
  
  cat("   Bagging completed\n")
  
  
  # ==========================================================
  # MODEL 5: RANDOM FOREST
  # ==========================================================
  
  p <- length(predictor_names)
  
  
  rf_mtry_grid <- unique(
    c(
      floor(sqrt(p)),
      floor(p / 3),
      floor(p / 2)
    )
  )
  
  
  rf_mtry_grid <- rf_mtry_grid[
    rf_mtry_grid >= 1 &
      rf_mtry_grid < p
  ]
  
  
  rf_nodesize_grid <- c(
    5,
    10,
    20
  )
  
  
  rf_best_rmse <- Inf
  rf_best_model <- NULL
  rf_best_mtry <- NA_integer_
  rf_best_nodesize <- NA_integer_
  
  
  for (mtry_value in rf_mtry_grid) {
    
    for (nodesize in rf_nodesize_grid) {
      
      seed_value <- 2000 +
        stock_number +
        mtry_value * 10 +
        nodesize
      
      
      set.seed(seed_value)
      
      
      current_model <- ranger::ranger(
        Target ~ .,
        data = tree_train,
        num.trees = 500,
        mtry = mtry_value,
        min.node.size = nodesize,
        replace = TRUE,
        sample.fraction = 1,
        seed = seed_value
      )
      
      
      current_prediction <- predict(
        current_model,
        data = tree_validation
      )$predictions
      
      
      current_rmse <- calc_rmse(
        y_validation,
        current_prediction
      )
      
      
      if (current_rmse < rf_best_rmse) {
        
        rf_best_rmse <- current_rmse
        rf_best_model <- current_model
        rf_best_mtry <- mtry_value
        rf_best_nodesize <- nodesize
        
      }
      
    }
    
  }
  
  
  rf_test_pred <- predict(
    rf_best_model,
    data = tree_test
  )$predictions
  
  
  model_outputs[["Random Forest"]] <- list(
    prediction = rf_test_pred,
    validation_rmse = rf_best_rmse,
    hyperparameters = paste0(
      "trees=500; mtry=",
      rf_best_mtry,
      "; min.node.size=",
      rf_best_nodesize
    )
  )
  
  
  cat("   Random Forest completed\n")
  
  
  # ==========================================================
  # MODEL 6: GRADIENT BOOSTING
  # ==========================================================
  
  boost_depth_grid <- c(
    1,
    2,
    3
  )
  
  
  boost_shrinkage_grid <- c(
    0.01,
    0.05
  )
  
  
  boost_tree_grid <- c(
    100,
    300,
    500,
    1000
  )
  
  
  boost_best_rmse <- Inf
  boost_best_model <- NULL
  boost_best_depth <- NA_integer_
  boost_best_shrinkage <- NA_real_
  boost_best_trees <- NA_integer_
  
  
  for (depth in boost_depth_grid) {
    
    for (shrinkage_value in boost_shrinkage_grid) {
      
      seed_value <- 3000 +
        stock_number +
        depth * 10 +
        round(shrinkage_value * 100)
      
      
      set.seed(seed_value)
      
      
      current_boost_model <- gbm::gbm(
        formula = Target ~ .,
        data = tree_train,
        distribution = "gaussian",
        n.trees = max(boost_tree_grid),
        interaction.depth = depth,
        shrinkage = shrinkage_value,
        n.minobsinnode = 10,
        bag.fraction = 1,
        train.fraction = 1,
        verbose = FALSE
      )
      
      
      for (tree_count in boost_tree_grid) {
        
        current_prediction <- as.numeric(
          predict(
            current_boost_model,
            newdata = tree_validation,
            n.trees = tree_count
          )
        )
        
        
        current_rmse <- calc_rmse(
          y_validation,
          current_prediction
        )
        
        
        if (current_rmse < boost_best_rmse) {
          
          boost_best_rmse <- current_rmse
          boost_best_model <- current_boost_model
          boost_best_depth <- depth
          boost_best_shrinkage <- shrinkage_value
          boost_best_trees <- tree_count
          
        }
        
      }
      
    }
    
  }
  
  
  boost_test_pred <- as.numeric(
    predict(
      boost_best_model,
      newdata = tree_test,
      n.trees = boost_best_trees
    )
  )
  
  
  model_outputs[["Gradient Boosting"]] <- list(
    prediction = boost_test_pred,
    validation_rmse = boost_best_rmse,
    hyperparameters = paste0(
      "trees=",
      boost_best_trees,
      "; depth=",
      boost_best_depth,
      "; shrinkage=",
      boost_best_shrinkage
    )
  )
  
  
  cat("   Gradient Boosting completed\n")
  
  
  # ==========================================================
  # TEST-PERIOD METRICS
  # ==========================================================
  
  summary_table <- dplyr::bind_rows(
    
    lapply(
      
      names(model_outputs),
      
      function(model_name) {
        
        output <- model_outputs[[model_name]]
        
        
        metrics <- evaluate_forecast(
          actual = y_test,
          predicted = output$prediction
        )
        
        
        dplyr::bind_cols(
          
          tibble::tibble(
            
            Ticker = ticker,
            sector = sector_name,
            Model = model_name,
            ValidationRMSE = output$validation_rmse,
            Hyperparameters = output$hyperparameters,
            TestObservations = length(y_test)
            
          ),
          
          metrics
          
        )
        
      }
      
    )
    
  )
  
  
  # ==========================================================
  # DAILY TEST-PERIOD FORECASTS
  # ==========================================================
  
  prediction_table <- dplyr::bind_rows(
    
    lapply(
      
      names(model_outputs),
      
      function(model_name) {
        
        output <- model_outputs[[model_name]]
        
        
        tibble::tibble(
          
          Ticker = ticker,
          sector = sector_name,
          Model = model_name,
          ForecastOrigin = test$Date,
          TargetDate = test$TargetDate,
          ActualReturn = y_test,
          Prediction = output$prediction
          
        )
        
      }
      
    )
    
  )
  
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  cat(
    "Completed:",
    ticker,
    "\n"
  )
  
  
  return(
    list(
      summary = summary_table,
      predictions = prediction_table
    )
  )
  
}

# ============================================================
# 12. RUN MACHINE LEARNING FOR ALL 33 STOCKS
# ============================================================

ticker_list <- unique(
  model_data$Ticker
)


ml_outputs <- vector(
  
  mode = "list",
  
  length =
    length(
      ticker_list
    )
  
)


names(
  ml_outputs
) <- ticker_list


for(i in seq_along(
  ticker_list
)){
  
  ticker <-
    ticker_list[i]
  
  
  stock_df <- model_data %>%
    
    filter(
      Ticker == ticker
    )
  
  
  ml_outputs[[ticker]] <-
    run_ml_stock(
      
      stock_df =
        stock_df,
      
      stock_number =
        i
      
    )
  
}
 
# ============================================================
# 13. SINGLE MASTER MACHINE-LEARNING TABLE
# ============================================================

ml_results_table <- bind_rows(
  
  lapply(
    
    ml_outputs,
    
    function(x){
      x$summary
    }
    
  )
)

ml_results_table <- ml_results_table %>%
  
  left_join(
    
    stock_characteristics,
    
    by = c(
      "Ticker",
      "sector"
    )
    
  ) %>%
  
  select(
    
    Sector =
      sector,
    
    Ticker,
    
    VolatilityGroup,
    
    LiquidityGroup,
    
    TrainingVolatility,
    
    TrainingILLIQ,
    
    TrainingDollarVolume,
    
    Model,
    
    Hyperparameters,
    
    ValidationRMSE,
    
    TestObservations,
    
    RMSE,
    
    MAE,
    
    BenchmarkRMSE,
    
    BenchmarkMAE,
    
    RelRMSE,
    
    RelMAE,
    
    RMSEGain,
    
    MAEGain,
    
    R2_OOS,
    
    DirectionalAccuracy
    
  ) %>%
  
  arrange(
    
    Sector,
    
    Ticker,
    
    Model
    
  )

View(
  ml_results_table
)

print(
  ml_results_table,
  n = Inf
)

# ============================================================
# 14. MASTER DAILY PREDICTION DATASET
# ============================================================

ml_predictions <- bind_rows(
  
  lapply(
    
    ml_outputs,
    
    function(x){
      x$predictions
    }
    
  )
)

head(
  ml_predictions
)

# ============================================================
# CHECK FINAL ML RESULTS
# ============================================================

nrow(ml_results_table)

length(
  unique(
    ml_results_table$Ticker
  )
)

length(
  unique(
    ml_results_table$Model
  )
)

ml_results_table %>%
  summarise(
    
    Missing_RMSE =
      sum(is.na(RMSE)),
    
    Missing_MAE =
      sum(is.na(MAE)),
    
    Missing_R2 =
      sum(is.na(R2_OOS)),
    
    Missing_DA =
      sum(is.na(DirectionalAccuracy))
    
  )

# ============================================================
# TABLE 1:
# ALL 198 ML RESULTS RANKED BY PREDICTIVE ACCURACY
# ============================================================

ml_accuracy_ranking <- ml_results_table %>%
  
  arrange(
    desc(RMSEGain)
  ) %>%
  
  mutate(
    OverallRank = row_number()
  ) %>%
  
  select(
    OverallRank,
    Sector,
    Ticker,
    VolatilityGroup,
    LiquidityGroup,
    Model,
    RMSE,
    RelRMSE,
    RMSEGain,
    MAE,
    RelMAE,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy
  )

View(
  ml_accuracy_ranking
)

ml_accuracy_ranking <- ml_accuracy_ranking %>%
  
  mutate(
    
    RMSE_pct =
      100 * RMSE,
    
    MAE_pct =
      100 * MAE,
    
    DirectionalAccuracy_pct =
      100 * DirectionalAccuracy
    
  )

ml_accuracy_table <- ml_accuracy_ranking %>%
  
  select(
    OverallRank,
    Sector,
    Ticker,
    VolatilityGroup,
    LiquidityGroup,
    Model,
    RMSE_pct,
    RMSEGain,
    MAE_pct,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy_pct
  ) %>%
  
  mutate(
    
    across(
      c(
        RMSE_pct,
        RMSEGain,
        MAE_pct,
        MAEGain,
        R2_OOS,
        DirectionalAccuracy_pct
      ),
      ~ round(.x, 3)
    )
    
  )

View(
  ml_accuracy_table
)

# ============================================================
# TABLE 2:
# RANKING WITHIN EACH SECTOR
# ============================================================

sector_accuracy_ranking <- ml_results_table %>%
  
  group_by(
    Sector
  ) %>%
  
  arrange(
    desc(RMSEGain),
    .by_group = TRUE
  ) %>%
  
  mutate(
    SectorRank = row_number()
  ) %>%
  
  ungroup() %>%
  
  select(
    Sector,
    SectorRank,
    Ticker,
    Model,
    VolatilityGroup,
    LiquidityGroup,
    RMSEGain,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy
  )

View(
  sector_accuracy_ranking
)

# ============================================================
# TABLE 3:
# RESULTS THAT BEAT THE ZERO-RETURN BENCHMARK
# ============================================================

ml_positive_results <- ml_results_table %>%
  
  filter(
    RMSEGain > 0
  ) %>%
  
  arrange(
    desc(RMSEGain)
  ) %>%
  
  select(
    Sector,
    Ticker,
    Model,
    RMSEGain,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy
  )

nrow(
  ml_positive_results
)

mean(
  ml_results_table$RMSEGain > 0
) * 100

# ============================================================
# TABLE 4:
# PERFORMANCE BY MACHINE LEARNING MODEL
# ============================================================

model_summary <- ml_results_table %>%
  
  group_by(
    Model
  ) %>%
  
  summarise(
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    Median_MAEGain =
      median(
        MAEGain,
        na.rm = TRUE
      ),
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    Mean_Directional_Accuracy =
      mean(
        DirectionalAccuracy,
        na.rm = TRUE
      ) * 100,
    
    Stocks_Beating_Benchmark =
      sum(
        RMSEGain > 0,
        na.rm = TRUE
      ),
    
    Percentage_Beating_Benchmark =
      mean(
        RMSEGain > 0,
        na.rm = TRUE
      ) * 100,
    
    .groups = "drop"
    
  ) %>%
  
  arrange(
    desc(
      Mean_RMSEGain
    )
  )

View(
  model_summary
)

# ============================================================
# TABLE 5:
# MODEL PERFORMANCE BY SECTOR
# ============================================================

sector_model_summary <- ml_results_table %>%
  
  group_by(
    Sector,
    Model
  ) %>%
  
  summarise(
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    Mean_DirectionalAccuracy =
      mean(
        DirectionalAccuracy,
        na.rm = TRUE
      ) * 100,
    
    Stocks_Beating_Benchmark =
      sum(
        RMSEGain > 0,
        na.rm = TRUE
      ),
    
    .groups = "drop"
    
  ) %>%
  
  arrange(
    Sector,
    desc(Mean_RMSEGain)
  )

View(
  sector_model_summary
)

# ============================================================
# TABLE 6:
# SELECTED ML MODEL PER STOCK
#
# IMPORTANT:
# selection uses VALIDATION performance only
# ============================================================

selected_ml_models <- ml_results_table %>%
  
  group_by(
    Ticker
  ) %>%
  
  slice_min(
    order_by = ValidationRMSE,
    n = 1,
    with_ties = FALSE
  ) %>%
  
  ungroup() %>%
  
  arrange(
    Sector,
    Ticker
  )

selected_ml_ranking <- selected_ml_models %>%
  
  arrange(
    desc(
      RMSEGain
    )
  ) %>%
  
  mutate(
    Rank = row_number()
  ) %>%
  
  select(
    Rank,
    Sector,
    Ticker,
    Model,
    VolatilityGroup,
    LiquidityGroup,
    ValidationRMSE,
    RMSE,
    RMSEGain,
    MAE,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy
  )

# ============================================================
# TABLE 7:
# DIRECTIONAL FORECASTING PERFORMANCE
# ============================================================

directional_ranking <- ml_results_table %>%
  
  arrange(
    desc(
      DirectionalAccuracy
    )
  ) %>%
  
  mutate(
    DirectionRank =
      row_number()
  ) %>%
  
  select(
    DirectionRank,
    Sector,
    Ticker,
    Model,
    DirectionalAccuracy,
    RMSEGain,
    R2_OOS
  )

# ============================================================
# TABLES 8-13
# FINAL TRADITIONAL ML ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------

if (!requireNamespace("forecast", quietly = TRUE)) {
  install.packages("forecast")
}

library(dplyr)
library(tidyr)
library(tibble)


# ------------------------------------------------------------
# 1. CHECK REQUIRED OBJECTS
# ------------------------------------------------------------

required_objects <- c(
  "ml_results_table",
  "ml_predictions",
  "model_data"
)

missing_objects <- required_objects[
  !sapply(required_objects, exists)
]

if (length(missing_objects) > 0) {
  
  stop(
    paste(
      "Missing objects:",
      paste(missing_objects, collapse = ", ")
    )
  )
}


# ------------------------------------------------------------
# 2. RECREATE SELECTED ML MODELS IF NECESSARY
#
# IMPORTANT:
# Selection is based ONLY on validation RMSE.
# ------------------------------------------------------------

if (!exists("selected_ml_models")) {
  
  selected_ml_models <- ml_results_table %>%
    
    group_by(Ticker) %>%
    
    slice_min(
      order_by = ValidationRMSE,
      n = 1,
      with_ties = FALSE
    ) %>%
    
    ungroup()
}


# ------------------------------------------------------------
# 3. KEEP ONLY DAILY PREDICTIONS FROM THE
#    VALIDATION-SELECTED MODEL
# ------------------------------------------------------------

selection_map <- selected_ml_models %>%
  
  select(
    Ticker,
    SelectedModel = Model
  )


selected_ml_predictions <- ml_predictions %>%
  
  inner_join(
    selection_map,
    by = "Ticker"
  ) %>%
  
  filter(
    Model == SelectedModel
  ) %>%
  
  select(
    -SelectedModel
  ) %>%
  
  arrange(
    Ticker,
    TargetDate
  )


# Check
length(
  unique(
    selected_ml_predictions$Ticker
  )
)

# ============================================================
# CORRECT PESARAN-TIMMERMANN TEST
#
# Based on the standard PT implementation.
# H0: predicted and realized directions are independent
# H1: positive directional predictability
# ============================================================

pesaran_timmermann <- function(actual, predicted) {
  
  # ----------------------------------------------------------
  # Keep only finite observations
  # ----------------------------------------------------------
  
  ok <- is.finite(actual) & is.finite(predicted)
  
  actual <- as.numeric(actual[ok])
  predicted <- as.numeric(predicted[ok])
  
  
  n <- length(actual)
  
  
  if (n < 10) {
    
    return(
      tibble::tibble(
        N_Directional = n,
        ZeroForecasts = NA_integer_,
        DA_PT = NA_real_,
        ExpectedDA = NA_real_,
        PT_Statistic = NA_real_,
        PT_pvalue = NA_real_
      )
    )
  }
  
  
  # ----------------------------------------------------------
  # Count exact zero forecasts
  # ----------------------------------------------------------
  
  zero_forecasts <- sum(
    predicted == 0
  )
  
  
  # ----------------------------------------------------------
  # Binary direction indicators
  #
  # x_t = realized return positive
  # y_t = forecast positive
  # ----------------------------------------------------------
  
  x_t <- rep(0, n)
  y_t <- rep(0, n)
  z_t <- rep(0, n)
  
  
  x_t[actual > 0] <- 1
  
  y_t[predicted > 0] <- 1
  
  
  # Correct directional prediction
  
  z_t[
    (predicted * actual) > 0
  ] <- 1
  
  
  # ----------------------------------------------------------
  # Marginal probabilities
  # ----------------------------------------------------------
  
  p_x <- mean(x_t)
  
  p_y <- mean(y_t)
  
  
  # Observed directional accuracy
  
  p_hat <- mean(z_t)
  
  
  # Expected accuracy under independence
  
  p_star <-
    p_y * p_x +
    (1 - p_y) *
    (1 - p_x)
  
  
  # ----------------------------------------------------------
  # Variance of observed success rate
  # ----------------------------------------------------------
  
  p_hat_var <-
    (
      p_star *
        (1 - p_star)
    ) / n
  
  
  # ----------------------------------------------------------
  # Variance associated with expected success probability
  #
  # IMPORTANT:
  # the THIRD term is divided by n^2
  # ----------------------------------------------------------
  
  p_star_var <-
    
    (
      (2 * p_y - 1)^2 *
        p_x *
        (1 - p_x)
    ) / n +
    
    (
      (2 * p_x - 1)^2 *
        p_y *
        (1 - p_y)
    ) / n +
    
    (
      4 *
        p_x *
        p_y *
        (1 - p_x) *
        (1 - p_y)
    ) / (n^2)
  
  
  # ----------------------------------------------------------
  # PT denominator
  # ----------------------------------------------------------
  
  variance_difference <-
    p_hat_var -
    p_star_var
  
  
  if (
    !is.finite(variance_difference) ||
    variance_difference <= 0
  ) {
    
    pt_stat <- NA_real_
    pt_pvalue <- NA_real_
    
  } else {
    
    # --------------------------------------------------------
    # Pesaran-Timmermann statistic
    # --------------------------------------------------------
    
    pt_stat <-
      (
        p_hat -
          p_star
      ) /
      sqrt(
        variance_difference
      )
    
    
    # --------------------------------------------------------
    # One-sided p-value:
    #
    # H1 = positive directional predictability
    # --------------------------------------------------------
    
    pt_pvalue <-
      1 -
      pnorm(
        pt_stat
      )
    
  }
  
  
  return(
    
    tibble::tibble(
      
      N_Directional =
        n,
      
      ZeroForecasts =
        zero_forecasts,
      
      DA_PT =
        p_hat,
      
      ExpectedDA =
        p_star,
      
      PT_Statistic =
        pt_stat,
      
      PT_pvalue =
        pt_pvalue
      
    )
    
  )
}

# ============================================================
# REBUILD TABLE 8
# ============================================================

table8_pt_raw <- selected_ml_predictions %>%
  
  group_by(
    Ticker,
    Model
  ) %>%
  
  group_modify(
    
    ~ pesaran_timmermann(
      
      actual =
        .x$ActualReturn,
      
      predicted =
        .x$Prediction
      
    )
    
  ) %>%
  
  ungroup()

table8_pt <- selected_ml_models %>%
  
  select(
    Sector,
    Ticker,
    VolatilityGroup,
    LiquidityGroup,
    Model,
    DirectionalAccuracy
  ) %>%
  
  left_join(
    
    table8_pt_raw,
    
    by = c(
      "Ticker",
      "Model"
    )
    
  ) %>%
  
  mutate(
    
    DA_Percent =
      100 *
      DirectionalAccuracy,
    
    
    ExpectedDA_Percent =
      100 *
      ExpectedDA,
    
    
    Significant_10pct =
      PT_pvalue < 0.10,
    
    
    Significant_5pct =
      PT_pvalue < 0.05,
    
    
    Significant_1pct =
      PT_pvalue < 0.01
    
  ) %>%
  
  arrange(
    PT_pvalue
  )


View(
  table8_pt
)

# ============================================================
# DIAGNOSTIC:
# HOW OFTEN DOES EACH SELECTED MODEL FORECAST POSITIVE RETURNS?
# ============================================================

pt_diagnostic <- selected_ml_predictions %>%
  
  group_by(
    Ticker,
    Model
  ) %>%
  
  summarise(
    
    N =
      n(),
    
    ActualPositiveShare =
      mean(
        ActualReturn > 0,
        na.rm = TRUE
      ),
    
    ForecastPositiveShare =
      mean(
        Prediction > 0,
        na.rm = TRUE
      ),
    
    ForecastNegativeShare =
      mean(
        Prediction < 0,
        na.rm = TRUE
      ),
    
    ForecastZeroShare =
      mean(
        Prediction == 0,
        na.rm = TRUE
      ),
    
    UniqueForecastSigns =
      n_distinct(
        sign(Prediction)
      ),
    
    .groups = "drop"
  ) %>%
  
  left_join(
    
    table8_pt %>%
      select(
        Ticker,
        PT_Statistic,
        PT_pvalue
      ),
    
    by = "Ticker"
  ) %>%
  
  arrange(
    ForecastPositiveShare
  )


View(pt_diagnostic)

table8_pt <- table8_pt %>%
  
  mutate(
    
    PT_pvalue_FDR = p.adjust(
      PT_pvalue,
      method = "BH"
    ),
    
    Significant_FDR_5pct =
      PT_pvalue_FDR < 0.05
    
  )

table8_pt %>%
  
  select(
    Ticker,
    Model,
    DA_Percent,
    ExpectedDA_Percent,
    PT_Statistic,
    PT_pvalue,
    PT_pvalue_FDR,
    Significant_5pct,
    Significant_FDR_5pct
  ) %>%
  
  arrange(PT_pvalue)

# ============================================================
# TABLE 7:
# DIRECTIONAL PERFORMANCE OF VALIDATION-SELECTED ML MODELS
# ============================================================

directional_ranking <- selected_ml_models %>%
  
  arrange(
    desc(DirectionalAccuracy)
  ) %>%
  
  mutate(
    DirectionRank = row_number()
  ) %>%
  
  select(
    DirectionRank,
    Sector,
    Ticker,
    Model,
    VolatilityGroup,
    LiquidityGroup,
    DirectionalAccuracy,
    RMSEGain,
    MAEGain,
    R2_OOS
  )


View(directional_ranking)

# ============================================================
# FINALIZE TABLE 8
# ============================================================

table8_final <- table8_pt %>%
  
  left_join(
    
    pt_diagnostic %>%
      select(
        Ticker,
        Model,
        ActualPositiveShare,
        ForecastPositiveShare,
        ForecastNegativeShare,
        UniqueForecastSigns
      ),
    
    by = c(
      "Ticker",
      "Model"
    )
    
  ) %>%
  
  mutate(
    
    PT_Status = case_when(
      
      is.na(PT_pvalue) &
        UniqueForecastSigns == 1 ~
        "Not testable: constant forecast direction",
      
      is.na(PT_pvalue) ~
        "Not testable: degenerate PT variance",
      
      PT_pvalue < 0.05 ~
        "Significant directional predictability",
      
      TRUE ~
        "No significant directional predictability"
      
    )
    
  ) %>%
  
  arrange(
    is.na(PT_pvalue),
    PT_pvalue
  )


View(table8_final)

# ============================================================
# TABLE 9:
# DIEBOLD-MARIANO TEST
#
# Selected ML model vs zero-return benchmark
# ============================================================


dm_vs_zero <- function(actual, predicted) {
  
  # Keep valid observations
  
  ok <- is.finite(actual) &
    is.finite(predicted)
  
  actual <- as.numeric(actual[ok])
  predicted <- as.numeric(predicted[ok])
  
  
  # ----------------------------------------------------------
  # FORECAST ERRORS
  # ----------------------------------------------------------
  
  # Selected ML model
  
  error_ml <-
    actual - predicted
  
  
  # Zero-return benchmark:
  # prediction = 0
  
  error_zero <-
    actual
  
  
  # ----------------------------------------------------------
  # LOSS DIFFERENTIALS
  #
  # Negative value:
  # ML has LOWER loss than benchmark
  # ----------------------------------------------------------
  
  squared_loss_diff <-
    error_ml^2 -
    error_zero^2
  
  
  absolute_loss_diff <-
    abs(error_ml) -
    abs(error_zero)
  
  
  # ----------------------------------------------------------
  # DIEBOLD-MARIANO:
  # SQUARED LOSS
  # ----------------------------------------------------------
  
  dm_squared <- tryCatch(
    
    forecast::dm.test(
      
      e1 = error_ml,
      e2 = error_zero,
      
      alternative = "two.sided",
      
      h = 1,
      
      power = 2
      
    ),
    
    error = function(e) NULL
    
  )
  
  
  # ----------------------------------------------------------
  # DIEBOLD-MARIANO:
  # ABSOLUTE LOSS
  # ----------------------------------------------------------
  
  dm_absolute <- tryCatch(
    
    forecast::dm.test(
      
      e1 = error_ml,
      e2 = error_zero,
      
      alternative = "two.sided",
      
      h = 1,
      
      power = 1
      
    ),
    
    error = function(e) NULL
    
  )
  
  
  # ----------------------------------------------------------
  # RETURN RESULTS
  # ----------------------------------------------------------
  
  tibble::tibble(
    
    N =
      length(actual),
    
    
    MeanSquaredLossDiff =
      mean(squared_loss_diff),
    
    
    DM_Squared_Statistic =
      if (is.null(dm_squared)) {
        NA_real_
      } else {
        as.numeric(dm_squared$statistic)
      },
    
    
    DM_Squared_pvalue =
      if (is.null(dm_squared)) {
        NA_real_
      } else {
        dm_squared$p.value
      },
    
    
    MeanAbsoluteLossDiff =
      mean(absolute_loss_diff),
    
    
    DM_Absolute_Statistic =
      if (is.null(dm_absolute)) {
        NA_real_
      } else {
        as.numeric(dm_absolute$statistic)
      },
    
    
    DM_Absolute_pvalue =
      if (is.null(dm_absolute)) {
        NA_real_
      } else {
        dm_absolute$p.value
      }
    
  )
}

# ============================================================
# RUN DM TEST FOR ALL 33 SELECTED ML MODELS
# ============================================================

table9_dm_raw <- selected_ml_predictions %>%
  
  group_by(
    Ticker,
    Model
  ) %>%
  
  group_modify(
    
    ~ dm_vs_zero(
      
      actual =
        .x$ActualReturn,
      
      predicted =
        .x$Prediction
      
    )
    
  ) %>%
  
  ungroup()

# ============================================================
# BUILD TABLE 9
# ============================================================

table9_dm <- selected_ml_models %>%
  
  select(
    Sector,
    Ticker,
    VolatilityGroup,
    LiquidityGroup,
    Model,
    RMSEGain,
    MAEGain,
    R2_OOS
  ) %>%
  
  left_join(
    
    table9_dm_raw,
    
    by = c(
      "Ticker",
      "Model"
    )
    
  ) %>%
  
  mutate(
    
    # Negative loss differential means ML performed better
    
    ML_Better_RMSE =
      MeanSquaredLossDiff < 0,
    
    ML_Better_MAE =
      MeanAbsoluteLossDiff < 0,
    
    
    Significant_RMSE_5pct =
      ML_Better_RMSE &
      DM_Squared_pvalue < 0.05,
    
    
    Significant_MAE_5pct =
      ML_Better_MAE &
      DM_Absolute_pvalue < 0.05
    
  ) %>%
  
  arrange(
    DM_Squared_pvalue
  )


View(table9_dm)

table9_dm <- table9_dm %>%
  
  mutate(
    
    DM_Squared_pvalue_FDR =
      p.adjust(
        DM_Squared_pvalue,
        method = "BH"
      ),
    
    
    DM_Absolute_pvalue_FDR =
      p.adjust(
        DM_Absolute_pvalue,
        method = "BH"
      ),
    
    
    Significant_RMSE_FDR_5pct =
      ML_Better_RMSE &
      DM_Squared_pvalue_FDR < 0.05,
    
    
    Significant_MAE_FDR_5pct =
      ML_Better_MAE &
      DM_Absolute_pvalue_FDR < 0.05
    
  )
view(table9_dm)

table9_dm %>%
  
  select(
    Sector,
    Ticker,
    Model,
    RMSEGain,
    DM_Squared_Statistic,
    DM_Squared_pvalue,
    DM_Squared_pvalue_FDR,
    MAEGain,
    DM_Absolute_Statistic,
    DM_Absolute_pvalue,
    DM_Absolute_pvalue_FDR
  ) %>%
  
  arrange(
    DM_Squared_pvalue
  )
view(table9_dm)

# ============================================================
# MASTER DATASET FOR FINAL ML ANALYSIS
#
# One row per stock
# Model chosen ONLY using validation RMSE
# ============================================================


ml_selected_analysis <- selected_ml_models %>%
  
  # ----------------------------------------------------------
# Add Pesaran-Timmermann results
# ----------------------------------------------------------

left_join(
  
  table8_pt %>%
    
    select(
      Ticker,
      Model,
      ExpectedDA,
      PT_Statistic,
      PT_pvalue,
      PT_pvalue_FDR,
      Significant_5pct,
      Significant_FDR_5pct
    ) %>%
    
    rename(
      PT_Significant_5pct = Significant_5pct,
      PT_Significant_FDR_5pct = Significant_FDR_5pct
    ),
  
  by = c(
    "Ticker",
    "Model"
  )
  
) %>%
  
  
  # ----------------------------------------------------------
# Add Diebold-Mariano results
# ----------------------------------------------------------

left_join(
  
  table9_dm %>%
    
    select(
      Ticker,
      Model,
      
      MeanSquaredLossDiff,
      DM_Squared_Statistic,
      DM_Squared_pvalue,
      DM_Squared_pvalue_FDR,
      
      MeanAbsoluteLossDiff,
      DM_Absolute_Statistic,
      DM_Absolute_pvalue,
      DM_Absolute_pvalue_FDR,
      
      ML_Better_RMSE,
      ML_Better_MAE,
      
      Significant_RMSE_5pct,
      Significant_MAE_5pct,
      
      Significant_RMSE_FDR_5pct,
      Significant_MAE_FDR_5pct
    ),
  
  by = c(
    "Ticker",
    "Model"
  )
  
) %>%
  
  
  # ----------------------------------------------------------
# Additional useful variables
# ----------------------------------------------------------

mutate(
  
  DA_Percent =
    100 * DirectionalAccuracy,
  
  
  ExpectedDA_Percent =
    100 * ExpectedDA,
  
  
  # Excess directional accuracy over PT expectation
  PT_Excess_DA_pp =
    100 * (
      DirectionalAccuracy -
        ExpectedDA
    ),
  
  
  Beats_RMSE_Benchmark =
    RMSEGain > 0,
  
  
  Beats_MAE_Benchmark =
    MAEGain > 0
  
)


# Check
nrow(ml_selected_analysis)

# Must return:
# 33

# ============================================================
# TABLE 10:
# SELECTED ML PERFORMANCE BY STRUCTURAL VOLATILITY GROUP
# ============================================================


table10_volatility <- ml_selected_analysis %>%
  
  group_by(
    VolatilityGroup
  ) %>%
  
  summarise(
    
    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------
    
    N_Stocks =
      n_distinct(Ticker),
    
    
    # --------------------------------------------------------
    # RMSE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    SD_RMSEGain =
      sd(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Stocks_Beating_RMSE_Benchmark =
      sum(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    Percent_Beating_RMSE_Benchmark =
      100 *
      mean(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # MAE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    Median_MAEGain =
      median(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # OUT-OF-SAMPLE R2
    # --------------------------------------------------------
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # DIRECTIONAL PERFORMANCE
    # --------------------------------------------------------
    
    Mean_DA =
      mean(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Median_DA =
      median(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Mean_PT_Excess_DA =
      mean(
        PT_Excess_DA_pp,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # SIGNIFICANCE COUNTS
    # --------------------------------------------------------
    
    PT_Significant_5pct =
      sum(
        PT_Significant_5pct,
        na.rm = TRUE
      ),
    
    
    DM_RMSE_Significant_Better =
      sum(
        Significant_RMSE_5pct,
        na.rm = TRUE
      ),
    
    
    DM_MAE_Significant_Better =
      sum(
        Significant_MAE_5pct,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  ) %>%
  
  
  mutate(
    
    VolatilityGroup =
      factor(
        VolatilityGroup,
        levels = c(
          "Low",
          "Medium",
          "High"
        )
      )
    
  ) %>%
  
  
  arrange(
    VolatilityGroup
  )


View(table10_volatility)

# ============================================================
# TABLE 11:
# SELECTED ML PERFORMANCE BY LIQUIDITY GROUP
# ============================================================


table11_liquidity <- ml_selected_analysis %>%
  
  group_by(
    LiquidityGroup
  ) %>%
  
  summarise(
    
    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------
    
    N_Stocks =
      n_distinct(Ticker),
    
    
    # --------------------------------------------------------
    # RMSE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    SD_RMSEGain =
      sd(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Stocks_Beating_RMSE_Benchmark =
      sum(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    Percent_Beating_RMSE_Benchmark =
      100 *
      mean(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # MAE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    Median_MAEGain =
      median(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # OUT-OF-SAMPLE R2
    # --------------------------------------------------------
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # DIRECTIONAL PERFORMANCE
    # --------------------------------------------------------
    
    Mean_DA =
      mean(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Median_DA =
      median(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Mean_PT_Excess_DA =
      mean(
        PT_Excess_DA_pp,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # SIGNIFICANCE COUNTS
    # --------------------------------------------------------
    
    PT_Significant_5pct =
      sum(
        PT_Significant_5pct,
        na.rm = TRUE
      ),
    
    
    DM_RMSE_Significant_Better =
      sum(
        Significant_RMSE_5pct,
        na.rm = TRUE
      ),
    
    
    DM_MAE_Significant_Better =
      sum(
        Significant_MAE_5pct,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  ) %>%
  
  
  mutate(
    
    LiquidityGroup =
      factor(
        LiquidityGroup,
        levels = c(
          "High",
          "Medium",
          "Low"
        )
      )
    
  ) %>%
  
  
  arrange(
    LiquidityGroup
  )


View(table11_liquidity)

# ============================================================
# TABLE 12:
# SELECTED ML PERFORMANCE BY SECTOR
# ============================================================


table12_sector <- ml_selected_analysis %>%
  
  group_by(
    Sector
  ) %>%
  
  summarise(
    
    # --------------------------------------------------------
    # SAMPLE SIZE
    # --------------------------------------------------------
    
    N_Stocks =
      n_distinct(Ticker),
    
    
    # --------------------------------------------------------
    # RMSE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Min_RMSEGain =
      min(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Max_RMSEGain =
      max(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Stocks_Beating_RMSE_Benchmark =
      sum(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    Percent_Beating_RMSE_Benchmark =
      100 *
      mean(
        Beats_RMSE_Benchmark,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # MAE PERFORMANCE
    # --------------------------------------------------------
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    Median_MAEGain =
      median(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # OUT-OF-SAMPLE R2
    # --------------------------------------------------------
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    
    # --------------------------------------------------------
    # DIRECTIONAL PERFORMANCE
    # --------------------------------------------------------
    
    Mean_DA =
      mean(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Median_DA =
      median(
        DA_Percent,
        na.rm = TRUE
      ),
    
    
    Mean_PT_Excess_DA =
      mean(
        PT_Excess_DA_pp,
        na.rm = TRUE
      ),
    
    
    PT_Significant_5pct =
      sum(
        PT_Significant_5pct,
        na.rm = TRUE
      ),
    
    
    DM_RMSE_Significant_Better =
      sum(
        Significant_RMSE_5pct,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  ) %>%
  
  
  arrange(
    desc(
      Mean_RMSEGain
    )
  )


View(table12_sector)

# ============================================================
# TABLE 13
# TIME-VARYING VOLATILITY REGIMES
#
# Low / Medium / High state thresholds are estimated
# EXCLUSIVELY from the 2010-2017 training sample.
# ============================================================


# ------------------------------------------------------------
# 1. CHECK REQUIRED OBJECTS
# ------------------------------------------------------------

required_regime_objects <- c(
  "model_data",
  "selected_ml_predictions"
)

missing_regime_objects <- required_regime_objects[
  !sapply(required_regime_objects, exists)
]

if(length(missing_regime_objects) > 0){
  
  stop(
    paste(
      "Missing objects:",
      paste(
        missing_regime_objects,
        collapse = ", "
      )
    )
  )
}

# ============================================================
# 2. TRAINING-PERIOD VOLATILITY THRESHOLDS
# ============================================================

volatility_thresholds <- model_data %>%
  
  filter(
    Sample == "Training"
  ) %>%
  
  group_by(
    Ticker
  ) %>%
  
  summarise(
    
    Vol20_q33 =
      quantile(
        Vol20,
        probs = 1 / 3,
        na.rm = TRUE
      ),
    
    
    Vol20_q67 =
      quantile(
        Vol20,
        probs = 2 / 3,
        na.rm = TRUE
      ),
    
    
    Training_Median_Vol20 =
      median(
        Vol20,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  )


View(volatility_thresholds)

# ============================================================
# 3. CLASSIFY TEST OBSERVATIONS
# ============================================================

test_volatility_states <- model_data %>%
  
  filter(
    Sample == "Test"
  ) %>%
  
  select(
    Ticker,
    ForecastOrigin = Date,
    TargetDate,
    Vol20
  ) %>%
  
  left_join(
    volatility_thresholds,
    by = "Ticker"
  ) %>%
  
  mutate(
    
    VolatilityState = case_when(
      
      Vol20 < Vol20_q33 ~
        "Low",
      
      Vol20 < Vol20_q67 ~
        "Medium",
      
      TRUE ~
        "High"
      
    ),
    
    
    VolatilityState =
      factor(
        VolatilityState,
        levels = c(
          "Low",
          "Medium",
          "High"
        )
      )
    
  )

# ============================================================
# 4. ATTACH VOLATILITY STATES TO SELECTED ML FORECASTS
# ============================================================

ml_regime_predictions <- selected_ml_predictions %>%
  
  left_join(
    
    test_volatility_states,
    
    by = c(
      "Ticker",
      "ForecastOrigin",
      "TargetDate"
    )
    
  )

sum(
  is.na(
    ml_regime_predictions$VolatilityState
  )
)

table(
  ml_regime_predictions$VolatilityState
)

# ============================================================
# 5. FUNCTION FOR FORECAST METRICS WITHIN EACH REGIME
# ============================================================

regime_metrics <- function(actual, predicted){
  
  ok <- is.finite(actual) &
    is.finite(predicted)
  
  
  actual <- actual[ok]
  predicted <- predicted[ok]
  
  
  n <- length(actual)
  
  
  if(n == 0){
    
    return(
      tibble(
        Observations = 0,
        RMSE = NA_real_,
        MAE = NA_real_,
        BenchmarkRMSE = NA_real_,
        BenchmarkMAE = NA_real_,
        RelRMSE = NA_real_,
        RelMAE = NA_real_,
        RMSEGain = NA_real_,
        MAEGain = NA_real_,
        R2_OOS = NA_real_,
        DirectionalAccuracy = NA_real_
      )
    )
    
  }
  
  
  RMSE <- sqrt(
    mean(
      (actual - predicted)^2
    )
  )
  
  
  MAE <- mean(
    abs(
      actual - predicted
    )
  )
  
  
  BenchmarkRMSE <- sqrt(
    mean(
      actual^2
    )
  )
  
  
  BenchmarkMAE <- mean(
    abs(actual)
  )
  
  
  RelRMSE <-
    RMSE /
    BenchmarkRMSE
  
  
  RelMAE <-
    MAE /
    BenchmarkMAE
  
  
  RMSEGain <-
    100 *
    (
      1 - RelRMSE
    )
  
  
  MAEGain <-
    100 *
    (
      1 - RelMAE
    )
  
  
  R2_OOS <-
    1 -
    (
      sum(
        (actual - predicted)^2
      ) /
        sum(
          actual^2
        )
    )
  
  
  DirectionalAccuracy <-
    mean(
      actual * predicted > 0
    )
  
  
  tibble(
    
    Observations =
      n,
    
    RMSE =
      RMSE,
    
    MAE =
      MAE,
    
    BenchmarkRMSE =
      BenchmarkRMSE,
    
    BenchmarkMAE =
      BenchmarkMAE,
    
    RelRMSE =
      RelRMSE,
    
    RelMAE =
      RelMAE,
    
    RMSEGain =
      RMSEGain,
    
    MAEGain =
      MAEGain,
    
    R2_OOS =
      R2_OOS,
    
    DirectionalAccuracy =
      DirectionalAccuracy
    
  )
}

# ============================================================
# TABLE 13A:
# STOCK-SPECIFIC PERFORMANCE BY VOLATILITY STATE
# ============================================================

table13_stock_regimes <- ml_regime_predictions %>%
  
  filter(
    !is.na(
      VolatilityState
    )
  ) %>%
  
  group_by(
    Ticker,
    sector,
    Model,
    VolatilityState
  ) %>%
  
  group_modify(
    
    ~ regime_metrics(
      
      actual =
        .x$ActualReturn,
      
      predicted =
        .x$Prediction
      
    )
    
  ) %>%
  
  ungroup() %>%
  
  arrange(
    Ticker,
    VolatilityState
  )


View(table13_stock_regimes)

# ============================================================
# TABLE 13B:
# AGGREGATE ML PERFORMANCE BY TIME-VARYING VOLATILITY STATE
# ============================================================

table13_regime_summary <- table13_stock_regimes %>%
  
  group_by(
    VolatilityState
  ) %>%
  
  summarise(
    
    N_Stocks =
      n_distinct(
        Ticker
      ),
    
    
    Mean_RMSEGain =
      mean(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Median_RMSEGain =
      median(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    SD_RMSEGain =
      sd(
        RMSEGain,
        na.rm = TRUE
      ),
    
    
    Stocks_Beating_Benchmark =
      sum(
        RMSEGain > 0,
        na.rm = TRUE
      ),
    
    
    Percent_Beating_Benchmark =
      100 *
      mean(
        RMSEGain > 0,
        na.rm = TRUE
      ),
    
    
    Mean_MAEGain =
      mean(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    Median_MAEGain =
      median(
        MAEGain,
        na.rm = TRUE
      ),
    
    
    Mean_R2_OOS =
      mean(
        R2_OOS,
        na.rm = TRUE
      ),
    
    
    Mean_DA =
      100 *
      mean(
        DirectionalAccuracy,
        na.rm = TRUE
      ),
    
    
    Median_DA =
      100 *
      median(
        DirectionalAccuracy,
        na.rm = TRUE
      ),
    
    
    .groups = "drop"
    
  )


View(table13_regime_summary)

# ============================================================
# TABLE 14:
# CONTINUOUS CROSS-SECTIONAL EMPIRICAL ANALYSIS
# ============================================================


# ------------------------------------------------------------
# Packages for robust standard errors
# ------------------------------------------------------------

packages_empirical <- c(
  "lmtest",
  "sandwich"
)

new_empirical_packages <- packages_empirical[
  !(packages_empirical %in% installed.packages()[, "Package"])
]

if(length(new_empirical_packages) > 0){
  install.packages(
    new_empirical_packages
  )
}

library(lmtest)
library(sandwich)

# ============================================================
# 6. CROSS-SECTIONAL REGRESSION DATASET
# ============================================================

empirical_ml <- ml_selected_analysis %>%
  
  select(
    Sector,
    Ticker,
    Model,
    TrainingVolatility,
    TrainingILLIQ,
    TrainingDollarVolume,
    RMSEGain,
    MAEGain,
    R2_OOS,
    DirectionalAccuracy
  ) %>%
  
  filter(
    is.finite(TrainingVolatility),
    is.finite(TrainingILLIQ),
    TrainingILLIQ > 0,
    is.finite(RMSEGain)
  ) %>%
  
  mutate(
    
    LogILLIQ =
      log(
        TrainingILLIQ
      ),
    
    
    # Standardize the continuous characteristics.
    #
    # This lets coefficients be interpreted as the change
    # in RMSEGain associated with a 1-SD increase in the
    # characteristic.
    
    Volatility_z =
      as.numeric(
        scale(
          TrainingVolatility
        )
      ),
    
    
    LogILLIQ_z =
      as.numeric(
        scale(
          LogILLIQ
        )
      )
    
  )

nrow(empirical_ml)

# ============================================================
# TABLE 14A:
# SPEARMAN CORRELATIONS
# ============================================================


spearman_vol <- cor.test(
  
  empirical_ml$TrainingVolatility,
  empirical_ml$RMSEGain,
  
  method = "spearman",
  
  exact = FALSE
  
)


spearman_illiq <- cor.test(
  
  empirical_ml$LogILLIQ,
  empirical_ml$RMSEGain,
  
  method = "spearman",
  
  exact = FALSE
  
)

table14_spearman <- tibble(
  
  Characteristic = c(
    "Training volatility",
    "Log Amihud illiquidity"
  ),
  
  
  Spearman_rho = c(
    
    as.numeric(
      spearman_vol$estimate
    ),
    
    as.numeric(
      spearman_illiq$estimate
    )
    
  ),
  
  
  p_value = c(
    
    spearman_vol$p.value,
    
    spearman_illiq$p.value
    
  )
  
) %>%
  
  mutate(
    
    Significant_10pct =
      p_value < 0.10,
    
    Significant_5pct =
      p_value < 0.05
    
  )


View(table14_spearman)

ml_reg_vol <- lm(
  
  RMSEGain ~
    Volatility_z,
  
  data =
    empirical_ml
  
)

ml_reg_illiq <- lm(
  
  RMSEGain ~
    LogILLIQ_z,
  
  data =
    empirical_ml
  
)

ml_reg_both <- lm(
  
  RMSEGain ~
    Volatility_z +
    LogILLIQ_z,
  
  data =
    empirical_ml
  
)

# ============================================================
# 7. HC3 ROBUST RESULTS
# ============================================================

robust_vol <- lmtest::coeftest(
  
  ml_reg_vol,
  
  vcov. =
    sandwich::vcovHC(
      ml_reg_vol,
      type = "HC3"
    )
  
)


robust_illiq <- lmtest::coeftest(
  
  ml_reg_illiq,
  
  vcov. =
    sandwich::vcovHC(
      ml_reg_illiq,
      type = "HC3"
    )
  
)


robust_both <- lmtest::coeftest(
  
  ml_reg_both,
  
  vcov. =
    sandwich::vcovHC(
      ml_reg_both,
      type = "HC3"
    )
  
)

robust_vol

robust_illiq

robust_both

# ============================================================
# TABLE 14B:
# ROBUST CROSS-SECTIONAL REGRESSIONS
# ============================================================

extract_robust_results <- function(
    model,
    robust_test,
    model_name
){
  
  coefficient_matrix <-
    as.matrix(
      robust_test
    )
  
  
  tibble(
    
    Model =
      model_name,
    
    
    Term =
      rownames(
        coefficient_matrix
      ),
    
    
    Estimate =
      coefficient_matrix[, 1],
    
    
    Robust_SE =
      coefficient_matrix[, 2],
    
    
    t_value =
      coefficient_matrix[, 3],
    
    
    p_value =
      coefficient_matrix[, 4],
    
    
    N =
      nobs(
        model
      ),
    
    
    R2 =
      summary(
        model
      )$r.squared,
    
    
    Adjusted_R2 =
      summary(
        model
      )$adj.r.squared
    
  )
}

table14_regression <- bind_rows(
  
  extract_robust_results(
    ml_reg_vol,
    robust_vol,
    "Volatility only"
  ),
  
  
  extract_robust_results(
    ml_reg_illiq,
    robust_illiq,
    "Illiquidity only"
  ),
  
  
  extract_robust_results(
    ml_reg_both,
    robust_both,
    "Volatility + Illiquidity"
  )
  
) %>%
  
  mutate(
    
    Significant_10pct =
      p_value < 0.10,
    
    Significant_5pct =
      p_value < 0.05
    
  )


View(table14_regression)

# ============================================================
# FINAL ML ROBUSTNESS:
# INFLUENCE DIAGNOSTICS FOR JOINT REGRESSION
# ============================================================

influence_table <- empirical_ml %>%
  mutate(
    CooksDistance = cooks.distance(ml_reg_both),
    Leverage = hatvalues(ml_reg_both),
    StudentizedResidual = rstudent(ml_reg_both)
  ) %>%
  select(
    Ticker,
    Sector,
    RMSEGain,
    TrainingVolatility,
    LogILLIQ,
    CooksDistance,
    Leverage,
    StudentizedResidual
  ) %>%
  arrange(desc(CooksDistance))

View(influence_table)
