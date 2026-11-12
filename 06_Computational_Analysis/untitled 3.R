
# ============================================================
# MASTER'S THESIS
# TRADITIONAL ML COMPUTATIONAL RESOURCE BENCHMARK
#
# 33 stocks - 6 ML algorithms
# Peak process RAM (RSS) + execution time
# ============================================================

# 1. INSTALL PACKAGES (only if needed)

packages <- c("dplyr", "ps", "callr")

missing <- packages[
  !vapply(packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing)) {
  install.packages(missing)
}

library(dplyr)

# 2. VERIFY EXISTING ML OBJECTS

stopifnot(
  exists("model_data"),
  exists("run_ml_stock"),
  is.function(run_ml_stock)
)

tickers <- unique(as.character(model_data$Ticker))

stopifnot(length(tickers) == 33)

# IMPORTANT:
# Keep the same stock order as your original ML run.
# The stock_number argument may affect random seeds.


# 3. CREATE A NEW OUTPUT DIRECTORY

output_dir <- file.path(
  getwd(),
  paste0(
    "ML_RAM_BENCHMARK_",
    format(Sys.time(), "%Y%m%d_%H%M%S")
  )
)

dir.create(output_dir, recursive = TRUE)

sample_file <- file.path(output_dir, "rss_samples.csv")
stop_file <- file.path(output_dir, "stop_monitor.flag")
ready_file <- file.path(output_dir, "monitor_ready.flag")

# R's own process ID
main_pid <- Sys.getpid()

cat("Monitoring R process:", main_pid, "\n")


# 4. START INDEPENDENT MEMORY MONITOR

# Runs in a separate R process.
# Samples the main R process every 50 ms.

monitor <- callr::r_bg(
  func = function(pid, stop_file, ready_file,
                  sample_file) {
    
    handle <- ps::ps_handle(pid)
    
    times <- numeric()
    rss <- numeric()
    
    file.create(ready_file)
    
    while (!file.exists(stop_file)) {
      
      current <- tryCatch(
        ps::ps_memory_info(handle)[["rss"]],
        error = function(e) NA_real_
      )
      
      times <- c(times, as.numeric(Sys.time()))
      rss <- c(rss, current)
      
      Sys.sleep(0.05)
    }
    
    write.csv(
      data.frame(
        Timestamp = times,
        RSS_Bytes = rss
      ),
      sample_file,
      row.names = FALSE
    )
  },
  args = list(
    pid = main_pid,
    stop_file = stop_file,
    ready_file = ready_file,
    sample_file = sample_file
  ),
  stdout = file.path(output_dir, "monitor_stdout.txt"),
  stderr = file.path(output_dir, "monitor_stderr.txt")
)

# Wait until monitoring is active
for (attempt in 1:100) {
  if (file.exists(ready_file)) break
  if (!monitor$is_alive()) {
    stop("Memory monitor terminated unexpectedly.")
  }
  Sys.sleep(0.1)
}

if (!file.exists(ready_file)) {
  stop("Memory monitor failed to start.")
}


# 5. RUN ORIGINAL ML PIPELINE

cost_results <- data.frame(
  Ticker = character(),
  Start_Time = numeric(),
  End_Time = numeric(),
  Time_Seconds = numeric(),
  Starting_RAM_MiB = numeric()
)

for (i in seq_along(tickers)) {
  
  ticker <- tickers[i]
  
  stock_df <- model_data %>%
    filter(Ticker == ticker)
  
  cat("\nBenchmarking:", ticker, "\n")
  
  start_ram <- ps::ps_memory_info(
    ps::ps_handle(main_pid)
  )[["rss"]] / 1024^2
  
  t0 <- as.numeric(Sys.time())
  clock0 <- proc.time()[["elapsed"]]
  
  # ORIGINAL ML FUNCTION: UNCHANGED
  result <- run_ml_stock(
    stock_df = stock_df,
    stock_number = i
  )
  
  elapsed <- proc.time()[["elapsed"]] - clock0
  t1 <- as.numeric(Sys.time())
  
  cost_results <- rbind(
    cost_results,
    data.frame(
      Ticker = ticker,
      Start_Time = t0,
      End_Time = t1,
      Time_Seconds = elapsed,
      Starting_RAM_MiB = start_ram
    )
  )
  
  cat("Completed:", ticker, "\n")
  cat("Elapsed:", round(elapsed, 2), "seconds\n")
  
  # Save progress after each stock
  write.csv(
    cost_results,
    file.path(output_dir, "ML_progress.csv"),
    row.names = FALSE
  )
}


# 6. STOP MEMORY MONITOR

file.create(stop_file)

monitor$wait(timeout = 30000)

if (!file.exists(sample_file)) {
  stop("RAM monitoring data not found.")
}

samples <- read.csv(sample_file)


# 7. CALCULATE PEAK RAM FOR EACH STOCK

cost_results$Peak_RAM_MiB <- vapply(
  seq_len(nrow(cost_results)),
  function(i) {
    
    x <- samples$RSS_Bytes[
      samples$Timestamp >= cost_results$Start_Time[i] &
        samples$Timestamp <= cost_results$End_Time[i]
    ]
    
    x <- x[is.finite(x)]
    
    if (!length(x)) return(NA_real_)
    
    max(x) / 1024^2
  },
  numeric(1)
)

# Peak above process RAM at stock start.
# This is NOT identical to net memory allocated
# by the ML models.
cost_results$Peak_Increase_MiB <- pmax(
  0,
  cost_results$Peak_RAM_MiB -
    cost_results$Starting_RAM_MiB
)

# Keep only the columns relevant to your thesis
cost_results <- cost_results %>%
  select(
    Ticker,
    Time_Seconds,
    Starting_RAM_MiB,
    Peak_RAM_MiB,
    Peak_Increase_MiB
  )


# 8. COMPUTATIONAL SUMMARY

cost_summary <- cost_results %>%
  summarise(
    Number_of_Stocks = n(),
    
    Total_Seconds = sum(Time_Seconds),
    
    Mean_Seconds_Per_Stock = mean(Time_Seconds),
    
    Median_Seconds_Per_Stock = median(Time_Seconds),
    
    Mean_Peak_RAM_MiB = mean(
      Peak_RAM_MiB, na.rm = TRUE
    ),
    
    Median_Peak_RAM_MiB = median(
      Peak_RAM_MiB, na.rm = TRUE
    ),
    
    Maximum_Peak_RAM_MiB = max(
      Peak_RAM_MiB, na.rm = TRUE
    ),
    
    Mean_Peak_Increase_MiB = mean(
      Peak_Increase_MiB, na.rm = TRUE
    )
  )

print(cost_results)
print(cost_summary)


# 9. SAVE RESULTS

write.csv(
  cost_results,
  file.path(output_dir, "ML_computational_costs.csv"),
  row.names = FALSE
)

write.csv(
  cost_summary,
  file.path(output_dir, "ML_cost_summary.csv"),
  row.names = FALSE
)

cat("\n======================================\n")
cat("ML RESOURCE BENCHMARK COMPLETE\n")
cat("======================================\n")
cat("Results saved to:", output_dir, "\n")

View(cost_results)
View(cost_summary)
