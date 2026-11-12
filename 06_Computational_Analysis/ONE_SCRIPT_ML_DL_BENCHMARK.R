# =====================================================================
# MASTER'S THESIS — ML vs DL RESOURCE BENCHMARK, ONE RSTUDIO SCRIPT
# =====================================================================
# Required in your RStudio Environment:
#   1. model_data (original 33-stock data frame)
#   2. run_ml_stock(stock_df, stock_number) plus its original dependencies
#   3. Python file train_dl_33_with_RAM.py on disk (selected when prompted)
#
# Measurements:
#  * time (seconds) for all six ML or DL candidates per stock
#  * starting, peak and incremental resident memory (MiB) per stock
#  * tables (CSV), plots (PNG/PDF), a human-readable summary (TXT)
#
# ML RSS sampled externally using a second R process ('ps' + 'callr');
# DL RSS sampled inside the Python process by its existing psutil monitor.
# Both refer to process RSS, but R and Python baselines are NOT equivalent.
# Always run from a fresh R session for a cleaner ML memory baseline.
# =====================================================================

required_packages <- c("dplyr", "ggplot2", "callr", "ps")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly=TRUE)
]
if (length(missing_packages)) install.packages(missing_packages)

stopifnot(exists("model_data"), is.data.frame(model_data),
          exists("run_ml_stock"), is.function(run_ml_stock))

# Original stock order matters because stock_number may affect random seeds.
tickers <- unique(as.character(model_data$Ticker))
stopifnot(length(tickers) == 33L)

# All DL predictors, as used in the original Python training pipeline.
predictor_cols <- c(
  paste0("ReturnLag",1:5),
  paste0("SMADist",c(5,10,20)),
  paste0("EMADist",c(5,10,20)),
  paste0("Momentum",c(5,10,20)),
  "RSI14","MACDRel","MACDSignalRel","BBPosition","StochK","StochD",
  "Vol5","Vol20","LogVolume","RelVolume20","DeltaOBV",
  "OvernightReturn","DollarVolume","ILLIQ20"
)
meta_cols <- c("Ticker","sector","Date","TargetDate","Sample","Target")
required_cols <- c(meta_cols,predictor_cols)
missing_cols <- setdiff(required_cols,names(model_data))
if (length(missing_cols)) stop("Missing model_data columns: ",paste(missing_cols,collapse=", "))
stopifnot(length(predictor_cols)==28L)

# IMPORTANT FAIRNESS WARNING:
# Verify the original run_ml_stock predictor set matches the DL 28 predictors.
# This benchmark does NOT modify the original forecasting specification.
if (exists("predictor_names")) {
  if (!setequal(as.character(predictor_names), predictor_cols)) {
    warning("ML predictor_names differ from the 28 DL predictors; disclose/resolve before inferring model-family effects.")
  }
}

# Select previously provided Python script from any location.
message("Select train_dl_33_with_RAM.py when the file browser appears.")
python_script <- normalizePath(file.choose(), winslash="/", mustWork=TRUE)
if (basename(python_script)!="train_dl_33_with_RAM.py") {
  stop("Select the supplied train_dl_33_with_RAM.py, not train_dl_33.py")
}
python_exe <- file.path(Sys.getenv("LOCALAPPDATA"),
                        "thesis-dl-torch", "Scripts", "python.exe")
if (!file.exists(python_exe)) stop("PyTorch Python not found: ",python_exe)

# Check dependencies, do not silently install packages during benchmarking.
python_check <- system2(python_exe,
  args=c("-c", shQuote("import torch, psutil, numpy, pandas")),
  stdout=FALSE, stderr=FALSE)
if (!identical(as.integer(python_check),0L)) {
  stop("Python dependencies missing. Run pip install for torch, psutil, numpy and pandas in thesis-dl-torch.")
}

# Create a NEW folder; never overwrite either thesis results or prior runs.
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
out <- file.path(getwd(), paste0("FINAL_RESOURCE_BENCHMARK_",run_id))
if (dir.exists(out)) stop("Benchmark folder already exists")
dir.create(out,recursive=TRUE)
dl_out <- file.path(out,"DL_new_training")
dir.create(dl_out)
plot_dir <- file.path(out,"figures")
dir.create(plot_dir)
message("All outputs will be saved under: ", normalizePath(out))

# Prepare the Python input EXACTLY as in original DL launcher.
data <- as.data.frame(model_data[,required_cols,drop=FALSE])
data$Sample <- as.character(data$Sample)
data$Date <- as.character(as.Date(data$Date))
data$TargetDate <- as.character(as.Date(data$TargetDate))
stopifnot(!anyNA(data),
          setequal(unique(data$Sample),c("Training","Validation","Test")),
          !anyDuplicated(data[c("Ticker","Date")]),
          all(is.finite(as.matrix(data[,c("Target",predictor_cols)]))))
input_file <- file.path(out,"dl_input_33.csv")
write.csv(data,input_file,row.names=FALSE)

# ---------------------------------------------------------------------
# 1. ML: INDEPENDENT PROCESS-RSS SAMPLER
# ---------------------------------------------------------------------
# Background monitor writes a CSV of timestamped samples when stopped.
# This observes RStudio's R *process*, including pre-existing workspace RAM.
start_rss_monitor <- function(pid, output_folder) {
  stop_flag <- file.path(output_folder,"stop_monitor.flag")
  ready_flag <- file.path(output_folder,"monitor_ready.flag")
  sample_file <- file.path(output_folder,"ML_RSS_samples.csv")
  if (file.exists(stop_flag)) file.remove(stop_flag)
  if (file.exists(ready_flag)) file.remove(ready_flag)
  monitor <- callr::r_bg(
    func=function(pid, stop_flag, ready_flag, sample_file) {
      handle <- ps::ps_handle(pid)
      t <- numeric(0)
      rss <- numeric(0)
      file.create(ready_flag)
      repeat {
        now <- as.numeric(Sys.time())
        val <- tryCatch(as.numeric(ps::ps_memory_info(handle)[["rss"]]),
                        error=function(e) NA_real_)
        t <- c(t,now)
        rss <- c(rss,val)
        if (file.exists(stop_flag)) break
        Sys.sleep(0.02) # same nominal sampling interval as DL
      }
      write.csv(data.frame(Timestamp=t,RSS_Bytes=rss),sample_file,row.names=FALSE)
    },
    args=list(pid=pid,stop_flag=stop_flag,ready_flag=ready_flag,
              sample_file=sample_file),
    stdout=file.path(output_folder,"ML_monitor_stdout.txt"),
    stderr=file.path(output_folder,"ML_monitor_stderr.txt")
  )
  for (j in 1:100) {
    if (file.exists(ready_flag)) break
    if (!monitor$is_alive()) stop("ML memory monitor failed to launch")
    Sys.sleep(0.1)
  }
  if (!file.exists(ready_flag)) stop("ML memory monitor did not become ready")
  list(proc=monitor, stop_flag=stop_flag, sample_file=sample_file)
}

message("\n=== BENCHMARK 1/2: TRADITIONAL MACHINE LEARNING ===")
pid <- Sys.getpid()
watcher <- start_rss_monitor(pid,out)
ml_rows <- list()
ml_success <- FALSE

# tryCatch ensures the sampler is stopped on failure as well.
tryCatch({
  for (i in seq_along(tickers)) {
    ticker <- tickers[i]
    stock_df <- model_data[as.character(model_data$Ticker)==ticker,,drop=FALSE]
    cat(sprintf("\n[ML %d/33] %s\n",i,ticker))
    start_ram <- as.numeric(ps::ps_memory_info(ps::ps_handle(pid))[["rss"]]) / 1024^2
    t0 <- as.numeric(Sys.time())
    clk <- proc.time()[["elapsed"]]
    # Fits and tunes all six ML models: unchanged original function.
    result <- run_ml_stock(stock_df=stock_df,stock_number=i)
    elapsed <- proc.time()[["elapsed"]]-clk
    t1 <- as.numeric(Sys.time())
    ml_rows[[i]] <- data.frame(Ticker=ticker,Start_Time=t0,End_Time=t1,
                                Time_Seconds=elapsed,Starting_RAM_MiB=start_ram)
    write.csv(dplyr::bind_rows(ml_rows),
              file.path(out,"ML_progress.csv"),row.names=FALSE)
    cat(sprintf("  ML completed in %.2f seconds\n",elapsed))
    rm(result)
  }
  ml_success <- TRUE
}, error=function(e) {
  message("ML fitting failed: ", conditionMessage(e))
})
file.create(watcher$stop_flag)
watcher$proc$wait(timeout=60000)
if (!ml_success) stop("ML run failed. Partial ML_progress.csv is preserved in: ",out)
if (!file.exists(watcher$sample_file)) stop("ML RSS sample data not saved")

samples <- read.csv(watcher$sample_file)
ml <- dplyr::bind_rows(ml_rows)
ml$Peak_RAM_MiB <- vapply(seq_len(nrow(ml)),function(j) {
  selected <- samples$RSS_Bytes[samples$Timestamp >= ml$Start_Time[j] &
                                 samples$Timestamp <= ml$End_Time[j]]
  selected <- selected[is.finite(selected)]
  if (!length(selected)) return(NA_real_)
  max(selected)/1024^2
}, numeric(1))
if (anyNA(ml$Peak_RAM_MiB)) {
  stop("Some ML stock windows had no memory samples: inspect ML_RSS_samples.csv")
}
ml$Peak_Increase_MiB <- pmax(0, ml$Peak_RAM_MiB-ml$Starting_RAM_MiB)
ml$Family <- "Traditional ML"
ml <- ml[,c("Ticker","Family","Time_Seconds","Starting_RAM_MiB",
             "Peak_RAM_MiB","Peak_Increase_MiB")]
write.csv(ml,file.path(out,"ML_computational_costs.csv"),row.names=FALSE)

# ---------------------------------------------------------------------
# 2. DL: RUN ORIGINAL PYTORCH WITH PER-STOCK RAM MONITOR
# ---------------------------------------------------------------------
message("\n=== BENCHMARK 2/2: DEEP LEARNING / PYTORCH ===")
# Unique dl_out directory: nothing has been trained there; nothing is skipped.
dl_log <- file.path(out,"DL_training_log.txt")
python_args <- c(shQuote(python_script),
                 "--input",shQuote(normalizePath(input_file,winslash="/")),
                 "--output",shQuote(normalizePath(dl_out,winslash="/")),
                 "--epochs","80","--patience","10")
dl_clock <- proc.time()[["elapsed"]]
# stdout/stderr sent to log so failures can be diagnosed.
dl_status <- system2(python_exe,args=python_args,
                     stdout=dl_log,stderr=dl_log,wait=TRUE)
dl_wall_total <- proc.time()[["elapsed"]]-dl_clock
if (!identical(as.integer(dl_status),0L)) {
  stop("DL training failed; inspect: ",dl_log)
}

cost_file <- file.path(dl_out,"dl_computational_costs.csv")
if (!file.exists(cost_file)) stop("DL computational CSV missing: ",cost_file)
dl_raw <- read.csv(cost_file,check.names=FALSE)
expected <- c("Ticker","Train_Tune_And_Output_Seconds",
              "Starting_Process_RSS_MiB","Peak_Process_RSS_MiB",
              "Peak_Above_Start_MiB")
if (!all(expected %in% names(dl_raw))) {
  stop("Missing DL cost columns: ", paste(setdiff(expected,names(dl_raw)),collapse=", "))
}
if (nrow(dl_raw)!=33 || length(unique(dl_raw$Ticker))!=33) {
  stop("DL did not produce exactly 33 distinct per-stock costs")
}
dl <- data.frame(Ticker=as.character(dl_raw$Ticker), Family="Deep Learning",
                 Time_Seconds=dl_raw$Train_Tune_And_Output_Seconds,
                 Starting_RAM_MiB=dl_raw$Starting_Process_RSS_MiB,
                 Peak_RAM_MiB=dl_raw$Peak_Process_RSS_MiB,
                 Peak_Increase_MiB=dl_raw$Peak_Above_Start_MiB)
write.csv(dl,file.path(out,"DL_computational_costs.csv"),row.names=FALSE)

# ---------------------------------------------------------------------
# 3. ALL COMPARISONS AND TABLES
# ---------------------------------------------------------------------
per_stock <- dplyr::bind_rows(ml,dl)
summary_table <- per_stock |>
  dplyr::group_by(Family) |>
  dplyr::summarise(
    Stocks=dplyr::n(),
    Total_Stock_Time_s=sum(Time_Seconds),
    Mean_Time_s=mean(Time_Seconds),
    Median_Time_s=median(Time_Seconds),
    Mean_Starting_RAM_MiB=mean(Starting_RAM_MiB),
    Mean_Peak_RAM_MiB=mean(Peak_RAM_MiB),
    Median_Peak_RAM_MiB=median(Peak_RAM_MiB),
    Max_Peak_RAM_MiB=max(Peak_RAM_MiB),
    Mean_Incremental_RAM_MiB=mean(Peak_Increase_MiB),
    Median_Incremental_RAM_MiB=median(Peak_Increase_MiB),
    Max_Incremental_RAM_MiB=max(Peak_Increase_MiB),
    .groups="drop"
  )
write.csv(per_stock,file.path(out,"ML_vs_DL_per_stock.csv"),row.names=FALSE)
write.csv(summary_table,file.path(out,"ML_vs_DL_summary.csv"),row.names=FALSE)

paired <- merge(ml,dl,by="Ticker",suffixes=c("_ML","_DL"))
paired$Peak_RAM_Diff_ML_minus_DL <- paired$Peak_RAM_MiB_ML-paired$Peak_RAM_MiB_DL
paired$Incremental_RAM_Diff_ML_minus_DL <- paired$Peak_Increase_MiB_ML-paired$Peak_Increase_MiB_DL
paired$Time_Ratio_ML_div_DL <- paired$Time_Seconds_ML/paired$Time_Seconds_DL
write.csv(paired,file.path(out,"ML_vs_DL_paired_33_stocks.csv"),row.names=FALSE)

cat("\n========== RESOURCE COMPARISON ==========\n")
print(summary_table,width=Inf)
cat("\nDL complete-run wall time (includes Python startup):",
    round(dl_wall_total,2),"seconds\n")
cat("\nIMPORTANT: total DL per-stock durations exclude some process startup/aggregation.\n")

# ---------------------------------------------------------------------
# 4. THREE THESIS-READY FIGURES (PNG + vector PDF)
# ---------------------------------------------------------------------
# No inference of intrinsic model efficiency from the plots.
plot_theme <- ggplot2::theme_minimal(base_size=12) +
  ggplot2::theme(legend.position="bottom",panel.grid.minor=ggplot2::element_blank())

save_chart <- function(fig,stem,width=9,height=5) {
  ggplot2::ggsave(file.path(plot_dir,paste0(stem,".png")),fig,
                  width=width,height=height,dpi=300,bg="white")
  ggplot2::ggsave(file.path(plot_dir,paste0(stem,".pdf")),fig,
                  width=width,height=height,bg="white")
}

p1 <- ggplot2::ggplot(per_stock,
                     ggplot2::aes(x=Family,y=Peak_RAM_MiB,fill=Family)) +
  ggplot2::geom_boxplot(width=0.5,alpha=0.75,outlier.shape=NA) +
  ggplot2::geom_jitter(width=0.08,height=0,alpha=0.65,size=1.6) +
  ggplot2::labs(title="Absolute resident memory by model family",
                subtitle="33 stocks; includes R/Python runtime and retained workspace memory",
                x=NULL,y="Observed peak process RSS (MiB)") +
  plot_theme + ggplot2::guides(fill="none")
save_chart(p1,"Figure_1_absolute_peak_RSS")

p2 <- ggplot2::ggplot(per_stock,
                     ggplot2::aes(x=Family,y=Peak_Increase_MiB,fill=Family)) +
  ggplot2::geom_boxplot(width=0.5,alpha=0.75,outlier.shape=NA) +
  ggplot2::geom_jitter(width=0.08,height=0,alpha=0.65,size=1.6) +
  ggplot2::labs(title="Incremental resident memory by model family",
                subtitle="Peak RSS minus start-of-stock RSS; not isolated model memory",
                x=NULL,y="Incremental peak process RSS (MiB)") +
  plot_theme + ggplot2::guides(fill="none")
save_chart(p2,"Figure_2_incremental_peak_RSS")

p3 <- ggplot2::ggplot(per_stock,
                     ggplot2::aes(x=stats::reorder(Ticker,Time_Seconds),
                                  y=Time_Seconds,fill=Family)) +
  ggplot2::geom_col(position="dodge") +
  ggplot2::coord_flip() +
  ggplot2::labs(title="End-to-end computational time by stock",
                subtitle="Six candidates per family; language/implementation differences remain",
                x="Stock",y="Wall-clock seconds") +
  plot_theme
save_chart(p3,"Figure_3_execution_time",width=10,height=9)

# ---------------------------------------------------------------------
# 5. HUMAN-READABLE REPORT AND R OBJECTS
# ---------------------------------------------------------------------
report <- c(
  "MASTER'S THESIS — COMPUTATIONAL RESOURCE BENCHMARK",
  paste("Benchmark run:",run_id),
  paste("Stocks:",length(tickers)),
  "ML: run_ml_stock() on six candidate methods; R process RSS (20-ms sampling)",
  "DL: six original PyTorch configurations; Python process RSS (20-ms sampling)",
  paste("ML R process PID:",pid),
  paste("Python executable:",python_exe),
  paste("DL script:",python_script),
  paste("PyTorch configured with:","--epochs 80 --patience 10"),
  paste("DL overall wall-clock seconds:",round(dl_wall_total,3)),
  "",
  "CAUTIONS:",
  "1. ML RStudio memory may include unrelated objects loaded in the Environment.",
  "2. DL Python process baseline includes imported libraries and earlier stocks.",
  "3. Incremental peak RSS is not exclusive memory allocated by each model.",
  "4. ML/DL runtimes may differ due to R/Python, threading and implementation.",
  "5. Compare predictors and training protocols before attributing differences to model family.",
  "6. Memory peaks are sampled estimates and may miss short-lived spikes."
)
writeLines(report,file.path(out,"READ_ME_Methodology_and_Cautions.txt"))
saveRDS(list(ml=ml,dl=dl,summary=summary_table,paired=paired),
        file.path(out,"computational_comparison.rds"))
cat("\nDONE. All tables, plots, logs and data saved in:\n",out,"\n")
if (interactive()) {
  View(summary_table)
  View(per_stock)
  print(p1)
  print(p2)
  print(p3)
}
