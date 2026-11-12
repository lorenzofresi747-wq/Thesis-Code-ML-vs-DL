# MASTER'S THESIS - Make multi-seed stock-level DL performance the primary robustness summary
# Run in RStudio after the existing 06_analyse_seed_stability.R has completed.
# Input: FINAL_DL_vs_ML/seed_stability/seed_05_all_seed_scores.csv
# No training or re-estimation is performed.

cat('Select seed_05_all_seed_scores.csv from FINAL_DL_vs_ML/seed_stability\n')
infile <- file.choose()
d <- read.csv(infile, stringsAsFactors = FALSE)
required <- c('Ticker', 'SeedLabel', 'RMSE_ML', 'RMSE_DL', 'DLGain_RMSE')
if (!all(required %in% names(d))) stop('Input lacks: ', paste(setdiff(required, names(d)), collapse=', '))
if (nrow(d) != 132L || length(unique(d$Ticker)) != 33L) stop('Expected 4 runs for each of 33 stocks (132 rows).')
if (anyDuplicated(d[, c('Ticker','SeedLabel')])) stop('Repeated stock/seed combination.')
if (!setequal(unique(d$SeedLabel), c('Original','Seed_101','Seed_202','Seed_303'))) stop('Unexpected seed labels.')
if (anyNA(d[, required]) || any(!is.finite(d$RMSE_ML)) || any(!is.finite(d$RMSE_DL))) stop('Missing or nonfinite values.')
if (any(d$RMSE_ML <= 0) || any(d$RMSE_DL <= 0)) stop('RMSE must be positive.')
if (any(abs(d$DLGain_RMSE - 100*(1-d$RMSE_DL/d$RMSE_ML)) > 1e-5)) stop('DLGain/RMSE inconsistency.')

# A stable 4-run summary per stock, not an average of per-run medians.
sp <- split(d, d$Ticker)
stock <- do.call(rbind, lapply(sp, function(s) {
  if (nrow(s)!=4 || length(unique(s$SeedLabel))!=4 || diff(range(s$RMSE_ML))>1e-8) stop('Invalid per-stock runs for ',s$Ticker[1])
  data.frame(Ticker=s$Ticker[1], RMSE_ML=s$RMSE_ML[1],
             Mean_RMSE_DL=mean(s$RMSE_DL), Median_RMSE_DL=median(s$RMSE_DL),
             Mean_DLGain=mean(s$DLGain_RMSE), Median_DLGain=median(s$DLGain_RMSE),
             DL_Win_Fraction=mean(s$DLGain_RMSE>0),
             DL_Wins_All_4=all(s$DLGain_RMSE>0),
             ML_Wins_All_4=all(s$DLGain_RMSE<0),
             Original_DLGain=s$DLGain_RMSE[s$SeedLabel=='Original'])
}))
row.names(stock) <- NULL
stock <- stock[order(stock$Ticker), ]

# Two alternative stock-level summaries: mean and median over seed-specific RMSEs.
# Headline: each stock contributes once, so n=33, not n=132.
summary <- data.frame(
  Method=c('Mean DL RMSE across 4 seeds','Median DL RMSE across 4 seeds','Original initialization only'),
  DL_Better_Stocks=c(sum(stock$Mean_DLGain>0),sum(stock$Median_DLGain>0),sum(stock$Original_DLGain>0)),
  ML_Better_Stocks=c(sum(stock$Mean_DLGain<0),sum(stock$Median_DLGain<0),sum(stock$Original_DLGain<0)),
  Mean_DLGain_Percent=c(mean(stock$Mean_DLGain),mean(stock$Median_DLGain),mean(stock$Original_DLGain)),
  Median_DLGain_Percent=c(median(stock$Mean_DLGain),median(stock$Median_DLGain),median(stock$Original_DLGain))
)

cat('\n=== PRIMARY MULTI-SEED COMPARISON, 33 STOCKS ===\n')
print(summary, row.names=FALSE, digits=4)
cat('\nStocks DL wins under ALL 4 seeds:',sum(stock$DL_Wins_All_4),'\n')
cat('Stocks ML wins under ALL 4 seeds:',sum(stock$ML_Wins_All_4),'\n')

# An optional descriptive per-run table for retaining Table 20.
seed_summary <- aggregate(DLGain_RMSE ~ SeedLabel, d, function(x) c(NWins=sum(x>0), Mean=mean(x), Median=median(x)))
seed_summary <- data.frame(Seed=seed_summary$SeedLabel,
                           DL_Better_Stocks=seed_summary$DLGain_RMSE[,1],
                           Mean_DLGain_Percent=seed_summary$DLGain_RMSE[,2],
                           Median_DLGain_Percent=seed_summary$DLGain_RMSE[,3])
seed_summary <- seed_summary[match(c('Original','Seed_101','Seed_202','Seed_303'),seed_summary$Seed), ]
cat('\n=== PER-SEED SENSITIVITY (SECONDARY) ===\n')
print(seed_summary, row.names=FALSE, digits=4)

outdir <- file.path(dirname(infile),'multi_seed_primary')
dir.create(outdir, showWarnings=FALSE)
write.csv(stock,file.path(outdir,'multi_seed_33_stock_results.csv'),row.names=FALSE)
write.csv(summary,file.path(outdir,'multi_seed_primary_summary.csv'),row.names=FALSE)
write.csv(seed_summary,file.path(outdir,'multi_seed_per_run_summary.csv'),row.names=FALSE)

pdf(file.path(outdir,'multi_seed_primary_DLGain_per_stock.pdf'),width=9,height=8)
par(mar=c(5,9,3,1))
o <- order(stock$Mean_DLGain)
cols <- ifelse(stock$Mean_DLGain[o]>0, 'steelblue','grey65')
barplot(stock$Mean_DLGain[o], names.arg=stock$Ticker[o], horiz=TRUE,
        las=1, cex.names=.7, col=cols, border=NA,
        xlab='Mean DLGain across four seeds (%)',
        main='Multi-seed DL advantage by stock (DL vs ML)')
abline(v=0, lty=2)
dev.off()
cat('\nSaved three CSVs and one PDF chart to: ',normalizePath(outdir,winslash='/'),'\n',sep='')
