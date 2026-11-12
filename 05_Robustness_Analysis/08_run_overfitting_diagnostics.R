# ================================================================
# THESIS: DL overfitting diagnostic from ORIGINAL PyTorch checkpoints
# Do NOT retrain or retune the full experiment.
# ================================================================
python_exe <- file.path(Sys.getenv('LOCALAPPDATA'),
                        'thesis-dl-torch', 'Scripts', 'python.exe')
if (!file.exists(python_exe)) stop('Cannot locate the original PyTorch environment.')

choose_existing <- function(default, description) {
  if (file.exists(default)) return(normalizePath(default, mustWork=TRUE))
  message('Choose ', description, ' in the file selection window.')
  normalizePath(file.choose(), mustWork=TRUE)
}

input_file <- choose_existing(file.path(getwd(), 'dl_input_33.csv'), 'dl_input_33.csv')
selected_file <- choose_existing(file.path(getwd(), 'dl_output', 'dl_selected_models.csv'),
                                 'the original dl_selected_models.csv')
trainer_file <- choose_existing(file.path(getwd(), 'train_dl_33.py'), 'train_dl_33.py')
checkpoints_dir <- file.path(dirname(selected_file), 'model_checkpoints')
if (!dir.exists(checkpoints_dir)) {
  message('Choose ANY .pt file from your original model_checkpoints folder.')
  checkpoint_example <- normalizePath(file.choose(), mustWork=TRUE)
  checkpoints_dir <- dirname(checkpoint_example)
}
python_file <- choose_existing(file.path(getwd(), '08_overfitting_diagnostics.py'),
                               '08_overfitting_diagnostics.py')
output_dir <- file.path(dirname(selected_file), 'overfitting_diagnostic')
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

cat('\nUsing saved weights at:', checkpoints_dir, '\n')
cat('No original models will be overwritten.\n')

# Keep FALSE for the first run. If you later want training curves,
# change to TRUE and rerun. Only four illustrative models are refitted.
DO_LEARNING_CURVES <- FALSE
args <- c(shQuote(python_file),
          '--input', shQuote(input_file),
          '--selected', shQuote(selected_file),
          '--trainer', shQuote(trainer_file),
          '--checkpoints', shQuote(checkpoints_dir),
          '--output', shQuote(output_dir))
if (DO_LEARNING_CURVES) args <- c(args, '--curve-tickers', 'AMT,NVDA,MSFT,KO')
status <- system2(python_exe, args=args, stdout='', stderr='', wait=TRUE)
if (!identical(as.integer(status), 0L)) stop('Diagnostic ended with a Python error.')

overfit_selected <- read.csv(file.path(output_dir, 'selected_DL_overfitting_gaps.csv'))
overfit_all <- read.csv(file.path(output_dir, 'all_architectures_train_validation_test.csv'))
cat('\nKEY SUMMARY\n')
cat('Number of stocks:', nrow(overfit_selected), '\n')
cat('Median Training - Validation gain (percentage points):',
    round(median(overfit_selected$TrainValGap_pp), 3), '\n')
cat('Training beats zero but validation loses:',
    sum(overfit_selected$TrainBetterThanZero_ValWorse), '/33\n')
print(overfit_selected[order(-overfit_selected$TrainValGap_pp),
                       c('Ticker','Model','RMSEGain_Training',
                         'RMSEGain_Validation','RMSEGain_Test','TrainValGap_pp')], row.names=FALSE)

# All plots are diagnostic/descriptive; no threshold can prove overfitting.
# Save PDF vector graphics directly for Overleaf.
pdf(file.path(output_dir, 'Overfit_01_Training_vs_validation.pdf'), width=7, height=6)
x <- overfit_selected$RMSEGain_Training
y <- overfit_selected$RMSEGain_Validation
lim <- range(c(x,y,0),finite=TRUE)
plot(x,y,pch=19,col='#2466a8',xlim=lim,ylim=lim,
     xlab='Training RMSE gain over zero (%)',
     ylab='Validation RMSE gain over zero (%)',
     main='Training vs validation: selected neural networks')
abline(a=0,b=1,lty=2,col='grey45')
abline(h=0,v=0,lty=3,col='grey75')
text(x,y,labels=overfit_selected$Ticker,pos=3,cex=0.56,col='grey30')
mtext('Below the diagonal: better fit on training than validation',side=1,line=4,cex=0.8)
dev.off()

pdf(file.path(output_dir, 'Overfit_02_Generalization_gap.pdf'),width=9,height=7)
ix <- order(overfit_selected$TrainValGap_pp)
barplot(overfit_selected$TrainValGap_pp[ix],
        names.arg=overfit_selected$Ticker[ix], las=2,
        col=ifelse(overfit_selected$TrainValGap_pp[ix]>0,'#3577a8','#c28a59'),
        cex.names=0.68, ylab='Train gain minus validation gain (pp)',
        main='Validation generalization gap by stock')
abline(h=0,col='grey35')
dev.off()

# optional curves are saved when DO_LEARNING_CURVES = TRUE
curve_file <- file.path(output_dir, 'illustrative_learning_curves.csv')
if (DO_LEARNING_CURVES && file.exists(curve_file)) {
  curves <- read.csv(curve_file)
  pdf(file.path(output_dir,'Overfit_03_Learning_curves.pdf'),width=8,height=7)
  par(mfrow=c(2,2),mar=c(4,4,3,1))
  for (tk in unique(curves$Ticker)) {
    dd <- curves[curves$Ticker==tk,]
    yy <- range(c(dd$TrainMSE_scaled,dd$ValidationMSE_scaled),finite=TRUE)
    plot(dd$Epoch,dd$TrainMSE_scaled,type='l',lwd=2,col='#2466a8',ylim=yy,
         xlab='Epoch',ylab='MSE (scaled target)',
         main=paste(tk,dd$Model[1]))
    lines(dd$Epoch,dd$ValidationMSE_scaled,lwd=2,col='#bf6533')
    abline(v=dd$Epoch[which.min(dd$ValidationMSE_scaled)],col='grey50',lty=3)
    legend('topright',legend=c('Training','Validation'),
           col=c('#2466a8','#bf6533'),lty=1,bty='n',cex=0.7)
  }
  par(mfrow=c(1,1))
  dev.off()
}
cat('\nDONE. CSV and PDF charts in:',output_dir,'\n')
cat('Inspect: View(overfit_selected)\n')
