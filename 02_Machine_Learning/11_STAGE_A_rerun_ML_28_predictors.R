# ============================================================================
# MASTER'S THESIS — STAGE A: CORRECTED 28-PREDICTOR ML RERUN
# ============================================================================
# Goal: make the traditional-ML information set match the 28 predictors
#       already used by the deep-learning experiment (adds DollarVolume).
#
# Methods: OLS, Ridge, LASSO, Bagging, Random Forest, Gradient Boosting.
# Training: 2010-2017; selection ONLY on validation 2018-2019;
# Test: 2020-2025, evaluated AFTER selection, never used to tune.
#
# This is a controlled correction of the ORIGINAL ML design, not the
# additional regularization experiment. Keep the original model settings.
# Do not compare different strategies by choosing the best TEST results.
#
# Requirements: model_data data.frame in RStudio, or select model_data.csv,
#                model_data.rds, or DL_export_bundle.zip when prompted.
# ============================================================================

required_packages <- c('glmnet','ranger','gbm')
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               FUN.VALUE=logical(1), quietly=TRUE)]
if (length(missing_packages)) {
  stop('Install first: install.packages(c(',
       paste(sprintf('"%s"',missing_packages),collapse=', '), '))')
}

# 0. Read the same fully engineered dataset used by the original experiment.
get_model_data <- function() {
  if (exists('model_data',envir=.GlobalEnv,inherits=FALSE)) {
    obj <- get('model_data',envir=.GlobalEnv)
    if (is.data.frame(obj)) return(obj)
  }
  message('Select model_data.csv / model_data.rds / DL_export_bundle.zip')
  f <- file.choose()
  ext <- tolower(tools::file_ext(f))
  if (ext == 'csv') return(read.csv(f, stringsAsFactors=FALSE,
                                check.names=FALSE))
  if (ext == 'rds') return(readRDS(f))
  if (ext == 'zip') {
    files <- utils::unzip(f, list=TRUE)$Name
    if (!'model_data.csv' %in% files)
      stop('Selected ZIP lacks model_data.csv.')
    con <- unz(f,'model_data.csv',open='r')
    on.exit(close(con),add=TRUE)
    return(read.csv(con, stringsAsFactors=FALSE,check.names=FALSE))
  }
  stop('Expected a CSV, RDS, or ZIP file, not: ',f)
}

model_data <- get_model_data()

# 1. Lock the 28 predictors to the DL export (including DollarVolume).
predictor_names <- c(
  paste0('ReturnLag',1:5),
  paste0('SMADist',c(5,10,20)),
  paste0('EMADist',c(5,10,20)),
  paste0('Momentum',c(5,10,20)),
  'RSI14','MACDRel','MACDSignalRel','BBPosition','StochK','StochD',
  'Vol5','Vol20','LogVolume','RelVolume20','DeltaOBV',
  'OvernightReturn','DollarVolume','ILLIQ20'
)
stopifnot(length(predictor_names)==28L,
          length(unique(predictor_names))==28L,
          'DollarVolume' %in% predictor_names)
needed_cols <- c('Ticker','sector','Date','TargetDate','Sample','Target',
                 'Return',predictor_names)
if (!all(needed_cols %in% names(model_data)))
  stop('model_data is missing: ',paste(setdiff(needed_cols,names(model_data)),
                                      collapse=', '))
model_data$Ticker <- as.character(model_data$Ticker)
model_data$Sample <- as.character(model_data$Sample)
model_data$Date <- as.Date(model_data$Date)
model_data$TargetDate <- as.Date(model_data$TargetDate)
model_data <- model_data[order(model_data$Ticker,model_data$Date),,
                         drop=FALSE]
if (anyNA(model_data[,c('Target',predictor_names)]) ||
    any(!is.finite(as.matrix(model_data[,c('Target',predictor_names)]))))
  stop('The modeling inputs contain NA, NaN, or infinite values.')
if (length(unique(model_data$Ticker))!=33L)
  stop('Expected 33 stocks, found ',length(unique(model_data$Ticker)))
if (!setequal(unique(model_data$Sample),c('Training','Validation','Test')))
  stop('Sample column should contain Training, Validation, and Test.')

# The per-stock index determines the original ranger/gbm seeds.
ticker_order <- unique(model_data$Ticker)

# Output goes into a NEW directory. Original files remain untouched.
outdir <- file.path(getwd(),'ML_28features_STAGE_A')
dir.create(outdir,showWarnings=FALSE,recursive=TRUE)
writeLines(predictor_names,file.path(outdir,'predictors_28.txt'))

rmse <- function(y,yhat) sqrt(mean((y-yhat)^2))
score <- function(y,yhat) {
  if (length(y)!=length(yhat) || !length(y) ||
      any(!is.finite(y)) || any(!is.finite(yhat)))
    stop('Non-finite or misaligned target and predictions.')
  err <- y-yhat
  b <- sqrt(mean(y^2))
  if (b==0) stop('Zero-return benchmark RMSE unexpectedly equals zero.')
  data.frame(RMSE=sqrt(mean(err^2)),MAE=mean(abs(err)),
             BenchmarkRMSE=b,BenchmarkMAE=mean(abs(y)),
             RelRMSE=sqrt(mean(err^2))/b,
             RelMAE=mean(abs(err))/mean(abs(y)),
             RMSEGain=100*(1-sqrt(mean(err^2))/b),
             MAEGain=100*(1-mean(abs(err))/mean(abs(y))),
             R2_OOS=1-sum(err^2)/sum(y^2),
             DirectionalAccuracy=mean(y*yhat>0))
}

scale_stock <- function(train,validation,test) {
  xtr <- as.matrix(train[,predictor_names,drop=FALSE])
  xva <- as.matrix(validation[,predictor_names,drop=FALSE])
  xte <- as.matrix(test[,predictor_names,drop=FALSE])
  means <- colMeans(xtr)
  sds <- apply(xtr,2,sd)
  sds[!is.finite(sds) | sds==0] <- 1
  transform <- function(x) sweep(sweep(x,2,means,'-'),2,sds,'/')
  list(train=transform(xtr),validation=transform(xva),test=transform(xte))
}

# Each candidate stores its chosen validation RMSE, hyperparameters,
# and predictions for training, validation, and test.
run_one_stock <- function(ticker,stock_number) {
  stock <- model_data[model_data$Ticker==ticker,,drop=FALSE]
  tr <- stock[stock$Sample=='Training',,drop=FALSE]
  va <- stock[stock$Sample=='Validation',,drop=FALSE]
  te <- stock[stock$Sample=='Test',,drop=FALSE]
  if (!nrow(tr) || !nrow(va) || !nrow(te)) stop('Missing sample for ',ticker)
  ytr <- tr$Target; yva <- va$Target; yte <- te$Target
  x <- scale_stock(tr,va,te)
  tree_train <- as.data.frame(tr[,c('Target',predictor_names),drop=FALSE])
  tree_va <- as.data.frame(va[,predictor_names,drop=FALSE])
  tree_te <- as.data.frame(te[,predictor_names,drop=FALSE])
  candidate <- list()
  keep <- function(name, ptrain, pvalid, ptest, hp) {
    candidate[[name]] <<- list(
      name=name,hyperparameters=hp,
      train=as.numeric(ptrain),validation=as.numeric(pvalid),
      test=as.numeric(ptest),validation_rmse=rmse(yva,pvalid)
    )
  }

  # OLS — identical specification, now with 28 columns.
  ols <- stats::lm(Target~.,data=data.frame(Target=ytr,x$train))
  ols_pred <- function(xmat) as.numeric(stats::predict(ols,
                                   newdata=data.frame(xmat)))
  keep('OLS',ols_pred(x$train),ols_pred(x$validation),ols_pred(x$test),'None')

  # Ridge/LASSO — choose lambda only by validation RMSE.
  for (method in c('Ridge','LASSO')) {
    alpha <- if (method=='Ridge') 0 else 1
    fit <- glmnet::glmnet(x=x$train,y=ytr,alpha=alpha,
                          standardize=FALSE,intercept=TRUE,nlambda=100)
    val_pred_grid <- as.matrix(stats::predict(fit,newx=x$validation,
                                              s=fit$lambda))
    val_scores <- apply(val_pred_grid,2,function(p) rmse(yva,p))
    idx <- which.min(val_scores)
    lambda <- fit$lambda[idx]
    p <- function(xx) as.numeric(stats::predict(fit,newx=xx,s=lambda))
    keep(method,p(x$train),p(x$validation),p(x$test),
         paste0('lambda=',signif(lambda,5)))
  }

  # Bagging — original grid and seed generation.
  best <- NULL
  for (node in c(5L,10L,20L)) {
    seed <- 1000L+stock_number+node
    set.seed(seed)
    fit <- ranger::ranger(Target~.,data=tree_train,num.trees=500,
                          mtry=length(predictor_names),min.node.size=node,
                          replace=TRUE,sample.fraction=1,seed=seed)
    pva <- as.numeric(predict(fit,data=tree_va)$predictions)
    val_error <- rmse(yva,pva)
    if (is.null(best) || val_error<best$err)
      best <- list(fit=fit,err=val_error,node=node)
  }
  ranger_predict <- function(dat) as.numeric(predict(best$fit,data=dat)$predictions)
  keep('Bagging',ranger_predict(tree_train),ranger_predict(tree_va),
       ranger_predict(tree_te),
       paste0('trees=500; min.node.size=',best$node))

  # Random Forest — original mtry/node-size tuning and seed generation.
  p <- length(predictor_names)
  mtry_grid <- unique(c(floor(sqrt(p)),floor(p/3),floor(p/2)))
  mtry_grid <- mtry_grid[mtry_grid>=1 & mtry_grid<p]
  best <- NULL
  for (mtry in mtry_grid) for (node in c(5L,10L,20L)) {
    seed <- 2000L+stock_number+mtry*10L+node
    set.seed(seed)
    fit <- ranger::ranger(Target~.,data=tree_train,num.trees=500,
                          mtry=mtry,min.node.size=node,
                          replace=TRUE,sample.fraction=1,seed=seed)
    pva <- as.numeric(predict(fit,data=tree_va)$predictions)
    val_error <- rmse(yva,pva)
    if (is.null(best) || val_error<best$err)
      best <- list(fit=fit,err=val_error,node=node,mtry=mtry)
  }
  ranger_predict <- function(dat) as.numeric(predict(best$fit,data=dat)$predictions)
  keep('Random Forest',ranger_predict(tree_train),ranger_predict(tree_va),
       ranger_predict(tree_te),
       sprintf('trees=500; mtry=%d; min.node.size=%d',best$mtry,best$node))

  # Gradient Boosting — original depth/shrinkage/number of trees grids.
  best <- NULL
  for (depth in c(1L,2L,3L)) for (shrinkage in c(0.01,0.05)) {
    seed <- 3000L+stock_number+depth*10L+round(shrinkage*100)
    set.seed(seed)
    fit <- gbm::gbm(Target~.,data=tree_train,distribution='gaussian',
                    n.trees=1000,interaction.depth=depth,
                    shrinkage=shrinkage,n.minobsinnode=10,
                    bag.fraction=1,train.fraction=1,verbose=FALSE)
    for (n_trees in c(100L,300L,500L,1000L)) {
      pva <- as.numeric(predict(fit,newdata=tree_va,n.trees=n_trees))
      val_error <- rmse(yva,pva)
      if (is.null(best) || val_error<best$err)
        best <- list(fit=fit,err=val_error,depth=depth,
                     shrinkage=shrinkage,n_trees=n_trees)
    }
  }
  boost_predict <- function(dat) as.numeric(predict(best$fit,newdata=dat,
                                                     n.trees=best$n_trees))
  keep('Gradient Boosting',boost_predict(tree_train),boost_predict(tree_va),
       boost_predict(tree_te),
       sprintf('trees=%d; depth=%d; shrinkage=%g',
               best$n_trees,best$depth,best$shrinkage))

  sector <- as.character(tr$sector[1])
  model_summaries <- do.call(rbind,lapply(candidate,function(z) {
    s_test <- score(yte,z$test)
    s_tr <- score(ytr,z$train)
    s_va <- score(yva,z$validation)
    data.frame(Sector=sector,Ticker=ticker,Model=z$name,
               Hyperparameters=z$hyperparameters,
               ValidationRMSE=z$validation_rmse,
               TestObservations=length(yte),
               s_test,
               TrainRMSE=s_tr$RMSE,
               TrainRMSEGain=s_tr$RMSEGain,
               ValidationRMSEGain=s_va$RMSEGain,
               Gap_pp=s_tr$RMSEGain-s_va$RMSEGain,
               stringsAsFactors=FALSE)
  }))
  # Select by validation RMSE ONLY, with original fixed model order tie-break.
  winner <- candidate[[which.min(vapply(candidate,`[[`,numeric(1),
                                               'validation_rmse'))]]
  selected_name <- winner$name

  predictions <- do.call(rbind,lapply(candidate,function(z) {
    data.frame(Ticker=ticker,sector=sector,Model=z$name,
               ForecastOrigin=as.Date(te$Date),
               TargetDate=as.Date(te$TargetDate),
               ActualReturn=yte,Prediction=z$test)
  }))
  selected_predictions <- data.frame(
    Ticker=ticker,sector=sector,Model=selected_name,
    ForecastOrigin=as.Date(te$Date),TargetDate=as.Date(te$TargetDate),
    ActualReturn=yte,Prediction=winner$test)

  list(summary=model_summaries,predictions=predictions,
       selected_predictions=selected_predictions)
}

# 2. Run every stock, using ORIGINAL model search grids and fixed seeds.
all_runs <- vector('list',length(ticker_order))
names(all_runs) <- ticker_order
for (i in seq_along(ticker_order)) {
  ticker <- ticker_order[i]
  message(sprintf('[%02d/33] Re-estimating six ML candidates for %s',i,ticker))
  all_runs[[i]] <- run_one_stock(ticker,i)
  # Save completed stock safely without touching the old experiment.
  saveRDS(all_runs[[i]],file.path(outdir,paste0('stock_',ticker,'.rds')))
  bestrow <- all_runs[[i]]$summary[
    which.min(all_runs[[i]]$summary$ValidationRMSE),,drop=FALSE]
  message(sprintf('    Selected: %s | validation RMSE=%.6f',
                  bestrow$Model,bestrow$ValidationRMSE))
}

# 3. Assemble files using column names compatible with original analyses.
ml_results_table <- do.call(rbind,lapply(all_runs,`[[`,'summary'))
ml_predictions <- do.call(rbind,lapply(all_runs,`[[`,'predictions'))
selected_ml_predictions <- do.call(rbind,lapply(all_runs,`[[`,
                                               'selected_predictions'))
selected_ml_models <- do.call(rbind,lapply(split(ml_results_table,
                                                ml_results_table$Ticker),
                                    function(x)x[which.min(x$ValidationRMSE),,
                                                 drop=FALSE]))
rownames(selected_ml_models) <- NULL
ml_overfit_results <- selected_ml_models[,c('Ticker','Model','TrainRMSE',
                                           'ValidationRMSE','TrainRMSEGain',
                                           'ValidationRMSEGain','Gap_pp')]

# Optional stock structural characteristics, recomputed training-only as
# in original code. This data frame is used by downstream group analyses.
characteristics <- do.call(rbind,lapply(ticker_order,function(ticker) {
  stock <- model_data[model_data$Ticker==ticker &
                        model_data$Sample=='Training',,drop=FALSE]
  data.frame(Ticker=ticker,sector=as.character(stock$sector[1]),
             TrainingVolatility=stats::sd(stock$Return)*sqrt(252),
             TrainingILLIQ=stats::median(stock$ILLIQ20),
             TrainingDollarVolume=stats::median(stock$DollarVolume))
}))
# Reproduce dplyr::ntile x -> group (3 equal groups; exactly 33 stocks).
assign_tertiles <- function(x) {
  o <- order(x,seq_along(x))
  groups <- integer(length(x))
  groups[o] <- rep(1:3,each=11)
  groups
}
characteristics$VolatilityRank <- assign_tertiles(characteristics$TrainingVolatility)
characteristics$LiquidityRank <- assign_tertiles(characteristics$TrainingILLIQ)
characteristics$VolatilityGroup <- c('Low','Medium','High')[characteristics$VolatilityRank]
characteristics$LiquidityGroup <- c('High','Medium','Low')[characteristics$LiquidityRank]
stock_characteristics <- characteristics

# Restore the structural variables used by the existing thesis R scripts.
char_cols <- c('Ticker','VolatilityGroup','LiquidityGroup',
               'TrainingVolatility','TrainingILLIQ','TrainingDollarVolume')
ml_results_table <- merge(ml_results_table,
                          characteristics[,char_cols,drop=FALSE],
                          by='Ticker',sort=FALSE)
ml_results_table <- ml_results_table[order(ml_results_table$Sector,
                                            ml_results_table$Ticker,
                                            ml_results_table$Model),,drop=FALSE]
selected_ml_models <- do.call(rbind,lapply(split(ml_results_table,
                                                ml_results_table$Ticker),
                                    function(x)x[which.min(x$ValidationRMSE),,
                                                 drop=FALSE]))
selected_ml_models <- selected_ml_models[match(ticker_order,
                                               selected_ml_models$Ticker),,
                                               drop=FALSE]
rownames(selected_ml_models) <- NULL

stopifnot(nrow(ml_results_table)==198L,nrow(selected_ml_models)==33L,
          nrow(ml_predictions)==33L*6L*1508L,
          nrow(selected_ml_predictions)==33L*1508L,
          !anyDuplicated(selected_ml_models$Ticker),
          !anyDuplicated(selected_ml_predictions[,c('Ticker','TargetDate')]))

write.csv(ml_results_table,
          file.path(outdir,'ML_28_all_198_model_results.csv'),row.names=FALSE)
write.csv(ml_predictions,
          file.path(outdir,'ML_28_all_model_test_predictions.csv'),row.names=FALSE)
write.csv(selected_ml_models,
          file.path(outdir,'selected_ml_models.csv'),row.names=FALSE)
write.csv(selected_ml_predictions,
          file.path(outdir,'selected_ml_predictions.csv'),row.names=FALSE)
write.csv(ml_overfit_results,
          file.path(outdir,'ML_28_selected_train_validation_gaps.csv'),
          row.names=FALSE)
write.csv(stock_characteristics,
          file.path(outdir,'stock_characteristics.csv'),row.names=FALSE)
saveRDS(list(ml_results_table=ml_results_table,
             ml_predictions=ml_predictions,
             selected_ml_models=selected_ml_models,
             selected_ml_predictions=selected_ml_predictions,
             ml_overfit_results=ml_overfit_results,
             stock_characteristics=stock_characteristics),
        file.path(outdir,'ML_28_ALL_results.rds'))

# Simple figure: selected ML train vs validation generalization gap.
pdf(file.path(outdir,'ML_28_generalization_gaps.pdf'),width=10,height=9)
vis <- ml_overfit_results[order(ml_overfit_results$Gap_pp),]
par(mar=c(4.5,5.5,2.5,1.2))
barplot(vis$Gap_pp,names.arg=vis$Ticker,las=1,horiz=TRUE,
        cex.names=.7,col=ifelse(vis$Gap_pp>=0,'#316b91','#ce854a'),
        border=NA,xlab='Train gain minus validation gain (percentage points)',
        main='Selected ML models: training–validation generalization gap')
abline(v=0,lty=2,col='gray45')
dev.off()

cat('\n================= STAGE A COMPLETE =================\n')
print(table(selected_ml_models$Model))
cat('Selected ML models beating ZERO (test):',
    sum(selected_ml_models$RMSEGain>0),'/33\n')
cat('Selected ML models with positive train–validation gap:',
    sum(ml_overfit_results$Gap_pp>0),'/33\n')
cat('Median training–validation gap (pp):',
    round(median(ml_overfit_results$Gap_pp),3),'\n')
cat('Outputs saved to: ',normalizePath(outdir),'\n',sep='')
cat('View(selected_ml_models)\nView(ml_overfit_results)\n')
cat('NB: We have already inspected the 2020–2025 test sample. ',
    'Treat this rerun as a documented consistency correction, ',
    'not an untouched confirmation experiment.\n',sep='')
