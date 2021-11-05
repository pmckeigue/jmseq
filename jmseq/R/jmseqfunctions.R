
#' Generate list of five ctStan models: linear mixed model (LMM), autoregressive drift and Wiener diffusion process only, LMM + drift, LMM + diffusion, LMM + drift + diffusion 
#'
#' @param biomarkers Character vector of names of biomarkers.
#' @param timeinvar.long Character vector of names of time-invariant covariates 
#' @return List of five objects of class ctStanModel.
#' @examples 
#' data(pbc)
#' models.list <- listmodels(pbc$biomarkers, pbc$timeinvar.long)
#' summary(models.list)
#' 
#' @export
listmodels <- function(biomarkers, timeinvar.long) {
    numbio <- length(biomarkers)
    model.lmm <- ctsem::ctModel(type='stanct',
                         manifestNames=biomarkers, 
                         latentNames=biomarkers,
                         MANIFESTMEANS=matrix(rep(0, numbio), nrow=numbio), 
                         DRIFT=matrix(rep(0, numbio^2), nrow=numbio), # autoregressive effect: A matrix
                         DIFFUSION=matrix(rep(0, numbio^2), nrow=numbio), # Wiener process: G matrix
                         CINT=matrix(paste0(rep('slope', numbio), 1:numbio), nrow=numbio),  
                         time="Time",
                         TDpredNames=NULL, 
                         TIpredNames = timeinvar.long,
                         LAMBDA=diag(numbio)) 
    model.nolmm <- ctsem::ctModel(type='stanct',
                           manifestNames=biomarkers,
                           latentNames=biomarkers,
                           MANIFESTMEANS=matrix(rep(0, numbio), nrow=numbio), 
                                        #DRIFT=matrix(rep(0, numbio^2), nrow=numbio), 
                                        #DIFFUSION=matrix(rep(0, numbio^2), nrow=numbio),
                                        #CINT=matrix(paste0(rep('slope', numbio), 1:numbio), nrow=numbio),  
                           time="Time",
                           TDpredNames=NULL, 
                           TIpredNames = timeinvar.long,
                           LAMBDA=diag(numbio))
    model.lmmdiff <- ctsem::ctModel(type='stanct',
                             manifestNames=biomarkers, 
                             latentNames=biomarkers,
                             DRIFT=matrix(rep(0, numbio^2), nrow=numbio), 
                                        #DIFFUSION=0, 
                             MANIFESTMEANS=matrix(rep(0, numbio), nrow=numbio),
                             CINT=matrix(paste0(rep('slope', numbio), 1:numbio), nrow=numbio),
                             time="Time",
                             TDpredNames=NULL, 
                             TIpredNames = timeinvar.long,
                             LAMBDA=diag(numbio))
    model.lmmdrift <- ctsem::ctModel(type='stanct',
                              manifestNames=biomarkers, 
                              latentNames=biomarkers,
                                        #DRIFT=0, 
                              DIFFUSION=matrix(rep(0, numbio^2), nrow=numbio), 
                              MANIFESTMEANS=matrix(rep(0, numbio), nrow=numbio),
                              CINT=matrix(paste0(rep('slope', numbio), 1:numbio), nrow=numbio),
                              time="Time",
                              TDpredNames=NULL, 
                              TIpredNames = timeinvar.long,
                              LAMBDA=diag(numbio)) 
    model.lmmdriftdiff <- ctsem::ctModel(type='stanct',
                                  manifestNames=biomarkers, 
                                  latentNames=biomarkers,
                                  MANIFESTMEANS=matrix(rep(0, numbio), nrow=numbio),
                                        #DRIFT=0, # 
                                        #DIFFUSION=0, 
                                  CINT=matrix(paste0(rep('slope', numbio), 1:numbio), nrow=numbio), 
                                  time="Time",
                                  TDpredNames=NULL, 
                                  TIpredNames = timeinvar.long,
                                  LAMBDA=diag(numbio))

    models.list <- list(model.lmm=model.lmm,
                        model.nolmm=model.nolmm,
                        model.lmmdiff=model.lmmdiff,
                        model.lmmdrift=model.lmmdrift,
                        model.lmmdriftdiff=model.lmmdriftdiff)
    return(models.list)
}

#' Test/train split of survival table, censoring follow-up at landmark.time for individuals in ids.test
#'
#' @param ids.test Vector of individual identifiers.
#' @param dataSurv Data.table with columns id = individual identifier, Time.cens = time of exit as numeric, event=0-1 indicator for event status at exit. 
#' @param landmark.time Landmark time at which to censor follow-up for individuals in ids.test.
#' @return Survival dataset censored at landmark.time for all individuals in ids.test, with Time renamed as Time.cens.  
#' 
#' @export
trainsplit.surv <- function(ids.test, dataSurv, landmark.time) {
    stopifnot(all(c("id", "Time.cens", "event") %in% names(dataSurv)))
    dataSurv.train <- data.table::copy(dataSurv) ## ? otherwise dataSurv will be modified by reference
    rows.censor <- which(dataSurv.train$id %in% ids.test)
    dataSurv.train <- as.data.table(dataSurv.train)
    stopifnot(is.data.table(dataSurv.train) == TRUE)
    dataSurv.train[rows.censor, event := 0]
    dataSurv.train[rows.censor, Time.cens := landmark.time]
    return(dataSurv.train)
}

#' Test/train split of longitudinal measurements, censoring follow-up at landmark.time for all individuals in ids.test
#'
#' @param ids.test Vector of individual identifiers.
#' @param dataSurv.train Data.table with columns id = individual identifier, Time.cens = time of exit.
#' @param dataLong Data.table of longitudinal measurements with column Time = observation time.
#' @param landmark.time Landmark time at which to censor follow-up for individuals in ids.test.
#' @param biomarkers Character vector of names of biomarker variables in dataLong.
#' @return Longitudinal dataset with one extra record for each individual at landmark.time, censored at landmark.time for all individuals in ids.test.
#'
#'
#' @export
trainsplit.long <- function(ids.test, dataSurv.train, dataLong, landmark.time=5, biomarkers) {
    stopifnot(all(c("id", biomarkers) %in% names(dataLong)))
    data.table::setkey(dataSurv.train, id)
    data.table::setkey(dataLong, id)
    dataLong.train <- dataSurv.train[, .(id, Time.cens)][dataLong]
    dataLong.train <- dataLong.train[Time < Time.cens]
    ## add a record for each individual with Time := landmark.time and missing biomarker values
    dataLong.lmtime <- unique(dataLong.train[id %in% ids.test], by="id")
    dataLong.lmtime[, Time := landmark.time]
    dataLong.lmtime[, (biomarkers) := NA]
    dataLong.train <- rbind(dataLong.train, dataLong.lmtime) 
    return(dataLong.train)
}

#' Fit ctsem model to longitudinal dataset
#' 
#' @param train.dataset Longitudinal dataset.
#' @param ctmodel ctmodel.
#' @return object of class ctStanFit. 
#' 
#' @export
ctstanfit.fold <- function(train.dataset, ctmodel) {
    ctsem::ctStanFit(datalong=train.dataset$Long,
              ctstanmodel=ctmodel,
              nopriors=FALSE,
              cores=1, optimise=TRUE)
}

#' Impute values at beginning of each person-time interval from 0 to maxtime from ctstanfit object using Kalman filter, and cast to wide format
#' 
#' @param train.fit Object of class ctstanfit.
#' @param timestep Interval length.
#' @param maxtime Maximum time up to which to generate imputations.
#' @return Data.table of imputed values with time in column tstart. 
#' 
#' @export
kalmanwide <- function(train.fit, timestep, maxtime=maxtime) {
    stopifnot(class(train.fit)=="ctStanFit")
    ## removeObs=FALSE ensures that observed biomarker values from train.fit are used in imputation
    ## ctKalman option timestep takes arguments "auto" or "asdata"
    ## "auto" gives sd(time) / 50 
    kal.tsplit <- ctsem::ctKalman(fit=train.fit,
                           timerange=c(0, maxtime),
                           timestep=timestep, 
                           subjects=unique(train.fit$data$subject), 
                           removeObs=FALSE, plot=FALSE)
    kal.tsplit <- as.data.table(kal.tsplit)
    kal.tsplit <- kal.tsplit[Element=="yupd"]
    kal.wide <- dcast(kal.tsplit, formula=Subject + Time ~ Row)
    ## FIXME -- this should work when id is not integer
    kal.wide[, id := as.integer(as.character(Subject))]
    setnames(kal.wide, "Time", "tstart", skip_absent=TRUE) # time at which biomarkers are imputed 
    setkey(kal.wide, id)
    return(kal.wide)
}

#' Merge imputed biomarker values with survival data.table and fit Poisson regression model
#' 
#' @param kal.wide Data.table of imputed biomarker values, with time in column named tstart.     
#' @param dataSurv Data.table of survival values with columns id, Time.cens, event
#' @param timeinvar.surv Character vector of names of time-invariant covariates in dataSurv.train
#' @param biomarkers Character vector of names of biomarkers in dataSurv.train.
#' @param splines Logical variable as to whether to model time as a spline function. 
#' @return Matrix of regression coefficients with one column.
#' 
#' @export
fit.poissontsplit <- function(kal.wide, dataSurv, timeinvar.surv, biomarkers,
                                splines=FALSE) {
    ## left join dataSurv with biomarkers imputed at intervals of length timestep, to fit Poisson regression model
    setkey(dataSurv, id)
    tsplit.Surv <- kal.wide[dataSurv]
    tsplit.Surv[, tstop := c(tstart[-1], NA), by=id]
    tsplit.Surv[is.na(tstop), tstop := Time.cens]
    ## set event indicator to 0 for all person-time intervals ending before censoring date
    tsplit.Surv[tstop < Time.cens, event := 0]
    ## restrict to person-time intervals before censoring date
    tsplit.Surv <- tsplit.Surv[tstart < Time.cens]
    ## calculate tobs within each interval
    tsplit.Surv[, tobs := pmin(tstop, Time.cens) - tstart]
    
    keep.vars <- c("id", "event", "tstart", "tstop", "tobs",
                   timeinvar.surv, biomarkers)
    tsplit.Surv <- tsplit.Surv[, ..keep.vars]
    X <- model.matrix(
        as.formula(paste0("event ~ ",
                          ifelse(splines,
                                 "splines::bs(tstart, df=6)", 
                                 "tstart"),
                          " + ",
                          paste(timeinvar.surv, collapse=" + "),
                          " + ",
                          paste(biomarkers, collapse=" + "))),
        data=tsplit.Surv)

    stan <- TRUE
    if(stan) {
        data.stan <- list(N=nrow(X),
                          P=ncol(X) - 1,
                          X=X[, -1], # no intercept
                          logtobs=log(tsplit.Surv$tobs), 
                          y=tsplit.Surv$event)

        #poissonglm.model <- rstan::stan_model(file="jmseq/stan/poissonglm.stan")
        #poissonglm.model <- poissonglm.stanmodel()
        poissonglm.opt <- rstan::optimizing(poissonglm.model,
                                        #tol_rel_obj=0.005, elbo_samples=200, grad_samples=2,
                                        #pars=c("beta0", "beta"),
                                            data=data.stan)
        coeffs.opt <- poissonglm.opt$par
        beta <- coeffs.opt[grep("beta", names(coeffs.opt))]
        names(beta) <- colnames(X)
    } else {
        poissonglm.fit <- glm.fit(x=X, y=tsplit.Surv$event,
                                  family=poisson(), offset=log(tsplit.Surv$tobs))
        beta <- matrix(poisson.glm.fit$coefficients, ncol=1)
        rownames(beta) <- names(poisson.glm.train$coefficients)
    }
    
    return(beta)
}

#' Summary statistics for predictive performance
#' @param testdata data.table with columns event, p.event, tobs.
#' 
#' @export
tabulate.predictions <- function(testdata) {
    pred <- copy(testdata)
    pred[, logscore := event * log(p.event) + (1 - event) * log(1 - p.event)]
    pred.sums <- pred[, lapply(.SD, sum), .SDcols=c("event", "p.event", "tobs", "logscore")]
    pred.roc <- with(pred, pROC::roc(event, p.event))
    C_statistic=as.numeric(pROC::auc(pred.roc))
    stats <- c(pred.sums, C_statistic)
    names(stats) <- c("Observed", "Predicted", "Person-years", "Log score", "C-statistic") 
    return(stats)
}

#' Predict event indicators from Poisson regression model using imputed biomarker values from kal.wide
#' 
#' @param beta Matrix of regression coefficients with one column.
#' @param ids.test Vector of individual identifiers
#' @param kal.wide Data.table of imputed values generated by kalmanwide(), one record per person-time interval.
#' @param dataSurv Data.table with column id used to fit ctsem model.
#' @param timeinvar.surv Character vector of names of time-invariant covariates in dataSurv.train.
#' @param biomarkers Character vector of names of biomarkers in dataSurv.train.
#' @param landmark.time Landmark time for start of prediction, as numeric variable. 
#' @return Data.table with columns id, event, tstart, tstop, toobs, covariates, p.event, cumulative survival prob for each individual. 
#' 
#' @export
test.imputed <- function(beta, ids.test, kal.wide, dataSurv,
                           landmark.time, 
                         timeinvar.surv, biomarkers) {
    tsplit.test <- kal.wide[id %in% ids.test]
    ## left join tsplit.test with dataSurv 
    setkey(tsplit.test, id)
    setkey(dataSurv, id)
    tsplit.test <- dataSurv[tsplit.test]
    ## tstart is time at which biomarkers are imputed
    ## restrict to person-time intervals starting after landmark.time and before censoring date
    tsplit.test <- tsplit.test[tstart >= landmark.time]
    tsplit.test <- tsplit.test[tstart < Time.cens]
    ## set end of each person-time interval
    tsplit.test[, tstop := c(tstart[-1], NA), by=id]

    ## do not truncate last interval at censoring date
    tobs.max <- with(tsplit.test, max(tstop - tstart, na.rm=TRUE))
    tsplit.test[is.na(tstop), tstop := tstart + tobs.max] ## should be timestep

    ## calculate tobs for each interval 
    tsplit.test[, tobs := tstop - tstart]
    ## set event indicator to 0 for all person-time intervals ending before censoring date
    if(!is.data.table(tsplit.test))
        tsplit.test <- as.data.table(tsplit.test)
    tsplit.test[tstop < Time.cens, event := 0]
    #tsplit.test[event==1, tobs := timestep]
    keep.cols <- c("id", "event", "tstart", "tstop", "tobs", timeinvar.surv, biomarkers)
    tobs <- tsplit.test$tobs
    tsplit.test <- tsplit.test[, ..keep.cols]

    covariatenames <- c("tstart", timeinvar.surv, biomarkers)
    X.test <- model.matrix(~ ., data=tsplit.test[, ..covariatenames])

   ## predict using timestep as offset -- ignore censoring of last interval
    tsplit.predict <- as.numeric(tobs * exp(X.test %*% beta)) # predict on scale of hazard rate
    ## calculate prob(events > 0) and cumulative survprob
    tsplit.test <- data.table(tsplit.test, p.event = 1 - exp(-tsplit.predict))
    tsplit.test[, survprob := cumprod(1 - p.event), by=id]
}


