#poissonglm.model <- rstan::stan_model(file="./jmseq/stan/poissonglm.stan", verbose=TRUE)
 
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

#' Split intervals of dataSurv and dataLong so that no interval is longer than max.interval
#'
#' @param dataSurv data table with one row per individual
#' @param dataLong data table with one row per biomarker observation
#' @param max.interval maximum interval length
#' @return data.table with one observation per interval
split.SurvLong <- function(dataSurv, dataLong, max.interval=1) {
    ids <- dataSurv$id

    ## left join dataLong with dataSurv on id
    setkey(dataLong, id)
    setkey(dataSurv, id)
    keep.cols <- c("id", "Time", "Time.cens", "event", timeinvar.surv, biomarkers)
    survLong <- dataSurv[dataLong]
    survLong <- survLong[, ..keep.cols]
    setorder(survLong, id, Time)
    survLong[, obsnum := .I, by=id]
    survLong[, lastobs := obsnum==max(obsnum), by=id]
    survLong[lastobs==FALSE, event := 0]
    
    ## add column for interval length
    survLong[, tstop := shift(Time, type="lead"), by=id]
    survLong[is.na(tstop), tstop := Time.cens]
    survLong[, interval.length := tstop - Time]
    ## add interval number in unsplit dataset
    survLong[, orig.interval := .I, by=id]
    
    cut.times <- cut.intervals(id=survLong$id, tstart=survLong$Time,
                               interval=survLong$interval.length, max.interval)

    ## loop over individuals to split intervals
    surv.cut <- NULL
    for(indiv in ids) {
        survlong.indiv <- survival::survSplit(formula=Surv(Time, tstop, event) ~ .,
                                              data=survLong[id==indiv],
                                              cut=cut.times[id==indiv, split.time])
        surv.cut <- rbind(surv.cut, survlong.indiv)
    }
    
    surv.cut <- as.data.table(surv.cut)
    ## set biomarker observations for split intervals to missing
    surv.cut[, splitinterval := c(1, diff(id)) + c(1, diff(orig.interval)) == 0]
    surv.cut[splitinterval == TRUE,
             (biomarkers) := lapply(.SD, function(x) rep(NA, length(x))),
             .SDcols=biomarkers]
    
    surv.cut[, splitinterval := NULL]
    return(surv.cut)
}

#' Split intervals so that no interval is longer than max.interval
#'
#' @param id Vector of ids.
#' @param tstart Vector of start times
#' @param interval.length Vector of lengths of intervals
#' @param max.interval Maximum interval length in output
#' @return data.table with colums id, cut.time that can be used to loop over ids calling survSplit. 
cut.intervals <- function(id, tstart, interval.length, max.interval) {
    cut.times <- NULL
    for(i in 1:length(tstart)) {
        if(interval.length[i] > max.interval) {
            tstarts <- tstart[i] +
                max.interval * (1:floor(interval.length[i] / max.interval))
            tstarts <- data.table(id=rep(id[i], length(tstarts)),
                                  split.time=tstarts)
            cut.times <- rbind(cut.times, tstarts)
        }
    }
    return(cut.times)
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
    stopifnot(all(c("id", "Time", "tstop", "event") %in% names(dataSurv)))
    dataSurv.train <- dataSurv[!(id %in% ids.test & tstop > landmark.time)]
    return(dataSurv.train)
}

#' Test/train split of longitudinal measurements, setting all biomarker measurements after  landmark.time to missing for all individuals in ids.test
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
trainsplit.long <- function(ids.test, dataLong, landmark.time=5, biomarkers) {
    stopifnot(all(c("id", "Time", "tstop", biomarkers) %in% names(dataLong)))
    dataLong.train <- copy(dataLong)
    dataLong.train[id %in% ids.test & Time > landmark.time, (biomarkers) := NA] 
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
    if("Long" %in% names(train.dataset)) {
        train.dataset <- train.dataset$Long
    }
    ctsem::ctStanFit(datalong=train.dataset, ## subset Long element in call to function
              ctstanmodel=ctmodel,
              nopriors=FALSE,
              cores=1, optimise=TRUE)
}

#' Impute values at beginning of each person-time interval from 0 to maxtime from ctstanfit object using Kalman filter, and cast to wide format
#' 
#' @param train.fit Object of class ctstanfit.
#' @param timestep Interval length.
#' @param subjects Vector of ids in field subject of train.fit
#' @param timerange Vector of length 2: start and end of time range. 
#' @return Data.table of imputed values with time in column tstart. 
#' 
#' @export
kalmanwide <- function(train.fit, subjects="all", timestep, timerange) {
    ## on training folds, call with timestep="asdata", timerange=c(0, maxtime), subjects=unique(train.fit$data$subject)
    ## on test folds, call with timestep=timestep" and timerange=c(landmark.time, maxtime), subjects=ids.test[[fold]]
    
    stopifnot(class(train.fit)=="ctStanFit")
    ## removeObs=FALSE ensures that observed biomarker values from train.fit are used in imputation
    ## ctKalman option timestep takes arguments "auto" or "asdata"
    ## "auto" gives sd(time) / 50
    if(subjects=="all") {
        subjects <- unique(train.fit$data$subject)
    }
    kal.tsplit <- ctsem::ctKalman(fit=train.fit,
                           timerange=timerange,
                           timestep=timestep, 
                           subjects=subjects, 
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

# kal.test <- impute.test(train.fit, landmark.time, maxtime)


#' Merge imputed biomarker values with survival data.table and fit Poisson regression model
#' 
#' @param kal.wide Data.table of imputed biomarker values, with time in column named tstart.     
#' @param dataSurv Data.table of survival values with columns id, Time.cens, event
#' @param timeinvar.surv Character vector of names of time-invariant covariates in dataSurv.train
#' @param biomarkers Character vector of names of biomarkers in dataSurv.train.
#' @param splines Logical variable as to whether to model time as a spline function.
#' @param stan Logical variable whether to use Stan to fit the Poisson regression
#' @param vb Logical variable whether to use variational Bayes approximation rather than optimising. 
#' @return Matrix of regression coefficients with one column.
#' 
#' @export
fit.poissontsplit <- function(kal.wide, dataSurv, timeinvar.surv, biomarkers,
                                splines=FALSE, stan=TRUE, vb=TRUE) {
    ## left join dataSurv with biomarkers imputed at intervals of length timestep, to fit Poisson regression model
    setkey(dataSurv, id, Time)
    setkey(kal.wide, id, tstart)
    tsplit.Surv <- kal.wide[dataSurv]
    #tsplit.Surv[, tstop := c(tstart[-1], NA), by=id]
    #tsplit.Surv[is.na(tstop), tstop := Time.cens]
    ## set event indicator to 0 for all person-time intervals ending before censoring date
    #tsplit.Surv[tstop < Time.cens, event := 0]
    ## restrict to person-time intervals before censoring date
    #tsplit.Surv <- tsplit.Surv[tstart < Time.cens]
    ## calculate tobs within each interval
    tsplit.Surv[, tobs := tstop - tstart]
    
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

    if(stan) {
        data.stan <- list(N=nrow(X),
                          P=ncol(X) - 1,
                          X=X[, -1], # no intercept
                          logtobs=log(tsplit.Surv$tobs), 
                          y=tsplit.Surv$event)

        if(vb) {
        poissonglm.vb <- rstan::vb(poissonglm.model,
                                                tol_rel_obj=0.005,
                                                elbo_samples=200, grad_samples=2,
                                                pars=c("beta0", "beta"),
                                   data=data.stan)
        coeffs <- as.data.frame(rstan::summary(poissonglm.vb, probs=c(0.025, 0.975))$c_summary)
        coeffs <- coeffs[1:(nrow(coeffs) - 1), ]
        } else {
         poissonglm.opt <- rstan::optimizing(poissonglm.model,
                                             data=data.stan)
         coeffs.opt <- poissonglm.opt$par
         coeffs <- coeffs.opt[grep("beta", names(coeffs.opt))]
         coeffs <- matrix(coeffs, ncol=1)
        }
        rownames(coeffs) <- colnames(X)
    } else {
        poissonglm.fit <- glm.fit(x=X, y=tsplit.Surv$event,
                                  family=poisson(), offset=log(tsplit.Surv$tobs))
        coeffs <- as.data.table(summary.glm(poissonglm.fit)$coefficients)
    }
    return(coeffs)
}

#' Predict event indicators from Poisson regression model using imputed biomarker values from kal.wide
#' 
#' @param beta Matrix of regression coefficients with one column.
#' @param imputed.dt Data.table of imputed values generated by kalmanwide(), one record per person-time interval.
#' @param dataSurv Data.table with column id used to fit ctsem model.
#' @param timeinvar.surv Character vector of names of time-invariant covariates in dataSurv.train.
#' @param biomarkers Character vector of names of biomarkers in dataSurv.train.
#' @param landmark.time Landmark time for start of prediction, as numeric variable. 
#' @return Data.table with columns id, event, tstart, tstop, toobs, covariates, p.event, cumulative survival prob for each individual. 
#' 
#' @export
predict.testdata <- function(beta, imputed.dt, dataSurv, timeinvar.surv, biomarkers, landmark.time) {
    ## left join tsplit.test with dataSurv
    dataSurv[, Time.cens := max(tstop), by=id]
    tsplit.test <- imputed.dt
    setkey(tsplit.test, id, tstart)
    setkey(dataSurv, id, Time)
    tsplit.test <- dataSurv[tsplit.test]
    setnames(tsplit.test, "Time", "tstart")
    ## tstart is time at which biomarkers are imputed
    ## restrict to person-time intervals starting after landmark.time 
    # tsplit.test <- tsplit.test[tstart >= landmark.time]
 
    ## do not truncate last interval at censoring date
    tobs.max <- with(tsplit.test, max(tstop - tstart, na.rm=TRUE)) # should be timestep
    tsplit.test[is.na(tstop) | tstop==Time.cens, tstop := tstart + tobs.max] 

    ## calculate tobs for each interval 
    tsplit.test[, tobs := tstop - tstart]
    ## set event indicator to 0 for all person-time intervals ending before censoring date
    if(!is.data.table(tsplit.test))
        tsplit.test <- as.data.table(tsplit.test)
    
    tsplit.test[, Time.cens := max(tstop), by="id"]
    tsplit.test[tstop < Time.cens, event := 0]
    keep.cols <- c("id", "event", "tstart", "tstop", "tobs", timeinvar.surv, biomarkers)
    tobs <- tsplit.test$tobs
    tsplit.test <- tsplit.test[, ..keep.cols]

    covariatenames <- c("tstart", timeinvar.surv, biomarkers)
    X.test <- model.matrix(~ ., data=tsplit.test[, ..covariatenames])

## predict using timestep as offset -- ignore censoring of last interval
    beta <- matrix(beta[[1]], ncol=1)
    tsplit.predict <- as.numeric(tobs * exp(X.test %*% beta)) # predict on scale of hazard rate
    ## calculate prob(events > 0) and cumulative survprob
    tsplit.test <- data.table(tsplit.test, p.event = 1 - exp(-tsplit.predict))
    tsplit.test[, survprob := cumprod(1 - p.event), by=id]
    return(tsplit.test)
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



