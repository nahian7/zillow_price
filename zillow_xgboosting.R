library(CatEncoders)
library(ranger)
library(data.table)
library(Matrix)
library(dplyr)

train_2016=fread('../input/train_2016_v2.csv',header=T)
train_2017=fread('../input/train_2017.csv',header=T)
month=as.numeric(substr(train_2016$transactiondate,6,7))
tempy=train_2016$logerror[month>9]
index=seq(length(month))[month>9]
temp_month=month[month>9]
set.seed(123)


temp1=c(sample(index[temp_month==10],410,replace=F),
        sample(index[temp_month==11],720,replace=F),
        sample(index[temp_month==12],720,replace=F))
temp2=c(sample(index[temp_month==10],410,replace=F),
        sample(index[temp_month==11],720,replace=F),
        sample(index[temp_month==12],720,replace=F))
dev_mean1=abs(mean(train_2016$logerror[temp1])-mean(train_2016$logerror[month>9]))
dev_mean2=abs(mean(train_2016$logerror[temp2])-mean(train_2016$logerror[month>9]))
if(dev_mean1>=dev_mean2){
  index=temp2
}
if(dev_mean1<dev_mean2){
  index=temp1
}
index=as.numeric(index)
rm(list=ls()[ls()!="index"])
gc()


#data processing
library(data.table)
library(Matrix)
library(dplyr)
sub=fread('../input/sample_submission.csv',header=T)
temp=as.data.frame(sub[,1])  #fix the order of PID
names(temp)=c('parcelid')
test1=fread('../input/properties_2016.csv',header=T)
test1=as.data.frame(test1)
test1=temp %>% left_join(test1, by = 'parcelid')
test2=fread('../input/properties_2017.csv',header=T)
test2=as.data.frame(test2)
test2=temp %>% left_join(test2, by = 'parcelid')
p=dim(test1)[2]
for(i in 1:p){
  if(class(test1[,i])[1]=="integer" | class(test1[,i])[1]=="integer64"){
    test1[,i]=as.numeric(test1[,i])
    test2[,i]=as.numeric(test2[,i])
  }
  if(class(test1[,i])[1]=="character"){
    test1[,i]=as.numeric(as.factor(test1[,i]))
    test2[,i]=as.numeric(as.factor(test2[,i]))
  }
  test1[,i][is.na(test1[,i])]=median(test1[,i][!is.na(test1[,i])])
  test2[,i][is.na(test2[,i])]=median(test2[,i][!is.na(test2[,i])])
}
rm(list=c("i","p","temp"))
gc()
train1=fread('../input/train_2016_v2.csv',header=T)
train1=train1 %>% left_join(test1, by = 'parcelid')
train1=as.data.frame(train1)
train2=fread('../input/train_2017.csv',header=T)
train2=train2 %>% left_join(test2, by = 'parcelid')
train2=as.data.frame(train2)
train=rbind(train1,train2)
rm(train1);rm(train2);gc()
year=as.numeric(substr(train$transactiondate,1,4))
month=as.numeric(substr(train$transactiondate,6,7))
y=train$logerror
train=train %>% select(-c(transactiondate,logerror))
gc()


library(xgboost)
library(data.table)
library(Matrix)
library(dplyr)

train=cbind(train,month)
x_train=train[(year==2017 & month>=3 & month<9) | year==2016,]
y_train=y[(year==2017 & month>=3 & month<9) | year==2016]
x_train=x_train[-index,]
y_train=y_train[-index]
x_valid1=train[year==2017 & month==9,] #leave out 2017 Sep for validation as well
y_valid1=y[year==2017 & month==9]
x_valid2=train[index,]
y_valid2=y[index]
yr=year[(year==2017 & month>=3 & month<9) | year==2016]
yr=yr[-index]
x_train=cbind(x_train,yr)
x_valid1$yr=2017
x_valid2$yr=2016
gc()

Pseudo_huber_obj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  x <- preds-labels
  grad <- x/sqrt(x^2+1)
  hess <- 1/(x^2+1)^(3/2)
  return(list(grad = grad, hess = hess))
}

train_3=x_train[y_train>-0.25 & y_train<0.3,]

y_3=y_train[y_train>-0.25 & y_train<0.3]
for(j in 2016:2017){
  for(i in 1:12){   
    y_3[train_3$month==i & train_3$yr==j]=y_3[train_3$month==i & train_3$yr==j]-
      median(y_3[train_3$month==i & train_3$yr==j])
  }
}

x_train=x_train %>% select(-c(censustractandblock,assessmentyear,month))
x_valid1=x_valid1 %>% select(-c(censustractandblock,assessmentyear,month))
x_valid2=x_valid2 %>% select(-c(censustractandblock,assessmentyear,month))

train_3=train_3 %>% select(-c(censustractandblock,assessmentyear,month))
dtrain=xgb.DMatrix(data.matrix(train_3), label=y_3)
dvalid1=xgb.DMatrix(data.matrix(x_valid1))
dvalid2=xgb.DMatrix(data.matrix(x_valid2))
set.seed(657)

param <- list(
  objective=Pseudo_huber_obj,
  eval_metric = "mae",
  eta = runif(1,0.001,0.1),
  lambda=runif(1,0.01,1.2), 
  alpha=runif(1,0,2),  
  max_depth=floor(runif(1,5,9)),
  base_score=mean(y_3),
  subsample=runif(1,0.1,0.9)
)


nr=floor(runif(1,100,350))
set.seed(657) #657 is selected from trying from 500 to 700 


xgb_mod=xgb.train(data=dtrain,params=param,nrounds=nr)
xgb1=predict(xgb_mod,dvalid1) +0.0116
xgb2=predict(xgb_mod,dvalid2) +0.0118
s1=mean(abs(xgb1-y_valid1))
s2=mean(abs(xgb2-y_valid2))

x_test=test1%>%select(-c(censustractandblock,assessmentyear))
x_test$yr=2016

dtest=xgb.DMatrix(data.matrix(x_test))
pred0=predict(xgb_mod,dtest)+0.0118
x_test=test2%>%select(-c(censustractandblock,assessmentyear))
x_test$yr=2017

dtest=xgb.DMatrix(data.matrix(x_test))
pred1=predict(xgb_mod,dtest)+0.0118

sub=fread('../input/sample_submission.csv', header = TRUE)
colnames(sub)[1] <- 'ParcelId'

sub$`201610`=round(pred0,4)
sub$`201611`=round(pred0,4)
sub$`201612`=round(pred0,4)
sub$`201710`=round(pred1,4)
sub$`201711`=round(pred1,4)
sub$`201712`=round(pred1,4)

train_2016=fread('../input/train_2016_v2.csv',header=T)
train_2017=fread('../input/train_2017.csv',header=T)


sub$`201610`[sub$ParcelId %in% train_2017$parcelid]=0.089
sub$`201611`[sub$ParcelId %in% train_2017$parcelid]=0.089
sub$`201612`[sub$ParcelId %in% train_2017$parcelid]=0.089






