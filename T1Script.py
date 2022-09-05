#Packages
import pandas as pd
import numpy as np
import pyreadr as pyr
import sklearn as sk
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from bootstrap_stat import bootstrap_stat as bp
from scipy.stats import bootstrap
from sklearn.impute import KNNImputer

#Setting work space and master data frame:
bd=pyr.read_r("bdPS1.Rdata")
print(bd.keys()) #Load database.
df0=bd["df"]
df=df0.drop(df0[df0.age<18].index) #Restrict the data to legal age individuals (age>=18).
df=df.drop(df[df.ocu==0].index) #Restrict the data to employed individuals (ocu==1).
dfsi=(df[["age", "maxEducLevel", "sizeFirm", "oficio", "hoursWorkUsual", "formal", "ingtot", "mes", "estrato1", "sex", 
"regSalud", "p6426", "p7090","p7510s7a1", "impa", "isa", "impaes", "isaes", "relab"]]) #Relevant variables selection.

#Reassining variables codification:
dfsi["impa"].replace(0, 1, inplace=True)
dfsi["impa"].replace(np.nan, 0, inplace=True)
dfsi["impaes"].replace(0, 1, inplace=True)
dfsi["impaes"].replace(np.nan, 0, inplace=True)
dfsi["isa"].replace(0, 1, inplace=True)
dfsi["isa"].replace(np.nan, 0, inplace=True)
dfsi["isaes"].replace(0, 1, inplace=True)
dfsi["isaes"].replace(np.nan, 0, inplace=True)

#Create compiled variables:
dfsi["impacomp"]=dfsi["impa"]+dfsi["impaes"]
dfsi["isacomp"]=dfsi["isa"]+dfsi["isaes"]

#Recover missing values: 
dfsi['impacomp']=dfsi['impacomp'].replace(0,np.nan)
dfsi['isacomp']=dfsi['isacomp'].replace(0,np.nan)
dfsi['impacomp']=dfsi['impacomp'].replace(1, 0)
dfsi['isacomp']=dfsi['isacomp'].replace(1, 0)
dfsi['impacomp']=dfsi['impacomp'].replace(2, 0)
dfsi['isacomp']=dfsi['isacomp'].replace(2, 0)

#Missing values count/share. 
dfsi.isnull().sum() 
dfsi.isnull().sum()/len(dfsi) 

#Imputing missing values:
imputer=KNNImputer(n_neighbors=19) 
imputer.fit(dfsi)
dfi=pd.DataFrame(imputer.transform(dfsi))
dict=({0: "age", 1: "maxEducLevel", 2: "sizeFirm", 3: "oficio", 4: "hoursWorkUsual", 5: "formal", 6: "ingtot", 7: "mes", 8: "estrato1", 
9: "sex", 10: "regSalud", 11: "p6426", 12: "p7090", 13: "p7510s7a1", 14: "impa", 15: "isa", 16: "impaes", 17: "isaes", 18: "relab", 
19: "impacomp", 20: "isacomp"})
dfi.rename(columns=dict, inplace=True)

#Convert categorical variables to dummy variables:
oficio_d=pd.get_dummies(dfi["oficio"], prefix="oficio") 
sizeFirm_d=pd.get_dummies(dfi["sizeFirm"], prefix="size") 
estrato1_d=pd.get_dummies(dfi["estrato1"], prefix="estrato") 
regSalud_d=pd.get_dummies(dfi["regSalud"], prefix="regs") 
maxeduc_d=pd.get_dummies(dfi["maxEducLevel"], prefix="educ") 
relab_d=pd.get_dummies(dfi["relab"], prefix="relab") 
mes_d=pd.get_dummies(dfi["mes"], prefix="mes")

#Merge dummy's variables data frame with master data frame:
dfi=pd.merge(dfi, oficio_d, left_index=True, right_index=True) 
dfi=pd.merge(dfi, sizeFirm_d, left_index=True, right_index=True)
dfi=pd.merge(dfi, estrato1_d, left_index=True, right_index=True)
dfi=pd.merge(dfi, regSalud_d, left_index=True, right_index=True)
dfi=pd.merge(dfi, maxeduc_d, left_index=True, right_index=True)
dfi=pd.merge(dfi, relab_d, left_index=True, right_index=True)
dfi=pd.merge(dfi, mes_d, left_index=True, right_index=True)
dfi["p7090"]=np.where(dfi["p7090"]==2, 0, 1)

#Transform categorial atribute variables to numerical variable: 
dfi["hoursWorkUsual"]=pd.to_numeric(dfi["hoursWorkUsual"])
dfi["formal"]=pd.to_numeric(dfi["formal"])
dfi["p6426"]=pd.to_numeric(dfi["p6426"])

#Descriptive statistics data frame:
ds=(dfi[["age", "hoursWorkUsual", "p6426", "sex", "formal", "p7090", "estrato_1.0", "estrato_2.0", "estrato_3.0", 
"estrato_4.0", "estrato_5.0", "estrato_6.0", "regs_1.0", "regs_2.0", "regs_3.0", "educ_1.0", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0"]].describe(include="all"))
ds=ds.T
ds=ds[["count", "mean", "std", "min", "50%", "max"]]
ds=ds.round(2)
print(ds.to_latex())

#Create dependent variable:
dfi["earnings"]=dfi["impacomp"]+dfi["isacomp"]
dfi["earnings"].describe()

#Winsorize data:
sc.stats.mstats.winsorize(dfi["earnings"], limits=[0,0.05], inplace=True)
sc.stats.mstats.winsorize(dfi["age"], limits=[0,0.05], inplace=True)
dfi["earnings"].describe()

#Create cuadratic age variable:
dfi["age2"]=dfi["age"]**2

#OLS regresion model:
y=dfi["earnings"]
x=dfi[["age", "age2"]]
x=sm.add_constant(x)
model1=sm.OLS(y,x)
results=model1.fit()
yhatp1=results.predict()
b1=results.params[1]
b2=results.params[2]
elasticidad=-b1/2*b2
print(results.summary())
print(np.sqrt(results.mse_resid))

#Bootstrap 
seboot=[0]*3
for j in range(0,3):
    def estimador(dfi):
        y=dfi["earnings"]
        x=dfi[["age", "age2"]]
        x=sm.add_constant(x)
        model1=sm.OLS(y,x)
        results= model1.fit()
        return results.params[j]

    dist = bp.EmpiricalDistribution(dfi)
    sea=bp.standard_error(dist, estimador, B=1000)
    seboot[j]=sea

b0=results.params[0]
b1=results.params[1]
b2=results.params[2]

b0low=b0-seboot[0]*1.96
b1low=b1-seboot[1]*1.96
b2low=b2-seboot[2]*1.96

b0upper=b0+seboot[0]*1.96
b1upper=b1+seboot[1]*1.96
b2upper=b2+seboot[2]*1.96

yhatlow=b0low+b1low*dfi['age']+b2low*dfi['age2']
yhatupper=b0upper+b1upper*dfi['age']+b2upper*dfi['age2']

#Figura 1
x_f1=pd.DataFrame(dfi['age'])
x_f1["Edad"]=x_f1
y_f1=pd.DataFrame(yhatp1)
y_f1["COP"]=y_f1
z_f1=pd.DataFrame(yhatlow)
z_f1["3"]=z_f1
w_f1=pd.DataFrame(yhatupper)
w_f1["4"]=w_f1

plot_data1=pd.concat([x_f1,y_f1,z_f1,w_f1], axis=1, join="inner")
plot_data1

fig, ax = plt.subplots()
ax= sns.lineplot(x='Edad', y='COP', data=plot_data1, label='Ingreso laboral', linewidth=2, color="black")
ax2 = sns.lineplot(x='Edad', y='4', data=plot_data1, label='IC banda superior', linewidth=1, color="gray")
ax1 = sns.lineplot(x='Edad', y='3', data=plot_data1, label='IC banda inferior', linewidth=1, color="gray")

#Create female variable:
dfi["female"]=np.where(dfi["sex"]==1, 0, 1)
dfi=dfi.drop(dfi[dfi.earnings==0].index)
dfi["log_earnings"]=np.log(dfi["earnings"])
dfi["log_earnings"].describe()

#OLS regresion model:
y=dfi["log_earnings"]
x=dfi["female"]
x=sm.add_constant(x)
model2=sm.OLS(y,x)
results= model2.fit()
print(results.summary())
print(np.sqrt(results.mse_resid))

#rMSE to COP units:
y_hat=results.predict()
from math import e
y_hat2=e**(y_hat)
E = dfi["earnings"]-y_hat2
E2 = E**2
MSE=sum(E2)/(len(E2)-2)
print(np.sqrt(MSE))

#Create interactions variable:
dfi["female_age"]=dfi["female"]*dfi["age"]
dfi["female_age2"]=dfi["female"]*dfi["age2"]

#OLS regresion model:
y=dfi["log_earnings"]
x=dfi[["age", "female", "female_age", "female_age2", "age2"]]
x=sm.add_constant(x)
model3=sm.OLS(y,x)
results= model3.fit()
yhatp3=results.predict()
print(results.summary())
results.params

#rMSE to COP units:
y_hat=results.predict()
from math import e
y_hat2=e**(y_hat)
E = dfi["earnings"]-y_hat2
E2 = E**2
MSE=sum(E2)/(len(E2)-6)
print(np.sqrt(MSE))

#Bootstrap 
seboot=[0]*6
for j in range(0,6):
    def estimador3(dfi):
        y=dfi["log_earnings"]
        x=dfi[["age", "female", "female_age", "female_age2", "age2"]]
        x=sm.add_constant(x)
        model1=sm.OLS(y,x)
        results= model1.fit()
        return results.params[j]

    dist = bp.EmpiricalDistribution(dfi)
    sea=bp.standard_error(dist, estimador3, B=1000)
    seboot[j]=sea

seboot

for k in range(0,6):
    locals()['b'+str(k)]=results.params[k]
    locals()['b'+str(k)+'low']=locals()['b'+str(k)]-seboot[k]*1.96
    locals()['b'+str(k)+'upper']=locals()['b'+str(k)]+seboot[k]*1.96


yhatlow=b0low+b1low*dfi['age']+b5low*dfi['age2']+b2low*dfi['female']+b3low*dfi['female_age']+b4low*dfi['female_age2']
yhatupper=b0upper+b1upper*dfi['age']+b5upper*dfi['age2']+b2upper*dfi['female']+b3upper*dfi['female_age']+b4upper*dfi['female_age2']

yhatupperh= np.ma.masked_where(dfi['female']==0, yhatupper)
yhatlowh = np.ma.masked_where(dfi['female']==0, yhatlow)

yhatupperm= np.ma.masked_where(dfi['female']==1, yhatupper)
yhatlowm = np.ma.masked_where(dfi['female']==1,yhatlow )

yhatp3m=np.ma.masked_where(dfi['female']==1, yhatp3)
yhatp3h=np.ma.masked_where(dfi['female']==0, yhatp3)

agem=np.ma.masked_where(dfi['female']==1, dfi['age'])
ageh=np.ma.masked_where(dfi['female']==0, dfi['age'])

#Figura 2m
x_f2m=pd.DataFrame(agem)
x_f2m["Edad"]=x_f2m
y_f2m=pd.DataFrame(yhatp3m)
y_f2m["COP"]=y_f2m
z_f2m=pd.DataFrame(yhatlowm)
z_f2m["3"]=z_f2m
w_f2m=pd.DataFrame(yhatupperm)
w_f2m["4"]=w_f2m

plot_data2m=pd.concat([x_f2m,y_f2m,z_f2m,w_f2m], axis=1, join="inner")
plot_data2m

fig, ax = plt.subplots()
ax= sns.lineplot(x='Edad', y='COP', data=plot_data2m, label='Ingreso laboral mujeres', linewidth=2, color="red")
ax2 = sns.lineplot(x='Edad', y='4', data=plot_data2m, label='IC banda superior', linewidth=1, color="orange")
ax1 = sns.lineplot(x='Edad', y='3', data=plot_data2m, label='IC banda inferior', linewidth=1, color="orange")

#Figura 2h
x_f2h=pd.DataFrame(ageh)
x_f2h["Edad"]=x_f2h
y_f2h=pd.DataFrame(yhatp3h)
y_f2h["COP"]=y_f2h
z_f2h=pd.DataFrame(yhatlowh)
z_f2h["3"]=z_f2h
w_f2h=pd.DataFrame(yhatupperh)
w_f2h["4"]=w_f2h

plot_data2h=pd.concat([x_f2h,y_f2h,z_f2h,w_f2h], axis=1, join="inner")
plot_data2h

fig, ax = plt.subplots()
ax= sns.lineplot(x='Edad', y='COP', data=plot_data2h, label='Ingreso laboral hombres', linewidth=2, color="blue")
ax2 = sns.lineplot(x='Edad', y='4', data=plot_data2h, label='IC banda superior', linewidth=1, color="green")
ax1 = sns.lineplot(x='Edad', y='3', data=plot_data2h, label='IC banda inferior', linewidth=1, color="green")

oficio=["oficio_1.0"]
for i in range(2,99):
    oficio.append("oficio_"+str(i)+".0")
    list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
    oficio2=[i for i in oficio if i not in list1]

#Long regression
#Define exogenous and endogenous variables:
y=dfi["log_earnings"]
x0=dfi[["female", "age", "age2", "female_age", "female_age2"]]
x1=(dfi[["sizeFirm", "hoursWorkUsual", "formal", "p6426", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0", "relab_5.0", "relab_6.0", 
"relab_7.0", "relab_8.0" , "relab_9.0", "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", 
"mes_7.0", "mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]])
x2=dfi[oficio2]
x=pd.concat([x0,x1,x2], axis=1, join="inner")
x=sm.add_constant(x)

#OLS regresion model:
model2=sm.OLS(y,x)
results= model2.fit()
print(results.summary())
print(np.sqrt(results.mse_resid))

#rMSE to COP units:
y_hat=results.predict()
from math import e
y_hat2=e**(y_hat)
E = dfi["earnings"]-y_hat2
E2 = E**2
MSE=sum(E2)/(len(E2)-131)
print(np.sqrt(MSE))

#FWL theorem
#Define exog and endog variables
y=dfi["log_earnings"]
x0=dfi[["female", "age", "age2", "female_age", "female_age2"]]
x_f=pd.concat([x1,x2], axis=1, join="inner")
x_f=sm.add_constant(x_f)
x0=sm.add_constant(x0)

#Residuals of log_earnings regressed on controls
model3=sm.OLS(y,x_f)
results_f= model3.fit()
resid_a=results_f.resid

#Residuals of female regressed on controls
model4=sm.OLS(x0, x_f)
results_f2= model4.fit()
resid_b=results_f2.resid

#Delete constant from residuals b
resid_b=resid_b.drop(["const"], axis=1)

#Results of resid_a  regressed on resid_b
model_FWL=sm.OLS(resid_a,resid_b)
results_FWL=model_FWL.fit()
print(results_FWL.summary())
print(np.sqrt(results.mse_resid))

#rMSE to COP units:
y_hat=results.predict()
from math import e
y_hat2=e**(y_hat)
E = dfi["earnings"]-y_hat2
E2 = E**2
MSE=sum(E2)/(len(E2)-5)
print(np.sqrt(MSE))

#Bootstrap FWL
se_boot = []
for koef in range(0,5):

    def estm(dfi):

        #FWL theorem
        #Define exog and endog variables
        oficio=["oficio_1.0"]
        for i in range(2,99):
            oficio.append("oficio_"+str(i)+".0")
            list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
            oficio2=[i for i in oficio if i not in list1]

        y=dfi["log_earnings"]
        x0=dfi[['female', "age", "age2", "female_age", "female_age2"]]
        x1=dfi[["sizeFirm", "hoursWorkUsual","formal", "p6426","educ_3.0", "educ_4.0", 
            "educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0","relab_5.0" , "relab_6.0" , "relab_7.0" , "relab_8.0" , "relab_9.0",
            "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", "mes_7.0","mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]]
        x2=dfi[oficio2]
        x_f = pd.concat([x1,x2], axis=1, join="inner")
        x_f=sm.add_constant(x_f)
        x0=sm.add_constant(x0)

        #Residuals of log_earnings regressed on controls
        model3=sm.OLS(y,x_f)
        results_f= model3.fit()
        resid_a=results_f.resid

        #residuals of female regressed on controls
        model4=sm.OLS(x0, x_f)
        results_f2= model4.fit()
        resid_b=results_f2.resid

        #Delete constant from residuals b
        resid_b=resid_b.drop(['const'], axis=1)

        #Results of resid_a  regressed on resid_b
        model_FWL=sm.OLS(resid_a,resid_b)
        results_FWL=model_FWL.fit()

        return results_FWL.params[koef]

    dist = bp.EmpiricalDistribution(dfi)
    se_koef=bp.standard_error(dist,estm, B=5)
    se_boot = se_boot + [str(se_koef)]
    print(se_koef)

#Create new variables
dfi['age_hours']=dfi['age']*dfi['hoursWorkUsual']
dfi['hours2']=dfi['hoursWorkUsual']**2
dfi['age_hours2']=dfi['age']*dfi['hours2']

for j in range(1,8):
    if j==2:
        pass
    else:
        dfi["female_educ"+str(j)]=dfi["female"]*dfi["educ_"+str(j)+".0"]

for j in range(1,10):
        dfi["female_relab"+str(j)]=dfi['female']*dfi["relab_"+str(j)+".0"]

for j in range(1,8):
    if j==2:
        pass
    else:
        dfi["age_educ"+str(j)]=dfi["age"]*dfi["educ_"+str(j)+".0"]

#Create new variables
dfi["p6426_2"]=dfi["p6426"]**2
dfi["p6426_3"]=dfi["p6426"]**3
dfi["p6426_age"]=dfi["p6426"]*dfi["age"]
dfi["p6426_2_age"]=dfi["p6426_2"]*dfi["age"]
dfi["p6426_3_age"]=dfi["p6426_3"]*dfi["age"]

dfi["p7510s7a1_hours"]=dfi["p7510s7a1"]*dfi["hoursWorkUsual"]

#Prediction
#Split the data
np.random.seed(500)
dfi_train, dfi_test = train_test_split(
                                dfi,
                                test_size=.3, 
                                random_state=911)

#Define X matrix
oficio=["oficio_1.0"]
for i in range(2,99):
    oficio.append("oficio_"+str(i)+".0")
    list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
    oficio2=[i for i in oficio if i not in list1]

x1=(dfi_train[["sizeFirm", "hoursWorkUsual", "formal", "p6426", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0", "relab_5.0", "relab_6.0", 
"relab_7.0", "relab_8.0" , "relab_9.0", "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", 
"mes_7.0", "mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]])
x2=dfi_train[oficio2]
X_train=pd.concat([x1,x2], axis=1, join="inner")


x11=(dfi_test[["sizeFirm", "hoursWorkUsual", "formal", "p6426", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0", "relab_5.0", "relab_6.0", 
"relab_7.0", "relab_8.0" , "relab_9.0", "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", 
"mes_7.0", "mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]])
x22=dfi_test[oficio2]
X_test=pd.concat([x11,x22], axis=1, join="inner")

rmsemodels=[0]*10

#Models
#Model 1 earnings age age2
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]
x_train=dfi_train[["age", "age2"]]
x_test=dfi_test[["age", "age2"]]
model1=LinearRegression(fit_intercept=True)
results1= model1.fit(x_train,y_train)
y_hattest=results1.predict(x_test)

rmse1=mean_squared_error(y_test,y_hattest, squared=False)
rmse1
rmsemodels[0]=rmse1
rmse1

#Models  
#Model 2 log_earnings female
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]
x_train=dfi_train[["female"]]
x_test=dfi_test[["female"]]
model2=LinearRegression(fit_intercept=True)
results2= model2.fit(x_train,y_train)
y_hattest=results2.predict(x_test)
rmse2=mean_squared_error(y_test,y_hattest, squared=False)
rmse2
rmsemodels[1]=rmse2

#Models 
#Model 3 log_earnings female age age2 agefemale agefemale
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]
x_train=dfi_train[["female", "age", "age2", "female_age", "female_age2"]]
x_test=dfi_test[["female", "age", "age2", "female_age", "female_age2"]]
model3=LinearRegression(fit_intercept=True)
results3= model3.fit(x_train,y_train)
y_hattest=results3.predict(x_test)
rmse3=mean_squared_error(y_test,y_hattest, squared=False)
rmse3
rmsemodels[2]=rmse3

#Models 
#Model 4 log_earnings female age age2 agefemale age2female X
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]



x_trainsub=dfi_train[["female", "age", "age2", "female_age", "female_age2"]]
x_testsub=dfi_test[["female", "age", "age2", "female_age", "female_age2"]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test,x_testsub], axis=1, join="inner")

model4=LinearRegression(fit_intercept=True)
results4= model4.fit(x_train,y_train)
y_hattest=results4.predict(x_test)
rmse4=mean_squared_error(y_test,y_hattest, squared=False)
rmse4

rmsemodels[3]=rmse4

#Models 
#Model 5 age age2 hoursworkusual hoursworkusual2 agehoursworkusual agehoursworkusual2 X
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]


x_trainsub=dfi_train[["age", "age2", "hours2", "age_hours" , "age_hours2"]]
x_testsub=dfi_test[["age", "age2", "hours2", "age_hours" , "age_hours2"]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test,x_testsub], axis=1, join="inner")

model5=LinearRegression(fit_intercept=True)
results5= model5.fit(x_train,y_train)
y_hattest=results5.predict(x_test)
rmse5=mean_squared_error(y_test,y_hattest, squared=False)
rmse5

rmsemodels[4]=rmse5

#Models 
#Model 6 age age2 female educ relab formal oficio p7090 mes reg_salud estrato 
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

educ=["educ_1.0"]
for i in range(3,7):
    educ.append("educ_"+str(i)+".0")
x_educt=dfi_train[educ]
x_eductt=dfi_test[educ]

relab=["relab_2.0"]
for i in range(3,9):
    relab.append("relab_"+str(i)+".0")
x_relabt=dfi_train[relab]
x_relabtt=dfi_test[relab]
   
regs=["regs_1.0"]
for i in range(2,3):
    regs.append("regs_"+str(i)+".0")
x_regst=dfi_train[regs]
x_regstt=dfi_test[regs]

mes=["mes_1.0"]
for i in range(2,12):
    mes.append("mes_"+str(i)+".0")
x_mest=dfi_train[mes]
x_mestt=dfi_test[mes]

estrato2=["estrato_1.0"]
for i in range(2,6):
    estrato2.append("estrato_"+str(i)+".0")
x_est=dfi_train[estrato2]
x_estt=dfi_test[estrato2]

oficio=["oficio_1.0"]
for i in range(2,99):
    oficio.append("oficio_"+str(i)+".0")
    list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
    oficio2=[i for i in oficio if i not in list1]
x_oficiot=dfi_train[oficio2]
x_oficiott=dfi_test[oficio2]

x_trainsub=dfi_train[["age", "age2", "female" , "p7090" ]]
x_testsub=dfi_test[["age", "age2", "female" , "p7090" ]]

x_train=pd.concat([x_regst, x_est, x_educt, x_relabt, x_mest, x_oficiot,x_trainsub], axis=1, join="inner")
x_test=pd.concat([x_regstt, x_estt, x_eductt, x_relabtt, x_mestt, x_oficiott,x_testsub], axis=1, join="inner")

#Run Model 6
model6=LinearRegression(fit_intercept=True)
results6= model6.fit(x_train,y_train)
y_hattest=results6.predict(x_test)
rmse6=mean_squared_error(y_test,y_hattest, squared=False)
rmse6

rmsemodels[5]=rmse6

#Models 
#Model 7 age age2 female educ femalexeduc p7090 femalexrelab relab X
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

female_edu=["female_educ1"]
for i in range(3,8):
    female_edu.append("female_educ"+str(i))
x_feduct=dfi_train[female_edu]
x_feductt=dfi_test[female_edu]

female_relab=["female_relab1"]
for i in range(2,10):
    female_relab.append("female_relab"+str(i))
x_ferelabt=dfi_train[female_relab]
x_ferelabtt=dfi_test[female_relab]

x_trainsub=dfi_train[["age", "age2", "female" ,"p7090"]]
x_testsub=dfi_test[["age", "age2", "female" ,"p7090"]]

x_train=pd.concat([X_train ,x_trainsub,x_feduct, x_ferelabt], axis=1, join="inner")
x_test=pd.concat([X_test,x_testsub,x_feductt, x_ferelabtt], axis=1, join="inner")

#Run Model 7
model7=LinearRegression(fit_intercept=True)
results7= model7.fit(x_train,y_train)
y_hattest=results7.predict(x_test)
rmse7=mean_squared_error(y_test,y_hattest, squared=False)
rmse7

rmsemodels[6]=rmse7

#Models 
#Model 8 age age2  p7510s7a1 p7090 X p7510s7a1_hours hoursworkusual
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

x_trainsub=dfi_train[["age", "age2", "p7510s7a1" , "p7090", "p7510s7a1_hours" ]]
x_testsub=dfi_test[["age", "age2",  "p7510s7a1" , "p7090", "p7510s7a1_hours" ]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test, x_testsub], axis=1, join="inner")

model8=LinearRegression(fit_intercept=True)
results8= model8.fit(x_train,y_train)
y_hattest=results8.predict(x_test)
rmse8=mean_squared_error(y_test,y_hattest, squared=False)
rmse8

rmsemodels[7]=rmse8

#Models 
#Model 9 p6426 p6426_2 p6426_3 age p7090 age2 p6426Xage p6426_2age p6426_3age  X
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

x_trainsub=dfi_train[["p6426_2" , "p6426_3" , "age" , "age2" , "p6426_age" , "p6426_2_age" , "p6426_3_age"]]
x_testsub=dfi_test[["p6426_2" , "p6426_3" , "age" , "age2" , "p6426_age" , "p6426_2_age" , "p6426_3_age"]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test, x_testsub], axis=1, join="inner")

model9=LinearRegression(fit_intercept=True)
results9= model9.fit(x_train,y_train)
y_hattest=results9.predict(x_test)
rmse9=mean_squared_error(y_test,y_hattest, squared=False)
rmse9

rmsemodels[8]=rmse9

#Models 
#Model 10  agexeduc educ age p7090 age2 X
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

x_trainsub=dfi_train[["age" , "p7090" , "age2"]]
x_testsub=dfi_test[["age" , "p7090" , "age2"]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test, x_testsub], axis=1, join="inner")

model10=LinearRegression(fit_intercept=True)
results10= model10.fit(x_train,y_train)
y_hattest=results10.predict(x_test)
rmse10=mean_squared_error(y_test,y_hattest, squared=False)
rmse10

rmsemodels[9]=rmse10

#Influence statistic
#Model 5 
y_train=dfi_train["log_earnings"]
y_test=dfi_test["log_earnings"]

x_trainsub=dfi_train[["age", "age2", "hours2", "age_hours" , "age_hours2"]]
x_testsub=dfi_test[["age", "age2",  "hours2", "age_hours" , "age_hours2"]]

x_train=pd.concat([X_train,x_trainsub], axis=1, join="inner")
x_test=pd.concat([X_test,x_testsub], axis=1, join="inner")

#compute diagonal of PX
olsmod= sm.OLS(y_test, x_test)
olsmodfit=olsmod.fit()
a=OLSInfluence(olsmodfit)
h=a.hat_matrix_diag

#Residuals in test
olsmod2= sm.OLS(y_train, x_train)
olsmodfit2=olsmod2.fit()
y_hattest2=olsmodfit2.predict(x_test)
residuals2=y_test-y_hattest

residuals2=np.array(residuals2)

#Create the influence for all j in test
influence=[0.0]*4880
influence=np.array(influence)
influence

for j in range(0,4880):
    influence[j]=residuals2[j]/(1-h[j])

plt.xlabel('Valor del estadistico de influencia')
plt.ylabel('Frecuencia')
n, bins, patches = plt.hist(influence , bins=100, range=[0,1] ,facecolor='g', alpha=0.2)
plt.savefig('influencia.png')

#CrossValidation
#Model 5 age age2 hoursworkusual hoursworkusual2 agehoursworkusual agehoursworkusual2 X
#Define Cross-validation variables
y_cv=dfi["log_earnings"]
x_cvsub=dfi[["age", "age2", "hours2", "age_hours" , "age_hours2"]]

oficio=["oficio_1.0"]
for i in range(2,99):
    oficio.append("oficio_"+str(i)+".0")
    list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
    oficio2=[i for i in oficio if i not in list1]

x1=(dfi[["sizeFirm", "hoursWorkUsual", "formal", "p6426", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0", "relab_5.0", "relab_6.0", 
"relab_7.0", "relab_8.0" , "relab_9.0", "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", 
"mes_7.0", "mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]])
x2=dfi[oficio2]
x_cv0=pd.concat([x1,x2], axis=1, join="inner")

x_cv=pd.concat([x_cv0,x_cvsub], axis=1, join="inner")

#CrossValidation of Model 5 ###############################3
#define cross-validation method to use
cv = LeaveOneOut()
#Define cross-validation model
model5=LinearRegression(fit_intercept=True)
#use LOOCV to evaluate model
scores = cross_val_score(model5, x_cv, y_cv, scoring='neg_root_mean_squared_error',
                         cv=cv, n_jobs=-1)

rMSE5 = np.mean(np.absolute(scores))

#CrossValidation Model 7 age age2 female educ femalexeduc p7090 femalexrelab relab X
#Define cross-validation variables
y_cv2=dfi["log_earnings"]

female_edu=["female_educ1"]
for i in range(3,8):
    female_edu.append("female_educ"+str(i))
x_feduct=dfi[female_edu]

female_relab=["female_relab1"]
for i in range(2,10):
    female_relab.append("female_relab"+str(i))
x_ferelabt=dfi[female_relab]

x_cv2sub=dfi[["age", "age2", "female" ,"p7090"]]

oficio=["oficio_1.0"]
for i in range(2,99):
    oficio.append("oficio_"+str(i)+".0")
    list1=['oficio_10.0', 'oficio_22.0', 'oficio_23.0', 'oficio_24.0', 'oficio_25.0', 'oficio_26.0', 'oficio_27.0', 'oficio_28.0', 'oficio_29.0', 'oficio_46.0', 'oficio_47.0', 'oficio_48.0', 'oficio_64.0', 'oficio_65.0', 'oficio_66.0', 'oficio_67.0', 'oficio_68.0', 'oficio_69.0', 'oficio_71.0']
    oficio2=[i for i in oficio if i not in list1]

x1=(dfi[["sizeFirm", "hoursWorkUsual", "formal", "p6426", "educ_3.0", "educ_4.0", 
"educ_5.0", "educ_6.0", "educ_7.0", "relab_2.0", "relab_3.0" , "relab_4.0", "relab_5.0", "relab_6.0", 
"relab_7.0", "relab_8.0" , "relab_9.0", "mes_1.0", "mes_2.0", "mes_3.0", "mes_4.0", "mes_5.0", "mes_6.0", 
"mes_7.0", "mes_9.0", "mes_10.0", "mes_11.0", "mes_12.0"]])
x2=dfi[oficio2]
x_cv02=pd.concat([x1,x2], axis=1, join="inner")

x_cv2=pd.concat([x_cv2sub,x_feduct, x_ferelabt, x_cv02 ], axis=1, join="inner")

#CrossValidation of Model 7 ###############################3
#define cross-validation method to use
cv = LeaveOneOut()
#Define cross-validation model
model7=LinearRegression(fit_intercept=True)
#use LOOCV to evaluate model
scores2 = cross_val_score(model7, x_cv2, y_cv2, scoring='neg_root_mean_squared_error',
                         cv=cv, n_jobs=-1)

#view RMSE
rMSE7 = np.mean(np.absolute(scores2))











