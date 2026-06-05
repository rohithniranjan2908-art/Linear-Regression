import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_dataset.csv")
#print(df.head())

x=df[["YearsExperience"]]
y=df["Salary"]

model=LinearRegression()
model.fit(x,y)

#print("theta0=",model.intercept_)
#print("theta1=",model.coef_[0])

new_data=pd.DataFrame({
    "YearsExperience":[6]
})
prediction=model.predict(new_data)
print("Predicted salary:",prediction)

plt.scatter(x,y)
plt.plot(x,model.predict(x),color="red")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()


