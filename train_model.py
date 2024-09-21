import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

# โหลดข้อมูลจากไฟล์ recruitment_data.csv
data = pd.read_csv('recruitment_data.csv')

# แยกฟีเจอร์และเป้าหมาย
X = data.drop('HiringDecision', axis=1)  # แทนที่ 'HiringDecision' ด้วยชื่อคอลัมน์ที่เป็นเป้าหมาย
y = data['HiringDecision']  # เป้าหมาย

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# กำหนดช่วงของ max_depth ที่ต้องการทดสอบ
param_grid = {'max_depth': range(1, 15)}  # ทดสอบตั้งแต่ความลึก 1 ถึง 15

# ใช้ GridSearchCV เพื่อค้นหาค่าที่ดีที่สุด
clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')  # ใช้การ cross-validation 5 fold
grid_search.fit(X_train, y_train)

# แสดงผลลัพธ์ที่ดีที่สุด
print(f"ค่าความลึกที่ดีที่สุดคือ: {grid_search.best_params_['max_depth']}")
print(f"ความแม่นยำที่ดีที่สุด: {grid_search.best_score_:.4f}")

# ฝึกโมเดลอีกครั้งด้วยค่าความลึกที่ดีที่สุด
best_depth = grid_search.best_params_['max_depth']
clf_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf_best.fit(X_train, y_train)

# บันทึกโมเดลที่ดีที่สุดลงไฟล์ใหม่ (best_model.pkl)
with open('best_model.pkl', 'wb') as file:
    pickle.dump(clf_best, file)

# ทำนายผลลัพธ์และแสดงความแม่นยำของโมเดลที่ดีที่สุดบนข้อมูลทดสอบ
accuracy_test = clf_best.score(X_test, y_test)
print(f"ความแม่นยำของโมเดลที่ดีที่สุดบนข้อมูลทดสอบ: {accuracy_test:.4f}")
