import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pickle

# โหลดข้อมูลจากไฟล์ recruitment_data.csv
data = pd.read_csv('recruitment_data.csv')

# แยกฟีเจอร์และเป้าหมาย
X = data.drop('HiringDecision', axis=1)  # แทนที่ 'HiringDecision' ด้วยชื่อคอลัมน์ที่เป็นเป้าหมาย
y = data['HiringDecision']  # เป้าหมาย

# โหลดโมเดลที่ถูกฝึกแล้วจากไฟล์ model.pkl
with open('best_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# แสดงภาพของ Decision Tree พร้อมข้อมูลเชิงลึก
plt.figure(figsize=(30,15))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=["Do not Hire", "Hire"], 
          filled=True, 
          rounded=True,  # แสดงโหนดแบบมีมุมโค้ง
          fontsize=10,   # ขนาดของข้อความ
          max_depth=3,  # จำกัดความลึกของต้นไม้ที่แสดงผลสูงสุดเป็น 3 ชั้น
        #   impurity=False, # ซ่อนค่า gini
          proportion=True,) # แสดงสัดส่วนของแต่ละคลาสในโหนด
plt.show()
