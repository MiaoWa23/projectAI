from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle

app = Flask(__name__)

# โหลดโมเดลที่ฝึกมาแล้ว
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากผู้ใช้ในรูปแบบ JSON
    data = request.json
    
    # ดึงข้อมูลฟีเจอร์จาก JSON (แทนที่ด้วยชื่อฟีเจอร์ที่ใช้ในโมเดล)
    features = [data['Age'], data['Gender'], data['EducationLevel'], data['ExperienceYears'], data['PreviousCompanies'], data['DistanceFromCompany'], data['InterviewScore'], data['SkillScore'], data['PersonalityScore'], data['RecruitmentStrategy']]  
    
    # ใช้โมเดลทำนายผล
    prediction = model.predict([features])
    
    # ส่งคำตอบกลับในรูปแบบ JSON
    return jsonify({'hire': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
