# Student Performance AI/ML Prediction Dashboard & API

A full-stack, production-grade Machine Learning web application predicting high school student examination scores (Math, Reading, Writing), academic pass/fail risk status, and providing personalized early interventions based on demographic, socioeconomic, and preparatory factors.

---

## 🚀 Live Local Deployment

The server is currently running in the background as a local service:

- **Dashboard UI**: [http://localhost:5050](http://localhost:5050)
- **Health Check**: [http://localhost:5050/health](http://localhost:5050/health)
- **Model Metrics API**: [http://localhost:5050/api/metrics](http://localhost:5050/api/metrics)
- **EDA Statistics API**: [http://localhost:5050/api/stats](http://localhost:5050/api/stats)

---

## 📡 REST API Documentation

### `POST /api/predict`
Calculates multi-subject forecasts, academic risk tier, and targeted pedagogical interventions.

#### Request Body (JSON)
```json
{
  "gender": "female",
  "race/ethnicity": "group B",
  "parental level of education": "bachelor's degree",
  "lunch": "standard",
  "test preparation course": "completed",
  "reading score": 85,
  "writing score": 88
}
```

#### Response (JSON)
```json
{
  "student_profile": {
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "completed"
  },
  "forecasted_scores": {
    "math_score": 73.9,
    "reading_score": 82.8,
    "writing_score": 85.8,
    "overall_average_score": 80.8,
    "refined_math_score_given_reading_writing": 76.1
  },
  "academic_risk_assessment": {
    "pass_probability": 73.1,
    "pass_status": "Likely to Pass",
    "predicted_tier": "Moderate Performer"
  },
  "actionable_recommendations": [
    "Advanced Placement: Candidate for honors / accelerated academic challenges."
  ]
}
```

---

## 🌐 Cloud Deployment Options

### Option 1: Docker (Any Cloud Provider)
```bash
# Build the container
docker build -t student-perf-predictor .

# Run container locally on port 5050
docker run -d -p 5050:5050 --name student-ai student-perf-predictor
```

### Option 2: Render (Free Web Service)
1. Push this directory to a GitHub repository.
2. In [Render Dashboard](https://dashboard.render.com), click **New + > Web Service**.
3. Select your repository.
4. Set:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -b 0.0.0.0:$PORT app:app`
5. Click **Create Web Service**.

### Option 3: Hugging Face Spaces (Free Cloud Hosting)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with SDK set to **Docker**.
2. Clone your Space repo and copy the project files (`app.py`, `dashboard.html`, `models/`, `visualizations/`, `Dockerfile`, `requirements.txt`).
3. Commit and push: Hugging Face will automatically build and host the interactive dashboard publicly!

### Option 4: Google Cloud Run
```bash
gcloud run deploy student-perf-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
