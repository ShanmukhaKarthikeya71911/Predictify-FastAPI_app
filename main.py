from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pickle, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database.db import get_db, Prediction, User
import numpy as np

app = FastAPI(title="SalaryIQ – Employment Salary Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("models/salary_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]
encoders = bundle["encoders"]
ROLES      = bundle["roles"]
INDUSTRIES = bundle["industries"]
LOCATIONS  = bundle["locations"]
EDUCATIONS = bundle["educations"]

class AuthRequest(BaseModel):
    username: str
    password: str

@app.on_event("startup")
def startup_event():
    db = next(get_db())
    try:
        # Seed default admin account
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            db.add(User(username="admin", password="admin123", role="admin"))
        
        # Seed default user account
        user = db.query(User).filter(User.username == "user").first()
        if not user:
            db.add(User(username="user", password="user123", role="user"))
        
        db.commit()
    except Exception as e:
        print(f"Error seeding database: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "roles": ROLES, "industries": INDUSTRIES,
        "locations": LOCATIONS, "educations": EDUCATIONS
    })

@app.post("/login", response_class=JSONResponse)
async def login(data: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or user.password != data.password:
        return JSONResponse(status_code=400, content={"error": "Invalid username or password"})
    return {"status": "success", "username": user.username, "role": user.role}

@app.post("/register", response_class=JSONResponse)
async def register(data: AuthRequest, db: Session = Depends(get_db)):
    if not data.username.strip() or not data.password.strip():
        return JSONResponse(status_code=400, content={"error": "Username and password cannot be empty"})
    
    existing = db.query(User).filter(User.username == data.username).first()
    if existing:
        return JSONResponse(status_code=400, content={"error": "Username already exists"})
    
    new_user = User(username=data.username, password=data.password, role="user")
    db.add(new_user)
    db.commit()
    return {"status": "success", "username": new_user.username, "role": new_user.role}

@app.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request,
    role: str       = Form(...),
    industry: str   = Form(...),
    location: str   = Form(...),
    education: str  = Form(...),
    experience: int = Form(...),
    num_skills: int = Form(...),
    username: str   = Form(None),
    db: Session     = Depends(get_db)
):
    try:
        feats = [
            encoders["role"].transform([role])[0],
            encoders["industry"].transform([industry])[0],
            encoders["location"].transform([location])[0],
            encoders["education"].transform([education])[0],
            experience,
            num_skills
        ]
        salary = float(model.predict([feats])[0])
        salary = round(max(2.5, salary), 2)

        # ± 15% confidence band
        low  = round(salary * 0.85, 2)
        high = round(salary * 1.15, 2)

        record = Prediction(
            username=username if (username and username.strip()) else None,
            role=role, industry=industry, location=location,
            education=education, experience=experience,
            num_skills=num_skills, predicted_salary=salary
        )
        db.add(record); db.commit()

        return {"salary": salary, "low": low, "high": high, "status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/history", response_class=JSONResponse)
async def history(username: str = None, db: Session = Depends(get_db)):
    query = db.query(Prediction)
    if username:
        query = query.filter(Prediction.username == username)
    rows = query.order_by(Prediction.created_at.desc()).limit(10).all()
    return [{"role": r.role, "location": r.location, "salary": r.predicted_salary,
             "experience": r.experience, "created_at": str(r.created_at)} for r in rows]

@app.get("/admin/history", response_class=JSONResponse)
async def admin_history(db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(30).all()
    return [{"role": r.role, "location": r.location, "salary": r.predicted_salary,
             "experience": r.experience, "created_at": str(r.created_at), "username": r.username or "guest"} for r in rows]

@app.post("/admin/clear-logs", response_class=JSONResponse)
async def clear_logs(db: Session = Depends(get_db)):
    try:
        db.query(Prediction).delete()
        db.commit()
        return {"status": "success", "message": "All predictions cleared successfully."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/stats", response_class=JSONResponse)
async def stats(db: Session = Depends(get_db)):
    rows = db.query(Prediction).all()
    if not rows:
        return {"total": 0, "avg": 0, "max": 0, "min": 0}
    salaries = [r.predicted_salary for r in rows]
    return {
        "total": len(rows),
        "avg":   round(sum(salaries) / len(salaries), 2),
        "max":   round(max(salaries), 2),
        "min":   round(min(salaries), 2)
    }

