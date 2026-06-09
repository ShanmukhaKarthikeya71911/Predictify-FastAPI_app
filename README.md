# SalaryIQ — Employment Salary Predictor

## Stack
- **ML**: Scikit-learn GradientBoostingRegressor (R²=0.876, MAE≈₹0.96 LPA)
- **API**: FastAPI + Uvicorn
- **DB**: SQLite via SQLAlchemy
- **Frontend**: Pure HTML/CSS/JS (no framework needed)

## Project Structure
```
salary-predictor/
├── main.py               ← FastAPI app (routes + prediction logic)
├── database/
│   └── db.py             ← SQLAlchemy models + session
├── models/
│   ├── train_model.py    ← Train & save the ML model
│   └── salary_model.pkl  ← Saved model (generated after training)
├── templates/
│   └── index.html        ← Full UI with animations
└── static/               ← CSS/JS assets (currently inline)
```

## Setup
```bash
pip install fastapi uvicorn scikit-learn pandas numpy sqlalchemy jinja2 python-multipart

# Train the model first (only needed once)
python models/train_model.py

# Run the server
uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000

## UI Features (Animations)
| Slide | Animation |
|-------|-----------|
| Hero | Staggered clip-path + translateY fade-up on load |
| Predictor | Scroll-triggered IntersectionObserver reveal |
| Result Card | Count-up number animation + gradient bar fill |
| How It Works | Staggered card reveal with hover lift |
| History | Live table refresh without page reload |
| Cursor | Ambient radial glow follows mouse |

## API Endpoints
- `GET /`         → Main UI
- `POST /predict` → Returns `{salary, low, high}`
- `GET /history`  → Last 10 predictions from DB
- `GET /stats`    → Aggregate stats

## Next Steps (Real Production)
1. Replace synthetic data with Naukri/LinkedIn scraped data
2. Add user authentication (FastAPI + JWT)
3. Add more features: company size, certifications, internships
4. Deploy on Railway / Render / AWS EC2
