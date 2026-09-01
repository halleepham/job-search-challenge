# CS 5542 Challenge 1 - Agentic AI Job Search & Profile Matching Application

**Author:** Hallee Pham  
**Branch:** `agent-exercise`

---

## 🎯 Project Overview

This application is an **Agentic AI Job Search and Profile Matching Application** built with **Streamlit**. It takes candidate information (skills, experience, target domain, education level, work preferences, and bio/resume summary) and dynamically compares it against a comprehensive catalog of job postings across multiple industries.

The engine computes an explainable, multi-factor **Match Score (0–100%)** for each job listing, ranks the listings from strongest to lowest fit, and provides detailed analytical breakdowns including matched skills, skill gaps to bridge, domain alignment, and side-by-side job comparisons.

---

## 🌟 Key Features

1. **Multi-Factor Ranking & Scoring Engine**:
   - **Technical Skill Match (40% Weight)**: Normalizes skill aliases (e.g., `React.js` ↔ `React`, `K8s` ↔ `Kubernetes`, `Node` ↔ `Node.js`) and calculates overlap against required and preferred skills.
   - **Semantic / Text Similarity (30% Weight)**: Uses **TF-IDF Vectorization** and **Cosine Similarity** between candidate profile text/headline and job descriptions/responsibilities.
   - **Experience Alignment (15% Weight)**: Evaluates candidate years of experience against job seniority requirements with optimal fit scoring and gap penalties.
   - **Domain & Preference Compatibility (15% Weight)**: Scores target industry match and work mode preferences (Remote, Hybrid, On-site).

2. **1-Click Candidate Profile Presets**:
   - Includes realistic presets for immediate testing:
     - *Full-Stack Web Developer (Mid-Level)*
     - *Machine Learning & AI Specialist (Senior)*
     - *UI/UX Product Designer (Mid-Level)*
     - *Cloud & DevOps Infrastructure Engineer (Senior)*
     - *Entry-Level Junior Software Developer*
     - *Cybersecurity Analyst (Mid-Level)*
   - Or fill in any custom profile from scratch.

3. **Interactive Visual Dashboard**:
   - **Ranked Job Cards**: Color-coded match tiers (*Strong*, *Good*, *Moderate*, *Low*), salary badges, experience tags, and expandable deep-dive score breakdowns.
   - **Skill Gap Analysis**: Highlights matched skills in green and missing required skills in red/orange to help candidates identify what to learn.
   - **Interactive Plotly Visualizations**:
     - *Domain Alignment Bar Chart*: Compares average and peak match scores across industries.
     - *Fit Radar Chart*: Visualizes multidimensional competency fit for top jobs.
   - **Side-by-Side Job Comparison**: Compare up to 3 jobs at once with score breakdowns and skill gaps.
   - **Job Catalog & Management**: Search and filter the full database, or dynamically add custom job postings.

---

## 📁 Project Structure

```
job-search-challenge/
├── app.py                     # Main Streamlit web application entrypoint
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── .gitignore                 # Git ignore rules
├── data/
│   └── jobs.json              # Curated job listings dataset across tech fields
├── src/
│   ├── __init__.py            # Package indicator
│   ├── models.py              # Data models (UserProfile, JobListing, JobMatchResult)
│   ├── matcher.py             # Multi-factor scoring, TF-IDF text matching & ranking
│   ├── data_loader.py         # Dataset loader, preset profiles & schema handlers
│   └── ui_components.py       # Custom card styles, badges, and Plotly charts
└── tests/
    ├── __init__.py
    └── test_matcher.py        # Comprehensive unit test suite (12 test cases)
```

---

## 🚀 Setup & Running Instructions

### 1. Prerequisites
- Python 3.10+ installed

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the automated test suite with `pytest`:
```bash
pytest -v
```
All 12 test cases cover skill normalization, TF-IDF text similarity, experience scoring curves, preference calculations, ranking order, dataset integrity, and edge cases.
