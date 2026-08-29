# Job Application Strategy AI

> **Don't just find jobs. Know which jobs to apply to first — and why.**

Job Application Strategy AI is an intent-aware job recommendation system that helps students and job seekers make **better decisions about where to apply**.

Instead of simply comparing a resume with job descriptions and returning a similarity score, the system combines:

* the candidate's **resume**
* the candidate's **actual career goal**
* job requirements
* skills
* experience
* education
* location
* semantic similarity
* and job/application signals

to create a **ranked application strategy**.

The goal is simple:

> **Help candidates apply strategically, not randomly.**

---

## Why I Built This

Many students know **how to apply for jobs**, but they often do not know:

* Which jobs should I prioritize?
* Is this job actually aligned with the career I want?
* Should I apply now or improve my resume first?
* Which skills am I missing?
* Why is one job a better opportunity than another?
* If a company is hiring for immediate joining, should I prioritize that opportunity?

Most basic resume-job matching systems answer only:

> **"How similar is your resume to this job description?"**

But **resume similarity is not the same as career intent**.

For example:

A student may currently have a Data Analyst resume but want to move into **Generative AI**.

A traditional similarity system may recommend another Data Analyst job because the resume contains Python, SQL, statistics and data-analysis keywords.

Our system asks a different question:

> **"What does this candidate want to do next?"**

That career intent becomes an important part of the recommendation strategy.

---

# What Makes This Different?

### Traditional approach

```text
Resume
   ↓
Compare with Job Descriptions
   ↓
Similarity Score
   ↓
Recommended Jobs
```

### Job Application Strategy AI

```text
Resume
   +
Career Goal
   ↓
Understand Candidate
   ↓
Understand Jobs
   ↓
Match Skills + Experience + Education + Location
   ↓
Check Career Intent
   ↓
Strategic Ranking
   ↓
Why Apply / Why Not Apply
   ↓
What Should I Do Next?
   ↓
Apply Now / Continue to LinkedIn
```

### The key difference

**Most systems answer:**

> "Which jobs look similar to your resume?"

**This system answers:**

> "Which jobs should you prioritize based on where you want your career to go?"

That is the main idea behind the project.

---

# Who Is This For?

### For Students

You don't need to understand machine learning to use the system.

You simply:

1. Upload your resume.
2. Tell the system what role you want.
3. Review the recommended jobs.
4. Understand why each job is recommended.
5. See what skills may be missing.
6. Prioritize the right opportunities.
7. Use **Apply Now** when an immediate-action opportunity is available.
8. Get redirected to the relevant LinkedIn/application page when applicable.

### For Recruiters / Technical Reviewers

The project demonstrates:

* NLP-based career intent understanding
* resume information extraction
* semantic search
* embeddings
* vector search
* hybrid recommendation
* deterministic ranking
* skill-gap analysis
* RAG-based explanations
* evaluation using Precision@K, Recall@K and NDCG@K
* FastAPI backend
* Next.js frontend
* MongoDB and Redis integration
* pluggable embedding and LLM providers
* production-oriented service design

---

# How the User Journey Works

## Step 1 — Upload Resume

The candidate uploads a resume.

The system extracts structured information such as:

```text
Skills
Experience
Education
Location
Projects / relevant resume information
```

The resume is then used as a personalized input for recommendation.

---

## Step 2 — Tell Us What You Want

The candidate can enter a role or a natural-language goal.

Examples:

```text
GenAI Engineer
```

or

```text
I want to move into Data Analytics
```

or

```text
I want to work with Large Language Models and RAG
```

The system converts this free-text goal into a structured **Career Intent Profile**.

```text
User's career goal
        ↓
Intent Normalization
        ↓
Role Family
        ↓
Canonical Role
        ↓
Related Roles
        ↓
Relevant Skills & Keywords
```

This means the system does not depend only on what the candidate has already done.

It also considers **where the candidate wants to go**.

---

# Step 3 — Understand the Job

Each job is analyzed for relevant information:

```text
Job Title
Required Skills
Experience
Education
Location
Job Description
Role / Domain
```

The system uses a skill taxonomy and job-title/skill relationships to make job information easier to compare with the candidate.

---

# Step 4 — Match the Candidate With Jobs

The recommendation engine combines multiple signals.

```text
                  ┌─────────────────────┐
                  │      Resume         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Candidate Profile   │
                  └──────────┬──────────┘
                             │
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
       Career Intent     Job Requirements   Semantic Fit
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Strategy Ranking    │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Ranked Opportunities│
                  └─────────────────────┘
```

The ranking considers:

| Signal              | What it means                                     |
| ------------------- | ------------------------------------------------- |
| Career Intent       | Does this job match what the candidate wants?     |
| Semantic Similarity | How similar are the resume and job description?   |
| Skill Match         | How many required skills does the candidate have? |
| Experience Match    | Does the experience level fit?                    |
| Education Match     | Does the education requirement fit?               |
| Location Match      | Does the location preference fit?                 |

---

# Step 5 — Create an Application Strategy

The system does not simply return a list.

It assigns an action based on the overall recommendation.

```text
Strong match + strong skill coverage
              ↓
          APPLY NOW

Good match + small skill gaps
              ↓
      TAILOR RESUME FIRST

Potential match + larger gaps
              ↓
    BUILD MISSING EVIDENCE

Poor / off-intent match
              ↓
        LOW PRIORITY
```

This turns job recommendation into an **application strategy**.

---

# Step 6 — Explain the Recommendation

Every recommendation should answer a simple question:

> **Why should I apply?**

The system provides:

### Why Apply

Examples:

* Your resume contains relevant skills required by this role.
* Your experience matches the expected level.
* The role aligns with your stated career goal.

### Why Not Apply

Examples:

* The role does not align strongly with your stated career intent.
* Some required skills are missing.
* The location does not match your preference.

This explanation is generated from retrieved evidence rather than allowing the language model to freely invent reasons.

---

# Step 7 — Take Action

The final stage is about reducing the gap between **recommendation and action**.

If a job represents an immediate-action opportunity, such as an **immediate joining requirement**, the interface can surface an **Apply Now** action.

The purpose of this button is **not application tracking**.

It is there to help the candidate identify:

> **"This is an opportunity where acting now may make sense."**

The candidate can then be redirected to the relevant application destination, including LinkedIn where applicable.

### LinkedIn Integration

The system does **not require access to the candidate's LinkedIn account or LinkedIn credentials**.

LinkedIn acts as a destination for the candidate to continue the application process.

```text
Recommended Job
      ↓
   Apply Now
      ↓
Relevant Application / LinkedIn
      ↓
Candidate continues application
```

The project therefore focuses on **recommendation and strategy**, rather than trying to replace the actual job platform.

---

# Complete System Flow

```text
                         USER
                          │
                          ▼
                  ┌───────────────┐
                  │ Upload Resume │
                  └───────┬───────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Resume Extraction │
                └─────────┬─────────┘
                          │
                          │
                          ▼
                ┌───────────────────┐
                │ Career Goal Input │
                │ "What do you want │
                │  to become?"      │
                └─────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Career Intent Layer     │
              │                         │
              │ NLP + taxonomy +        │
              │ fuzzy / keyword /       │
              │ embedding matching      │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Job Intelligence Layer  │
              │                         │
              │ Skills                  │
              │ Experience              │
              │ Education               │
              │ Location                │
              │ Job requirements        │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Strategy Ranking Layer  │
              │                         │
              │ Intent                  │
              │ Semantic similarity     │
              │ Skill match             │
              │ Experience              │
              │ Education               │
              │ Location                │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Ranked Jobs      │
                 └────────┬─────────┘
                          │
              ┌───────────┼────────────┐
              │           │            │
              ▼           ▼            ▼
        Why Apply    Why Not Apply   Skill Gaps
              │           │            │
              └───────────┼────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Action Strategy │
                 └────────┬────────┘
                          │
            ┌─────────────┼──────────────┐
            │             │              │
            ▼             ▼              ▼
        Apply Now    Tailor Resume   Build Skills
            │
            ▼
    LinkedIn / Application
```

---

# What Happens Inside the System?

The system is divided into three main layers.

```text
┌────────────────────────────────────────────────────┐
│                  LAYER 1                           │
│              CAREER INTENT                         │
│                                                    │
│  "I want to become a GenAI Engineer"              │
│                    ↓                               │
│          Role + Domain + Skills                    │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│                  LAYER 2                           │
│             JOB INTELLIGENCE                       │
│                                                    │
│   Job → Skills + Experience + Education +          │
│          Location + Role information               │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│                  LAYER 3                           │
│            STRATEGY RANKING                        │
│                                                    │
│ Resume + Intent + Job Requirements                 │
│                    ↓                               │
│             Final Recommendation                   │
│                    ↓                               │
│        Apply / Tailor / Build / Low Priority       │
└────────────────────────────────────────────────────┘
```

---

# Layer 1 — Career Intent

A resume describes what a candidate has done.

It does not always describe what the candidate wants to do next.

That is why career intent is stored separately from the resume.

For example:

```text
Resume:
Data Analyst

Career Intent:
GenAI Engineer
```

The system should not automatically assume:

```text
Data Analyst resume = Data Analyst career goal
```

Instead, the user's stated intent is normalized into a role family.

### Intent normalization

The system uses a four-stage cascade:

| Stage | Method               | Example                                        |
| ----- | -------------------- | ---------------------------------------------- |
| 1     | Exact alias          | `genai engineer` → `genai_engineer`            |
| 2     | Fuzzy matching       | `Data Analist` → `data_analyst`                |
| 3     | Keyword overlap      | `I want to work with LLMs and RAG` → GenAI     |
| 4     | Embedding similarity | Handles intent with weaker taxonomy vocabulary |

This layer is deliberately deterministic and inexpensive for normal classification.

---

# Layer 2 — Job Intelligence

Two datasets are used to support this layer.

## Dataset 1 — Job Skill Set

**Source:** Kaggle — Job Skill Set

https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set

### Used for:

* expanding the skill taxonomy
* connecting job titles with relevant skills
* enriching role intent with job-related skills

Expected columns:

```text
job_title
skill
```

The project includes a documented adapter and build script for integrating the real dataset.

---

## Dataset 2 — Resume Data for Ranking

**Source:** Kaggle — Resume Data for Ranking

https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking

### Used for:

* additional resume/job relevance cases
* evaluation dataset expansion

Expected columns:

```text
resume_text
job_title
job_description
match_label
```

The dataset is intentionally kept separate from the hand-labeled evaluation data because the two sources represent different definitions of relevance.

---

# Layer 3 — Strategy Ranking

The final recommendation combines multiple signals:

```text
Final Score =
    Intent Alignment
  + Semantic Similarity
  + Skill Match
  + Experience Match
  + Education Match
  + Location Match
```

Default weights:

| Component           | Weight |
| ------------------- | -----: |
| Intent Alignment    |   0.20 |
| Semantic Similarity |   0.25 |
| Hard Skill Match    |   0.25 |
| Experience Match    |   0.12 |
| Education Match     |   0.08 |
| Location Match      |   0.10 |

The weights are configurable.

---

# Intent Gating

Intent is more than another score.

If a candidate clearly wants a particular career direction, a job that is strongly unrelated to that direction should not become a top recommendation just because the resume contains similar keywords.

For example:

```text
Resume → Data Analyst
Intent → GenAI Engineer

Job → Data Analyst
```

A similarity-only system may rank this highly.

The intent-aware system can recognize:

```text
Strong resume similarity
        +
Poor career-intent alignment
        ↓
Lower priority
```

This is the mechanism that makes the system **strategy-aware rather than similarity-only**.

---

# RAG — Why Apply / Why Not Apply

The recommendation engine first calculates the score.

The explanation layer does **not** decide the score.

Instead:

```text
Ranking
   ↓
Score + Skill Gaps + Intent
   ↓
Retrieve Supporting Evidence
   ↓
LLM / Explanation Layer
   ↓
Groundedness Filter
   ↓
Why Apply / Why Not Apply
```

This separation is intentional.

The LLM explains an already-computed recommendation instead of controlling the ranking itself.

That gives us:

* reproducible ranking
* easier testing
* lower dependency on LLM availability
* less chance of unsupported explanations

---

# Evaluation

The system is evaluated using:

* Precision@K
* Recall@K
* NDCG@K
* average latency

The project compares:

```text
Keyword Search
       vs.
Vector Search
       vs.
Hybrid Search
       vs.
Hybrid + Career Intent
```

## Hybrid Search

On the current labeled evaluation set:

| Mode    |       P@3 |       R@3 |    NDCG@3 |
| ------- | --------: | --------: | --------: |
| Keyword |     0.429 |     0.738 |     0.716 |
| Vector  |     0.381 |     0.762 |     0.814 |
| Hybrid  | **0.524** | **0.929** | **0.880** |

Hybrid ranking performs better for the task of identifying the small set of jobs a candidate should look at first.

---

## Impact of Career Intent

The most important comparison is:

| Variant         |       P@3 |       R@3 |    NDCG@3 |
| --------------- | --------: | --------: | --------: |
| Without Intent  |     0.429 |     0.738 |     0.760 |
| **With Intent** | **0.524** | **0.929** | **0.881** |

On this evaluation set:

```text
NDCG@3
0.76 → 0.88

Recall@3
0.74 → 0.93
```

This shows why explicitly understanding **what the candidate wants** can improve ranking compared with relying only on resume/job similarity.

These results come from a small, manually curated evaluation set, so they should be treated as a project-level evaluation rather than a universal benchmark.

---

# Example: Why Intent Matters

Consider a candidate with:

```text
Current profile:
Data Analyst

Career goal:
GenAI Engineer
```

Without intent:

```text
Resume
   ↓
Similarity Search
   ↓
Data Analyst Job
   ↓
High Score
```

With intent:

```text
Resume
   +
GenAI Engineer Intent
   ↓
Hybrid Ranking
   ↓
Intent Alignment
   ↓
GenAI / AI-related Opportunity
   ↓
Higher Priority
```

This is the core problem the project is designed to solve.

---

# Technology Stack

## Frontend

* Next.js
* React
* TypeScript

## Backend

* Python
* FastAPI

## Data & Infrastructure

* MongoDB
* Redis
* Docker

## AI / ML

* NLP
* Resume information extraction
* Embeddings
* Vector search
* Hybrid ranking
* RAG
* LLM / local explanation provider
* Role and skill taxonomies

## Evaluation

* Precision@K
* Recall@K
* NDCG@K
* latency measurement
* unit and integration tests

---

# Why the LLM Does Not Control Everything

One of the design decisions in this project is to **separate recommendation logic from language generation**.

```text
              ┌──────────────────────┐
              │ Recommendation Engine│
              │                      │
              │ Deterministic        │
              │ Ranking + Gating     │
              └──────────┬───────────┘
                         │
                    Score / Gaps
                         │
                         ▼
              ┌──────────────────────┐
              │ Explanation Layer    │
              │                      │
              │ LLM / Local Provider │
              └──────────┬───────────┘
                         │
                         ▼
                 Human-readable
                  explanation
```

The LLM explains the recommendation.

It does not secretly change the recommendation.

This makes the system easier to evaluate and debug.

---

# Privacy & LinkedIn

The project is designed so that the recommendation system does not need access to a candidate's LinkedIn account.

The LinkedIn flow is intended as:

```text
Recommendation
      ↓
Apply Now
      ↓
LinkedIn / Application Page
      ↓
Candidate applies directly
```

The system does not need the candidate's LinkedIn password or credentials.

### Important implementation note

In the current demo architecture, resume and intent information can be persisted for the recommendation workflow. Authentication, encryption at rest and a complete production data-retention/deletion policy are **future production requirements**, not claims of the current demo.

---

# Current Limitations

No project should claim to solve everything.

Current limitations include:

### 1. Small evaluation dataset

The current hand-labeled evaluation set contains:

```text
7 candidates × 12 sample jobs
```

It demonstrates the methodology but is not large enough for strong statistical conclusions.

### 2. Intent gating can be too strict

A differently titled but genuinely relevant job can sometimes be deprioritized.

For example:

```text
Intent:
NLP Research Scientist

Job:
AI Engineer - Generative AI Platform
```

The two may be related even though their titles are different.

This is a known trade-off.

### 3. Intent can be ambiguous

For example:

```text
"Analyst"
```

could mean:

```text
Data Analyst
Business Analyst
Financial Analyst
...
```

The intent preview mechanism helps reduce this problem by allowing the user to confirm the normalized role.

### 4. Dataset integration

The project includes adapters and scripts for the Kaggle datasets, while the demo can run using synthetic fallback data.

The real datasets are intentionally not redistributed inside the repository.

### 5. Rule-based action thresholds

The current actions such as `apply_now` and `tailor_resume_first` use configurable thresholds.

A future version could learn these decisions from larger-scale user feedback.

---

# Trade-offs & Design Decisions

| Decision                            | Why                                                                 |
| ----------------------------------- | ------------------------------------------------------------------- |
| Career intent separate from resume  | A candidate's next goal can change without changing their resume    |
| Intent normalization without an LLM | Fast, deterministic and easy to test                                |
| Intent gating                       | Prevents generic resume similarity from dominating career direction |
| Hybrid ranking                      | Combines keyword precision with semantic similarity                 |
| LLM separated from ranking          | Keeps ranking reproducible                                          |
| Grounded RAG                        | Reduces unsupported explanations                                    |
| Synthetic fallback                  | Allows the complete system to run without external infrastructure   |
| Configurable providers              | Makes the architecture easier to move toward production             |

---

# API Overview

Base path:

```text
/api/v1
```

Important endpoints:

| Method | Endpoint                          | Purpose                              |
| ------ | --------------------------------- | ------------------------------------ |
| POST   | `/candidates/resume`              | Upload and process resume            |
| GET    | `/candidates/{id}/resume`         | Fetch parsed resume                  |
| POST   | `/intent`                         | Save career intent                   |
| GET    | `/intent/preview?text=`           | Preview intent normalization         |
| GET    | `/intent/{candidate_id}`          | Get current intent                   |
| POST   | `/jobs`                           | Add a job                            |
| POST   | `/jobs/bulk`                      | Add multiple jobs                    |
| POST   | `/jobs/reindex`                   | Rebuild job index                    |
| POST   | `/recommendations/rank`           | Rank jobs                            |
| POST   | `/recommendations/compare`        | Compare ranking approaches           |
| GET    | `/recommendations/{candidate_id}` | Get recommendations                  |
| POST   | `/feedback`                       | Record recommendation feedback       |
| GET    | `/evaluation/run`                 | Run evaluation                       |
| GET    | `/evaluation/intent-impact`       | Compare intent vs. no-intent ranking |

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

# Project Structure

```text
job-application-strategy-ai/
│
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── intent.py
│   │   │   ├── dataset_loaders.py
│   │   │   ├── ranking.py
│   │   │   ├── rag.py
│   │   │   ├── recommender.py
│   │   │   └── ...
│   │   │
│   │   ├── api/
│   │   │   └── routers/
│   │   │
│   │   ├── models/
│   │   └── repositories/
│   │
│   ├── data/
│   │   ├── role_taxonomy.json
│   │   ├── role_title_skills.json
│   │   ├── skill_taxonomy.json
│   │   ├── eval_labels.json
│   │   └── external/
│   │
│   ├── scripts/
│   └── tests/
│
├── frontend/
│   ├── components/
│   ├── app/
│   └── ...
│
├── evaluation/
│   ├── run_eval.py
│   └── results.md
│
├── postman/
│   └── JobApplicationStrategyAI.postman_collection.json
│
├── docker-compose.yml
└── README.md
```

---

# Quick Start

## Option 1 — Demo Mode

No Docker or external database is required.

### Backend

```bash
cd backend

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Frontend

In another terminal:

```bash
cd frontend

npm install

npm run dev
```

Open:

```text
http://localhost:3000
```

Then:

```text
Dashboard
   ↓
Enter Career Goal
   ↓
View Recommendations
```

---

# Run Tests

```bash
cd backend
pytest -v
```

The current test suite covers areas including:

* resume extraction
* intent normalization
* intent gating
* ranking
* evaluation metrics
* API integration

---

# Run Evaluation

With the backend running:

```bash
python evaluation/run_eval.py --base-url http://localhost:8000
```

You can also check intent impact:

```bash
curl "http://localhost:8000/api/v1/evaluation/intent-impact"
```

---

# Production Direction

The architecture is designed so that the local/demo implementation can be replaced with production services without rewriting the entire application.

Potential production mapping:

```text
Frontend
   ↓
Cloud Run

Backend
   ↓
Cloud Run

Redis
   ↓
Memorystore

Database
   ↓
MongoDB Atlas

Vector Search
   ↓
Vertex AI Vector Search / Atlas Vector Search

Embeddings
   ↓
Vertex AI / production embedding provider

LLM Explanation
   ↓
Production LLM provider

Secrets
   ↓
Secret Manager

CI/CD
   ↓
Cloud Build
```

---

# What I Would Build Next

The next improvements would focus on making the recommendation strategy more data-driven.

### 1. Larger evaluation set

Use more real resume/job relevance examples to improve statistical confidence.

### 2. Better intent understanding

Add an LLM-based fallback for genuinely open-ended career goals that the current taxonomy cannot resolve.

### 3. Learning from feedback

Use candidate actions such as:

```text
Accepted recommendation
Rejected recommendation
Applied
Saved
Ignored
```

to improve ranking over time.

### 4. Smarter application strategy

Move from fixed thresholds toward a learned re-ranking model based on accumulated feedback.

### 5. Better job freshness

Connect the system to live job sources so that opportunities can be prioritized using current availability and application timing.

---

# What This Project Demonstrates

This project is not just a resume parser and it is not just a chatbot.

It combines:

```text
NLP
 +
Resume Understanding
 +
Career Intent
 +
Semantic Search
 +
Vector Search
 +
Hybrid Recommendation
 +
Deterministic Ranking
 +
Skill Gap Analysis
 +
RAG
 +
LLM Explanations
 +
Evaluation
 +
Full-Stack Engineering
```

The important idea is the connection between these components.

```text
                  USER
                   │
                   ▼
                RESUME
                   │
                   ▼
            CAREER INTENT
                   │
                   ▼
            JOB INTELLIGENCE
                   │
                   ▼
          STRATEGY-BASED RANKING
                   │
                   ▼
       ┌───────────┴────────────┐
       │                        │
       ▼                        ▼
 WHY APPLY?               WHY NOT APPLY?
       │                        │
       └───────────┬────────────┘
                   ▼
             ACTION PLAN
                   │
                   ▼
             APPLY NOW
                   │
                   ▼
           LINKEDIN / JOB SITE
```

---

# The Core Idea

> **A resume tells us where a candidate has been.
> Career intent tells us where the candidate wants to go.
> Job Application Strategy AI uses both to decide where the candidate should focus next.**

That is the difference between **job matching** and **job application strategy**.

---

# Resume / Portfolio Highlights

The project demonstrates the following measurable engineering work:

* Built a Career Intent Layer using a four-stage resolution cascade: exact, fuzzy, keyword and embedding matching.
* Added intent-aware ranking and gating to prevent unrelated jobs from being recommended solely because of resume similarity.
* Improved **NDCG@3 from 0.76 to 0.88** and **Recall@3 from 0.74 to 0.93** on the current labeled evaluation set when career intent was applied.
* Built a hybrid recommendation system combining semantic similarity, skills, experience, education, location and career intent.
* Added grounded **Why Apply / Why Not Apply** explanations using an evidence-based RAG pipeline.
* Built dataset adapters for external job-skill and resume-ranking datasets.
* Designed the recommendation engine so ranking remains deterministic and independent from LLM availability.
* Built a full-stack system using FastAPI, Next.js, MongoDB, Redis, embeddings, vector search and evaluation tooling.

---

## Final Takeaway

**The problem is not that students cannot find jobs.**

The problem is that many students do not know **which opportunities deserve their attention first**.

Job Application Strategy AI is designed to make that decision easier:

```text
Understand Me
     ↓
Understand What I Want
     ↓
Understand the Job
     ↓
Rank the Opportunity
     ↓
Explain the Reason
     ↓
Tell Me What To Do
     ↓
Help Me Take Action
```

**From searching for jobs → to applying with a strategy.**
