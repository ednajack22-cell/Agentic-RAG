# Git Repository Setup Instructions

> **Save these instructions for when you are ready to publish the repository.**

---

## Option A: GitHub (Public or Private)

### 1. Initialize Git

```bash
cd agentic-rag-replication

git init
git add .
git commit -m "Initial commit: Agentic RAG replication package"
```

### 2. Create the Remote Repository

1. Go to https://github.com/new
2. Repository name: `agentic-rag-replication`
3. Description: "Replication package for: Governing Generative AI for Reliable Decision Support"
4. Choose **Public** (for peer review) or **Private** (share via link)
5. Do NOT initialize with README (we already have one)

### 3. Push

```bash
git remote add origin https://github.com/<your-username>/agentic-rag-replication.git
git branch -M main
git push -u origin main
```

### 4. Create a Release (Optional but recommended)

1. Go to your repo → Releases → Create new release
2. Tag: `v1.0.0`
3. Title: "Manuscript submission v1.0"
4. Attach the `results/` outputs if desired

---

## Option B: OSF (Open Science Framework)

### 1. Create OSF Project

1. Go to https://osf.io/
2. Create a new project: "Agentic RAG Replication Package"
3. Upload the entire `agentic-rag-replication/` folder

### 2. Get DOI

1. In the OSF project, go to Settings → Registration
2. Create a registration to get a permanent DOI
3. Use this DOI in your manuscript: `[Insert GitHub/OSF Archive Link Here]`

---

## Option C: Zenodo (DOI via GitHub)

### 1. Push to GitHub first (Option A above)

### 2. Connect to Zenodo

1. Go to https://zenodo.org/
2. Log in with GitHub
3. Enable the repository in Zenodo settings
4. Create a GitHub release (tag `v1.0.0`)
5. Zenodo will automatically archive it and assign a DOI

### 3. Use the Zenodo DOI in your manuscript

---

## Pre-Flight Checklist

Before pushing, verify:

- [ ] `.env` file is NOT tracked (check `.gitignore`)
- [ ] No API keys in any file (search for `sk-`, `AI`, actual key strings)
- [ ] No `.docx`, `.pdf`, or `.csv` data files included
- [ ] No `__pycache__/` or `.pyc` files
- [ ] No `venv/` or `.venv/` directories
- [ ] `README.md` renders correctly
- [ ] All Python files have proper docstrings

### Quick verification commands:

```bash
# Check for accidental secrets
git grep -i "api_key\s*=" -- "*.py" | grep -v "environ\|example\|your_"

# Check file sizes (nothing huge)
git ls-files -z | xargs -0 ls -la | sort -k5 -n -r | head -20

# Verify .gitignore works
git status
```
