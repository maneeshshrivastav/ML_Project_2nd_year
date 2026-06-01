# HOW TO USE PROJECT SETUP ENVIRONMENT

# set your working directory to the ROOT project directory,
# run the following, to set/activate your virtual environment, and get all dependencies:

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Place your 'test.csv' file inside the ROOT project directory.
# Run 'main.py'
# (FOR PROFESSOR) Sir, you can find predictions.txt in the 'test_files' directory
# ------------------------------------------------------------------

# USEFUL GITHUB COMMANDS (for developers, not professor)

# update local repo with remote (DO THIS BEFORE YOU START MAKING CHANGING):
git pull origin main


# stage all local changes (DO THIS AFTER YOU FINISH MAKING CHANGES:
git add .

# save local snapshot of changes:
git commit -m "message"

# upload changes to remote:
git push origin main