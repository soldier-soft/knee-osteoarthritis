<#
.SYNOPSIS
    Automates the process of initializing a Git repository and pushing to GitHub.
    
.DESCRIPTION
    This script will initialize a local Git repository, stage all files, 
    ask for a commit message, ask for your GitHub repository URL, and push the code.
#>

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Knee Osteoarthritis - GitHub Uploader     " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Found $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Git is not installed or not in your PATH. Please install Git first." -ForegroundColor Red
    exit 1
}

# 1. Initialize Repository
if (-not (Test-Path -Path ".git")) {
    Write-Host "Initializing new Git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "Git repository already initialized." -ForegroundColor Green
}

# 2. Add .gitignore if it doesn't exist
if (-not (Test-Path -Path ".gitignore")) {
    Write-Host "Creating default .gitignore file..." -ForegroundColor Yellow
    @"
__pycache__/
*.py[cod]
*$py.class
venv/
env/
.env
static/tests/
static/uploads/
.DS_Store
*.db
*.sqlite3
"@ | Out-File -FilePath .gitignore -Encoding utf8
}

# 3. Stage Files
Write-Host "Staging files for commit..." -ForegroundColor Yellow
git add .

# 4. Commit Files
$commitMsg = Read-Host "Enter a commit message (or press Enter for default 'Initial commit')"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Initial commit"
}
git commit -m "$commitMsg"

# 5. Connect to GitHub and Push
Write-Host ""
Write-Host "To upload this project, you need an empty repository on GitHub." -ForegroundColor Cyan
Write-Host "1. Go to https://github.com/new and create a new repository." -ForegroundColor Cyan
Write-Host "2. Copy the URL (e.g., https://github.com/yourusername/knee-osteoarthritis.git)" -ForegroundColor Cyan
$repoUrl = Read-Host "Paste your GitHub Repository URL here (or press Enter to skip uploading for now)"

if (-not [string]::IsNullOrWhiteSpace($repoUrl)) {
    # Check if remote exists, update or add it
    $remotes = git remote
    if ($remotes -contains "origin") {
        Write-Host "Updating existing remote 'origin'..." -ForegroundColor Yellow
        git remote set-url origin $repoUrl
    } else {
        Write-Host "Adding remote 'origin'..." -ForegroundColor Yellow
        git remote add origin $repoUrl
    }

    Write-Host "Renaming branch to 'main'..." -ForegroundColor Yellow
    git branch -M main

    Write-Host "Pushing code to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=============================================" -ForegroundColor Green
        Write-Host " Successfully uploaded to GitHub!            " -ForegroundColor Green
        Write-Host " $repoUrl                                    " -ForegroundColor Green
        Write-Host "=============================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Error pushing to GitHub. Check the output above." -ForegroundColor Red
    }
} else {
    Write-Host "Skipped pushing to GitHub. Your code is safely committed locally." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
