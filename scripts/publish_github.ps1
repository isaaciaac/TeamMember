$ErrorActionPreference = "Stop"

function Require-Env([string]$Name) {
  $v = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($v)) {
    throw "Missing env var: $Name"
  }
  return $v.Trim()
}

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git not found"
}

$token = Require-Env "GITHUB_TOKEN"
$owner = [Environment]::GetEnvironmentVariable("GITHUB_OWNER")
if ([string]::IsNullOrWhiteSpace($owner)) {
  $owner = Require-Env "GITHUB_USER"
}
$repo = [Environment]::GetEnvironmentVariable("GITHUB_REPO")
if ([string]::IsNullOrWhiteSpace($repo)) {
  $repo = "teammember"
}
$repo = $repo.Trim()

$isOrg = ([Environment]::GetEnvironmentVariable("GITHUB_IS_ORG") ?? "").Trim().ToLower() -in @("1","true","yes","y","on")
$defaultBranch = ([Environment]::GetEnvironmentVariable("GITHUB_DEFAULT_BRANCH") ?? "").Trim()
if ([string]::IsNullOrWhiteSpace($defaultBranch)) { $defaultBranch = "main" }

# Ensure repo exists locally
if (-not (Test-Path ".git")) {
  git init | Out-Null
}

# Ensure we are on main
try {
  git checkout -b $defaultBranch 2>$null | Out-Null
} catch {
  git checkout $defaultBranch | Out-Null
}

# Ensure at least one commit
$hasCommit = $true
try { git rev-parse --verify HEAD 1>$null 2>$null } catch { $hasCommit = $false }
if (-not $hasCommit) {
  git add -A
  git commit -m "chore: initial import" | Out-Null
}

# Create private repo on GitHub
$headers = @{
  Authorization = "Bearer $token"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
}

$body = @{
  name = $repo
  private = $true
  auto_init = $false
} | ConvertTo-Json

if ($isOrg) {
  $url = "https://api.github.com/orgs/$owner/repos"
} else {
  $url = "https://api.github.com/user/repos"
}

Write-Host "Creating repo: $owner/$repo (private=$true, org=$isOrg)"
try {
  Invoke-RestMethod -Method Post -Uri $url -Headers $headers -Body $body -ContentType "application/json" | Out-Null
} catch {
  # If it already exists, continue
  $msg = ($_ | Out-String)
  if ($msg -notmatch "name already exists" -and $msg -notmatch "422") {
    throw
  }
  Write-Host "Repo likely already exists, continuing..."
}

# Configure remote (default to https; use SSH if you prefer: set GITHUB_REMOTE=ssh)
$remoteMode = ([Environment]::GetEnvironmentVariable("GITHUB_REMOTE") ?? "https").Trim().ToLower()
if ($remoteMode -eq "ssh") {
  $remoteUrl = "git@github.com:$owner/$repo.git"
} else {
  $remoteUrl = "https://github.com/$owner/$repo.git"
}

$remotes = git remote 2>$null
if ($remotes -notmatch "origin") {
  git remote add origin $remoteUrl
} else {
  git remote set-url origin $remoteUrl
}

Write-Host "Pushing to origin/$defaultBranch ..."
git push -u origin $defaultBranch

Write-Host "Done: $remoteUrl"

