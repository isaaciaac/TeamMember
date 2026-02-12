#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

require_env() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "${val// }" ]]; then
    echo "Missing env var: $name" >&2
    exit 1
  fi
  echo "$val"
}

token="$(require_env GITHUB_TOKEN)"
owner="${GITHUB_OWNER:-${GITHUB_USER:-}}"
if [[ -z "${owner// }" ]]; then
  echo "Missing env var: GITHUB_OWNER (or GITHUB_USER)" >&2
  exit 1
fi
repo="${GITHUB_REPO:-teammember}"
is_org="${GITHUB_IS_ORG:-false}"
default_branch="${GITHUB_DEFAULT_BRANCH:-main}"
remote_mode="${GITHUB_REMOTE:-https}"

if [[ ! -d .git ]]; then
  git init
fi

git checkout -B "$default_branch" >/dev/null 2>&1 || git checkout "$default_branch"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add -A
  git commit -m "chore: initial import"
fi

api="https://api.github.com/user/repos"
if [[ "${is_org,,}" == "true" || "${is_org}" == "1" ]]; then
  api="https://api.github.com/orgs/$owner/repos"
fi

echo "Creating repo: $owner/$repo (private=true, org=$is_org)"
payload="$(printf '{"name":"%s","private":true,"auto_init":false}' "$repo")"
set +e
curl -sS -X POST "$api" \
  -H "Authorization: Bearer $token" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -d "$payload" >/dev/null
set -e

if [[ "${remote_mode,,}" == "ssh" ]]; then
  remote_url="git@github.com:$owner/$repo.git"
else
  remote_url="https://github.com/$owner/$repo.git"
fi

if ! git remote | grep -q '^origin$'; then
  git remote add origin "$remote_url"
else
  git remote set-url origin "$remote_url"
fi

echo "Pushing to origin/$default_branch ..."
git push -u origin "$default_branch"
echo "Done: $remote_url"

