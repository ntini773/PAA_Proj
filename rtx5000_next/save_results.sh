#!/usr/bin/env bash
# Save the small result files to a separate git branch (default: results-eval2) and push it, so nothing is lost if the GPU
# session ends. run_eval2.sh calls this after every stage; you can also run it by hand.
#
#   bash save_results.sh [label]        save now (label goes into the commit message)
#   bash save_results.sh --check        test that pushing works (credentials) without saving anything
#   NO_SAVE=1 bash run_eval2.sh ...     turn the automatic save off
#   SAVE_DIR=/some/other/disk bash save_results.sh    also copy the same files to another directory (a mount, a share)
#
# What is saved: results/ and logs/ text, csv, json, md, png (figures, contact sheets) and PROGRESS_EVAL2.md. What is left out:
# *.nsys-rep, *.ncu-rep, *.sqlite, *.tgz, *.pt, per-image outputs (p<i>*.png), build/, anything over 5 MB. Those stay on the server
# (copy them with the tar from E15).
# How: a second git worktree (outside this repo folder) holds an orphan branch with one folder, latest/, that is overwritten at each
# save; the branch history keeps every version. The main branch and your working files are never touched, so `git pull` is unaffected.
# Failures (no credentials, no network) never stop a run: the files stay committed in the worktree and the next save pushes them.
# Env: RESULTS_WT (worktree path, default ../PAA_results_eval2), RESULTS_BRANCH (default results-eval2), SAVE_REMOTE (default origin).
cd "$(dirname "$0")"
REPO=$(git rev-parse --show-toplevel) || { echo "not a git repo"; exit 0; }
WT=${RESULTS_WT:-$(dirname "$REPO")/PAA_results_eval2}
BR=${RESULTS_BRANCH:-results-eval2}
REMOTE=${SAVE_REMOTE:-origin}
LABEL=${1:-manual}

if [ "$LABEL" = "--check" ]; then
  echo "remote: $(git -C "$REPO" remote get-url "$REMOTE" 2>&1)"
  if git -C "$REPO" ls-remote "$REMOTE" > /dev/null 2>&1; then echo "read access: ok"; else echo "read access: FAILED"; fi
  if git -C "$REPO" push --dry-run "$REMOTE" HEAD:refs/heads/$BR > /tmp/save_check.log 2>&1; then echo "push access: ok (dry run)";
  else echo "push access: FAILED, so saves stay local in $WT. Fix the credentials, e.g. a GitHub token with the credential helper:"
       echo "    git config --global credential.helper store    # then one manual 'git push' asks for user name and token once"
       sed 's/^/    /' /tmp/save_check.log | tail -n 5; fi
  exit 0
fi

mkdir -p "$(dirname "$WT")"
exec 9> "$WT.lock"; flock 9   # parallel stages save one at a time

if [ ! -e "$WT/.git" ]; then
  git -C "$REPO" fetch -q "$REMOTE" "$BR" 2> /dev/null
  if git -C "$REPO" rev-parse -q --verify "refs/remotes/$REMOTE/$BR" > /dev/null 2>&1; then
    git -C "$REPO" worktree add -q -B "$BR" "$WT" "$REMOTE/$BR" || exit 0
  else
    git -C "$REPO" worktree add -q --detach "$WT" HEAD || exit 0
    git -C "$WT" checkout -q --orphan "$BR" && git -C "$WT" rm -rq -f . 2> /dev/null
  fi
fi

mkdir -p "$WT/latest"
tar -cf - --exclude='*.nsys-rep' --exclude='*.ncu-rep' --exclude='*.sqlite' --exclude='*.tgz' --exclude='*.pt' \
    --exclude='build' --exclude='__pycache__' --exclude='p[0-9]*.png' --exclude='*cuda_gpu_trace*' --exclude='*cuda_api_sum*' results logs PROGRESS_EVAL2.md 2> /dev/null | tar -xf - -C "$WT/latest"
find "$WT/latest" -type f -size +5M -delete   # keep the branch small
[ -n "$SAVE_DIR" ] && { mkdir -p "$SAVE_DIR" && cp -r "$WT/latest/." "$SAVE_DIR/" && echo "also copied to $SAVE_DIR"; }

cd "$WT" || exit 0
git add -A latest
if git diff --cached --quiet; then echo "save: nothing new"; exit 0; fi
ID=(); git config user.email > /dev/null || ID=(-c user.name=eval2-saver -c user.email=eval2-saver@localhost)
git "${ID[@]}" commit -q -m "eval2 results after $LABEL ($(date '+%Y-%m-%d %H:%M'))"
if git push -q "$REMOTE" "$BR" 2> /tmp/save_push.log; then echo "save: pushed $LABEL to $REMOTE/$BR"
else echo "save: committed locally in $WT; push failed (run: bash save_results.sh --check)"; tail -n 2 /tmp/save_push.log; fi
exit 0
