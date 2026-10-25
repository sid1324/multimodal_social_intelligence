# push(msg, paths): commit the given repo paths as Yash and push to main. Expects REPO, url, tok, run in scope.
def push(msg, paths):
    run('git config user.name "Yash Lothe" && git config user.email "ylothe@andrew.cmu.edu"')
    print(run('git add ' + ' '.join(paths)).stderr)
    print(run(f'git commit -q -m "{msg}"').stdout)
    r = run(f'git pull -q --rebase {url} main && git push -q {url} HEAD:main')
    print('pushed' if r.returncode == 0 else r.stderr.replace(tok, '***'))
