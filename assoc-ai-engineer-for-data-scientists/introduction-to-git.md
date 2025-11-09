# Introduction to Version Control
- Version control
    - processes and systems to manage changes to files, programs, and directories
    - track files in different states
    - combine different version of files
    - identify a particular version
    - revert changes
## Git
- popular version control system for software development and data projects
- open source
- scalable
- Benefits
    - git stores everything, so nothing is lost
    - we can compare files at different times
    - see what changes were made by who and when
    - revert to previous version of files
### Useful Commands
`pwd`: shows current working directory
`ls` : list of all files and directories
`cd` : change directory
`git --version` : check version

## Creating repos
- Git repo: directory containing files and sub-directories
- Benefits
    - systematically track versions
    - revert to pervious versions
    - compare versions at different points in time
```shell
git init mental-health-workspace
cd mental-health-workspace
git status
git add
```
### Convert a project into a repo
```shell
git init
git status
```
### Repository
- avoid
- don't create a git repo under an exisiting one

## Staging and committing files
- The Git workflow
    - edit and save files
    - add the files to the Git staging area
        - tracks what has been modified
    - commit files
        - git takes a snapshot of the files at the point in time
        - allows us to compare and revert files
### Adding to the staging area
```shell
git add README.md
git add . #all
```
### making a commit
```shell
git commit -m "Adding a README."
```

# Version History

## Viewing the version history
### The commit structure
Three parts
- commit
    - contains the metadata - author, log message, commit time
- tree
    - tracks the names and locations of files and directories in the repo
    - like a dictionary - mapping keys to files/directories
- blob
    - Binary Large Object
    - may contain data of any kind
    - all compressed snapshot of a file's contents
`git log`

## Version history tips and tricks
### Restricting the number of commits

`git log -3` : restrict to most recent commits
`git log report.md`: restrict by file
`git log -2 file.csv`
`git log --since='Month Day Year`
`git log --since='Apr 2 2024' --until='Apr 11 2024'`
`git show 12345678`

## Comparing versions
`git diff`
`git diff report.md`
`@@ -1, 5 +1,6 @@`
    - version a: change start in line 1 and has 5 lines
    - version b: change start in line 1 and has 6 lines
### Comparing to a staged file
`git diff --staged report.md`

### Comparing two commits
```shell
git log
git diff 1234567 7654321
git diff HEAD~1 HEAD
```
## Restoring and reverting files
### Reverting files
- restoring a repo to the sate prior to the previous commit
- `git revert`  
    - reinstate previous versions and makes a commit
    - restores all files updated in the given commit
- `git revert --no-edit HEAD`
- `git revert -n HEAD`
- `git checkout HEAD~1 -- report.md` :revert a single file
- `git restore --staged file.csv`: unstaging
- `git restore --staged`

