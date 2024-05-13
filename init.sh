#!/bin/bash

cd ~/git_repos
eval "$(ssh-agent)"
ssh-add github_key
cd ~/git_repos/NLP-MSAI
source venv/bin/activate
