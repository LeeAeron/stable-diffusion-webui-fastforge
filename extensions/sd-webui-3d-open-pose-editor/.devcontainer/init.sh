#!/bin/bash
echo "export GPG_TTY=\$(tty)" >> ~/.bashrc

sudo chown vscode:vscode .venv
pipenv install

sudo chown vscode:vscode node_modules
pnpm install
