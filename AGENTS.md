# Agent instructions for AI coding tools

This repository contains an API-key driven script. Follow these rules:

- NEVER read, print, copy, quote, or send the contents of any secrets
  file (.env, .env.*, ~/.config/ai_consensus/.env, *.pem, *.key).
- This project intentionally keeps no secrets file in the workspace:
  keys live at ~/.config/ai_consensus/.env, outside the repo. If you
  need to know whether a key is set, ask the human - never open the
  file and never guess values.
- To learn which environment variables exist, read .env.example -
  never any .env file.
- Never write API keys or tokens into source files, prompts, results,
  or logs.
