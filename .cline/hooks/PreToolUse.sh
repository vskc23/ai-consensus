#!/bin/bash
# Cline PreToolUse hook: block reading/editing secret files.
# .clineignore only filters auto-loaded context; this actually
# blocks Read/Write/Edit tool calls. Enable Hooks in Cline settings.
INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
if [ -z "$FILE_PATH" ]; then
  FILE_PATH=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
fi

for pattern in '.env' '.env.*' '*.pem' '*.key' '*/secrets/*' '*/credentials/*'; do
  case "$FILE_PATH" in
    $pattern)
      echo "BLOCKED: $FILE_PATH is a protected file." >&2
      exit 2
      ;;
  esac
done

exit 0
