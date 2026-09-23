# Live Jev calls. Run from the repository root after `uv sync`.
Output .cache/demo/raw.mp4
Output .cache/demo/session.txt

Require uv
Set Shell "bash"
Set Width 1440
Set Height 900
Set FontSize 30
Set FontFamily "Menlo"
Set Theme "Catppuccin Mocha"
Set Padding 40
Set Margin 32
Set MarginFill "#f6e7ee"
Set BorderRadius 18
Set WindowBar Colorful
Set Framerate 15
Set TypingSpeed 65ms
Set CursorBlink false

Env PS1 "jevgpt $ "
Env BASH_SILENCE_DEPRECATION_WARNING "1"

Hide
Type "clear"
Enter
Sleep 500ms
Show

Sleep 500ms
Type "uv run jevgpt --max-tokens 0"
Sleep 600ms
Enter
Wait+Line@60s /You:/
Sleep 500ms
Type "Why is the sky blue?"
Enter
Sleep 500ms
Wait+Line@600s /You:/
Sleep 2s
Type "Why does it turn red at sunset?"
Enter
Sleep 500ms
Wait+Line@600s /You:/
Sleep 3s
Type "How do I reverse a list in Python?"
Enter
Sleep 500ms
Wait+Line@600s /You:/
Sleep 5s
Screenshot .cache/demo/final.png
Sleep 200ms
Type "/quit"
Enter
Sleep 1s
