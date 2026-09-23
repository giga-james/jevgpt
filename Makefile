.PHONY: demo demo-render

VHS ?= vhs

# Executes real API requests using the local .env; answers and timing will vary.
demo:
	mkdir -p .cache/demo assets
	$(VHS) demo/chat.tape
	$(MAKE) demo-render

# Re-render the captured session without making more API calls.
demo-render:
	ffmpeg -y -i .cache/demo/raw.mp4 -vf "scale=1200:-2:flags=lanczos" -c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p -movflags +faststart -an assets/jevgpt-demo.mp4
	ffmpeg -y -i assets/jevgpt-demo.mp4 -filter_complex "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=sierra2_4a" -loop 0 assets/jevgpt-demo.gif
