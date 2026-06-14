# Spearhead

A terminal‑based learning tool using the *spear & expand* method.  
Generate an interactive knowledge tree on any topic, with content produced by Google Gemini **or** local Ollama models.  
Read, expand, add your own arguments, and export everything to a beautiful PDF on Google Drive – all from the keyboard.

## Features

- **Spear & Expand** – AI generates an overview + full chapter contents in *one* request, then you dive deeper by adding arguments.
- **Local & Cloud AI** – Switch seamlessly between Google Gemini and any Ollama model (all your local models are listed automatically).
- **Custom Tone** – Create a `~/.spearhead_tone` file and the AI will adopt your preferred writing style.
- **Streaming Content** – Watch the AI write in real time.
- **Single‑Key Export** – Press `s` to save your project **and** instantly create a LaTeX‑rendered PDF, saved locally **and** uploaded to a dedicated “Spearhead PDFs” folder on Google Drive.
- **Project Browser** – Open any previous project from a simple terminal menu.
- **Keyboard‑Driven Navigation** – tree navigation, content scrolling with left/right arrows, model switching, node deletion – no mouse needed.

## Requirements

- Python 3.8+
- [Google GenAI SDK](https://pypi.org/project/google-genai/)
- For **Gemini**: an API key (set `GEMINI_API_KEY` environment variable)
- For **local models**: [Ollama](https://ollama.com) with at least one model pulled
- For **PDF export**:
  - LaTeX (`pdflatex` – on Ubuntu: `sudo apt install texlive-latex-base`)
  - Google Cloud project with Drive API enabled (OAuth 2.0 desktop client → download `credentials.json`)

## Installation

```
git clone https://github.com/yourusername/spearhead.git
cd spearhead
pip install -r requirements.txt

# (Optional) Pull a local model
ollama pull gemma3:1b
```

## Quick Start

```
# 1. Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# 2. Run without arguments – you'll be prompted to create or open a project
python spearhead.py

# Or start with a topic immediately
python spearhead.py --topic "Quantum Physics"

# Force using a local model
python spearhead.py --topic "Ethics of AI" --local --local-model gemma3:1b
```

**Inside the app:**

| Key            | Action                                           |
|----------------|--------------------------------------------------|
| `↑` / `↓`     | Navigate the tree                                |
| `Enter`        | Generate content for selected node (if empty)    |
| `a`            | Add a new argument (child node)                  |
| `d`            | Delete selected node (cannot delete root)        |
| `r`            | Regenerate content for selected node             |
| `h` / `l`      | Collapse / expand tree node                      |
| `←` / `→`      | Scroll content view (page up / down)             |
| `PgUp` / `PgDn`| Also scroll content                              |
| `m`            | Switch AI model (Gemini or Ollama)               |
| `s`            | Save project **and** export PDF to Drive + local |
| `q`            | Quit                                             |

The current model is always shown in the footer.

## Tone / Style Personalisation

Create a file at `~/.spearhead_tone` (or pass `--tone-file /path/to/file`).  
Write any style instructions (see `example-tone.txt`). They will be appended to every AI prompt.

## PDF Export Setup

1. Install LaTeX (e.g., `sudo apt install texlive-latex-base`)
2. Place a `credentials.json` (Google OAuth 2.0 client ID) in the project folder.
3. The first export will open a browser for authorisation. Afterwards, a `token.json` is saved and used automatically.
4. PDFs are saved locally in `~/Documents/spearhead_projects/pdf/` and uploaded to a “Spearhead PDFs” folder on your Drive.

## Verbose Mode

Add `-v` to print full export logs after quitting, useful for debugging:

```
python spearhead.py --topic "Renaissance Art" -v
```

## Project Files

All projects are stored as JSON in `~/Documents/spearhead_projects/`.  
You can share them, version‑control them, or edit them manually.

## License

MIT – see [LICENSE](LICENSE) file.
