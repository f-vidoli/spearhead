#!/usr/bin/env python3
"""
spearhead - A curses-based learning tool using the "spear & expand" method.
Supports Google Gemini and local models via Ollama.
Switch models inside the app with 'm' – shows installed Ollama models
and multiple Gemini options. Model name permanently displayed in footer.
Content scrolling uses LEFT/RIGHT arrows as page-up/page-down.
On save ('s'), automatically exports PDF to Google Drive using spearhead_export.py.
Verbose logging with -v prints AI and export logs after quitting.

Requirements:
    pip install google-genai   (and on Windows: pip install windows-curses)
    For local models: install Ollama and pull a model (e.g. gemma3:1b)
    For PDF export: install texlive (or equivalent) and google-api-python-client.

Run:
    python spearhead.py [project_file.json]
    python spearhead.py --topic "Physics"
    python spearhead.py --local --local-model gemma3:1b
    python spearhead.py --tone-file /path/to/tone.txt
    python spearhead.py -v              # prints AI and export logs after exit
"""

import os
import sys
import json
import threading
import curses
import textwrap
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import re


# ----------------------------------------------------------------------
# AI Backend Abstraction
# ----------------------------------------------------------------------
class BaseAIClient(ABC):
    def __init__(self):
        self.tone = ""
        self.last_raw_response = ""

    def set_tone(self, tone_text: str):
        self.tone = tone_text

    def _add_tone_to_prompt(self, prompt: str) -> str:
        if self.tone:
            return f"{prompt}\n\nStyle instructions: {self.tone}"
        return prompt

    # ----- Prompt builders -----
    def _build_overview_prompt(self, topic: str) -> str:
        """Prompt for introduction + chapter titles only."""
        prompt = (
            f'You are an expert tutor. For the topic "{topic}", produce a learning outline as JSON.\n'
            f'The JSON must contain exactly two fields:\n'
            f'- "introduction": a string of 3-5 paragraphs, very broad and engaging (no markdown).\n'
            f'- "chapters": a list of 6-12 chapter titles (strings only, high-level, no descriptions).\n\n'
            f'Output ONLY valid JSON, no other text.'
        )
        return self._add_tone_to_prompt(prompt)

    def _build_node_prompt(self, node, root_title: str) -> str:
        ancestors = node.get_path_titles()[:-1]
        context_str = " -> ".join(ancestors) if ancestors else root_title
        if node.node_type == "chapter":
            prompt = (
                f'You are helping a learner explore the topic: "{root_title}".\n'
                f'The learner has already read the general introduction and now wants to dive into the chapter:\n'
                f'"{node.title}".\n\n'
                f'Write a thorough, clear explanation of this chapter\'s main ideas. '
                f'Use plain language, examples, and avoid markdown. Length: 400-800 words.\n'
                f'Start directly with the explanation, no greetings.'
            )
        else:
            parent_title = node.parent.title if node.parent else "unknown"
            prompt = (
                f'You are helping a learner explore the topic: "{root_title}".\n'
                f'The current context is:\n'
                f'{context_str} -> {parent_title}\n\n'
                f'Now write a detailed, accessible explanation of the specific point:\n'
                f'"{node.title}"\n\n'
                f'Provide insights, examples, and (if relevant) counterpoints or nuances. '
                f'Use plain text, no markdown. Length: 300-700 words.'
            )
        return self._add_tone_to_prompt(prompt)

    def _build_tagged_overview_prompt(self, topic: str) -> str:
        """Prompt for introduction + all chapter contents (tag‑based, used by 'f')."""
        prompt = (
            f'You are an expert tutor. For the topic "{topic}", produce a comprehensive learning document.\n'
            f'Use the following format EXACTLY:\n\n'
            f'[INTRODUCTION]\n'
            f'Write a broad, engaging introduction of 3-5 paragraphs. No markdown.\n\n'
            f'[NEW CHAPTER]\n'
            f'Chapter Title Here\n'
            f'[CHAPTER CONTENT]\n'
            f'Write a detailed explanation of this chapter (400-800 words). No markdown.\n\n'
            f'[NEW CHAPTER]\n'
            f'Next Chapter Title\n'
            f'[CHAPTER CONTENT]\n'
            f'Content for next chapter...\n\n'
            f'... (continue for 6-12 chapters)\n\n'
            f'[EOF]\n\n'
            f'IMPORTANT:\n'
            f'- Exactly one [INTRODUCTION] block at the start.\n'
            f'- Each chapter starts with [NEW CHAPTER] followed by the title on its own line, then [CHAPTER CONTENT] and the content.\n'
            f'- End the whole document with [EOF].\n'
            f'- No extra commentary, no markdown. Use plain text throughout.'
        )
        return self._add_tone_to_prompt(prompt)

    # ----- Parsers -----
    def _parse_overview_response(self, text: str) -> Tuple[str, List[str]]:
        """Parse old‑style JSON overview (introduction + chapter titles)."""
        text = text.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"AI returned invalid JSON: {e}") from e
        if "introduction" not in data or "chapters" not in data:
            raise RuntimeError("AI response missing required fields")
        if not isinstance(data["chapters"], list) or not data["chapters"]:
            raise RuntimeError("AI response 'chapters' must be a non-empty list")
        if not isinstance(data["introduction"], str):
            raise RuntimeError("AI response 'introduction' must be a string")
        return data["introduction"], data["chapters"]

    def _parse_tagged_overview(self, text: str) -> Tuple[str, List[dict]]:
        """Parse tag‑based format, ignoring anything before [INTRODUCTION] and after [EOF]."""
        text = text.strip()
        for fence in ("```", "```text", "```plain"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        intro_match = re.search(r'\[INTRODUCTION\]\s*(.*?)\s*(\[NEW CHAPTER\]|\[EOF\])', text, re.DOTALL)
        if not intro_match:
            raise RuntimeError("Missing [INTRODUCTION] tag")
        introduction = intro_match.group(1).strip()

        chapters = []
        chapter_pattern = re.compile(
            r'\[NEW CHAPTER\]\s*(.*?)\s*\[CHAPTER CONTENT\]\s*(.*?)\s*(?=\[NEW CHAPTER\]|\[EOF\]|$)',
            re.DOTALL
        )
        for match in chapter_pattern.finditer(text):
            title = match.group(1).strip()
            content = match.group(2).strip()
            if title and content:
                chapters.append({"title": title, "content": content})

        if not chapters:
            raise RuntimeError("No valid chapters found; expected at least one [NEW CHAPTER]")
        return introduction, chapters

    # ----- API -----
    @abstractmethod
    def generate_overview(self, topic: str) -> Tuple[str, List[str]]:
        """Old method: introduction + chapter titles only."""
        ...

    @abstractmethod
    def generate_node_content(self, node, root_title: str) -> str:
        """Generate content for a single node."""
        ...

    @abstractmethod
    def generate_full_overview(self, topic: str) -> Tuple[str, List[dict]]:
        """Full generation: introduction + all chapter contents."""
        ...

    def generate_node_content_stream(self, node, root_title: str, callback: Callable[[str], None]) -> str:
        text = self.generate_node_content(node, root_title)
        callback(text)
        return text

    @abstractmethod
    def get_model_display(self) -> str:
        ...


# Google Gemini client (lazy initialisation)
class GeminiClient(BaseAIClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self._client = None
        self.model = "gemini-2.5-flash"

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _friendly_error(self, exc: Exception) -> str:
        from google.genai import errors as genai_errors
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, 'code', None)
            if code == 429:
                return "Gemini quota exceeded. Switching to local model..."
            return f"Gemini API error ({code})"
        return str(exc)

    def generate_overview(self, topic: str) -> Tuple[str, List[str]]:
        from google.genai import types
        prompt = self._build_overview_prompt(topic)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4)
            )
            self.last_raw_response = response.text
            return self._parse_overview_response(response.text)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(self._friendly_error(e)) from e

    def generate_node_content(self, node, root_title: str) -> str:
        from google.genai import types
        prompt = self._build_node_prompt(node, root_title)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            self.last_raw_response = response.text
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(self._friendly_error(e)) from e

    def generate_node_content_stream(self, node, root_title: str, callback: Callable[[str], None]) -> str:
        from google.genai import types
        prompt = self._build_node_prompt(node, root_title)
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            full_text = ""
            for chunk in stream:
                full_text += chunk.text or ""
                callback(full_text)
            self.last_raw_response = full_text
            return full_text.strip()
        except Exception as e:
            raise RuntimeError(self._friendly_error(e)) from e

    def generate_full_overview(self, topic: str) -> Tuple[str, List[dict]]:
        from google.genai import types
        prompt = self._build_tagged_overview_prompt(topic)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4)
            )
            self.last_raw_response = response.text
            return self._parse_tagged_overview(response.text)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(self._friendly_error(e)) from e

    def get_model_display(self) -> str:
        return f"Gemini ({self.model})"


# Local Ollama client
class OllamaClient(BaseAIClient):
    def __init__(self, model: str = "gemma3:1b"):
        super().__init__()
        self.model = model
        import requests
        self.session = requests.Session()
        self.base_url = "http://localhost:11434/api"

    def _chat(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature}
        }
        resp = self.session.post(f"{self.base_url}/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def generate_overview(self, topic: str) -> Tuple[str, List[str]]:
        prompt = self._build_overview_prompt(topic)
        raw = self._chat(prompt, temperature=0.4)
        self.last_raw_response = raw
        raw = raw.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
        if raw.endswith("```"):
            raw = raw[:-3]
        return self._parse_overview_response(raw)

    def generate_node_content(self, node, root_title: str) -> str:
        prompt = self._build_node_prompt(node, root_title)
        raw = self._chat(prompt, temperature=0.7)
        self.last_raw_response = raw
        return raw

    def generate_node_content_stream(self, node, root_title: str, callback: Callable[[str], None]) -> str:
        prompt = self._build_node_prompt(node, root_title)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.7}
        }
        resp = self.session.post(f"{self.base_url}/chat", json=payload, stream=True, timeout=120)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    full_text += content
                    callback(full_text)
                except json.JSONDecodeError:
                    pass
        self.last_raw_response = full_text
        return full_text.strip()

    def generate_full_overview(self, topic: str) -> Tuple[str, List[dict]]:
        prompt = self._build_tagged_overview_prompt(topic)
        raw = self._chat(prompt, temperature=0.4)
        self.last_raw_response = raw
        return self._parse_tagged_overview(raw)

    def get_model_display(self) -> str:
        return f"Ollama ({self.model})"


# ----------------------------------------------------------------------
# Data model (unchanged)
# ----------------------------------------------------------------------
class Node:
    def __init__(self, title: str, node_type: str = "argument", parent: "Node" = None):
        self.title = title
        self.content = ""
        self.node_type = node_type
        self.parent = parent
        self.children: List['Node'] = []
        self.collapsed = False

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "node_type": self.node_type,
            "children": [c.to_dict() for c in self.children],
            "collapsed": self.collapsed,
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["Node"] = None) -> "Node":
        node = cls(data["title"], data["node_type"], parent)
        node.content = data["content"]
        node.collapsed = data.get("collapsed", False)
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, node)
            node.children.append(child)
        return node

    def get_path_titles(self) -> List[str]:
        path = []
        cur = self
        while cur:
            path.append(cur.title)
            cur = cur.parent
        return list(reversed(path))

    def is_root(self) -> bool:
        return self.node_type == "root"


# ----------------------------------------------------------------------
# Project file handling (unchanged)
# ----------------------------------------------------------------------
class ProjectManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath).expanduser().resolve()
        self.root: Optional[Node] = None

    def save(self):
        if self.root is None:
            return
        data = {"project_name": self.root.title, "root": self.root.to_dict()}
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> Node:
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = Node.from_dict(data["root"])
        return self.root

    @staticmethod
    def new_project(topic: str, save_dir: str = None) -> "ProjectManager":
        if save_dir is None:
            save_dir = os.path.join(str(Path.home()), "Documents", "spearhead_projects")
        os.makedirs(save_dir, exist_ok=True)
        safe_topic = "".join(c for c in topic if c.isalnum() or c in " _-").strip()
        filepath = os.path.join(save_dir, f"{safe_topic}.json")
        pm = ProjectManager(filepath)
        pm.root = Node(topic, node_type="root")
        return pm


# ----------------------------------------------------------------------
# Curses UI (with dual generation modes)
# ----------------------------------------------------------------------
class SpearheadApp:
    H_LINE, V_LINE = "─", "│"
    TL_CORNER, TR_CORNER = "┌", "┐"
    BL_CORNER, BR_CORNER = "└", "┘"
    L_SPLIT, R_SPLIT = "├", "┤"
    B_SPLIT, T_SPLIT = "┴", "┬"

    def __init__(self, stdscr, pm: ProjectManager, ai_client: BaseAIClient, verbose: bool = False):
        self.stdscr = stdscr
        self.pm = pm
        self.ai = ai_client
        self.root = pm.root
        self.current_node = self.root
        self.tree_offset = 0
        self.content_scroll = 0
        self.generating = False
        self.status_msg = ""
        self.is_error = False
        self.generating_base_msg = ""
        self.ui_needs_update = False
        self._stream_timeout = 150
        self.verbose = verbose

        self.height, self.width = stdscr.getmaxyx()
        if self.height < 10 or self.width < 40:
            raise RuntimeError("Terminal too small (need at least 40x10).")

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # header
        curses.init_pair(2, curses.COLOR_YELLOW, -1)                 # footer info
        curses.init_pair(3, curses.COLOR_CYAN, -1)                   # tree highlight
        curses.init_pair(4, curses.COLOR_GREEN, -1)                  # content title
        curses.init_pair(5, curses.COLOR_RED, -1)                    # error messages
        curses.init_pair(6, curses.COLOR_BLACK, -1)                  # borders

        curses.curs_set(0)

        if self.root.children:
            self.set_status("Project loaded. Select a node to begin.")
        else:
            self.set_status("New project created. Press Enter on root for overview, 'f' for full content.")

    def set_status(self, msg: str, is_error: bool = False):
        self.status_msg = msg
        self.is_error = is_error
        self.generating_base_msg = msg

    # ------------------------------------------------------------------
    # Safe addstr wrapper
    # ------------------------------------------------------------------
    def _safe_addstr(self, y: int, x: int, text: str, *attrs):
        try:
            if attrs:
                self.stdscr.addstr(y, x, text, *attrs)
            else:
                self.stdscr.addstr(y, x, text)
        except curses.error:
            pass

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _header_area(self): return 0, 0, 1, self.width
    def _tree_area(self):
        tree_w = max(12, self.width // 3)
        return 1, 0, self.height - 3, tree_w
    def _content_area(self):
        tree_w = self._tree_area()[3]
        return 1, tree_w + 2, self.height - 3, self.width - tree_w - 2
    def _footer_area(self): return self.height - 2, 0, 2, self.width
    def _divider_col(self): return self._tree_area()[3]

    # ------------------------------------------------------------------
    # Drawing (unchanged)
    # ------------------------------------------------------------------
    def draw_all(self):
        self.stdscr.clear()
        self.draw_borders()
        self.draw_header()
        self.draw_tree()
        self.draw_content()
        self.draw_footer()
        self.stdscr.refresh()

    def draw_borders(self):
        self.stdscr.attron(curses.color_pair(6))
        h, w = self.height, self.width
        div = self._divider_col()

        self._safe_addstr(0, 0, self.TL_CORNER)
        self._safe_addstr(0, w - 1, self.TR_CORNER)
        self._safe_addstr(0, 1, self.H_LINE * (w - 2))

        for y in range(1, h - 2):
            self._safe_addstr(y, 0, self.V_LINE)
            self._safe_addstr(y, w - 1, self.V_LINE)
            self._safe_addstr(y, div, self.V_LINE)

        self._safe_addstr(h - 2, 0, self.L_SPLIT)
        self._safe_addstr(h - 2, w - 1, self.R_SPLIT)
        self._safe_addstr(h - 2, 1, self.H_LINE * (w - 2))
        self._safe_addstr(h - 2, div, self.B_SPLIT)
        self.stdscr.attroff(curses.color_pair(6))

    def draw_header(self):
        y, x, h, w = self._header_area()
        self.stdscr.attron(curses.color_pair(1))
        title = f" SPEARHEAD — {self.root.title} "
        model_display = f"[{self.ai.get_model_display()}]"
        self._safe_addstr(y, x + 1, title[:w - 2].center(w - 2))
        model_str = model_display[:w - 2]
        self._safe_addstr(y, w - len(model_str) - 2, model_str)
        self.stdscr.attroff(curses.color_pair(1))

    def _get_tree_lines(self) -> List[Tuple["Node", str]]:
        result = []
        self._collect_tree_lines(self.root, result, "")
        return result

    def _collect_tree_lines(self, node: "Node", lines: list, prefix: str):
        indicator = ">" if node is self.current_node else " "
        if node.is_root():
            lines.append((node, f"{indicator} * {node.title}"))
        else:
            icon = ("v" if not node.collapsed else ">") if node.children else "."
            lines.append((node, prefix + f"{indicator} {icon} {node.title}"))
        if not node.collapsed:
            child_prefix = prefix + "    "
            for child in node.children:
                self._collect_tree_lines(child, lines, child_prefix)

    def draw_tree(self):
        y, x, h, w = self._tree_area()
        if not self.root:
            return
        lines = self._get_tree_lines()
        self.tree_offset = max(0, min(self.tree_offset, len(lines) - h))
        for i, (node, line) in enumerate(lines[self.tree_offset:]):
            if i >= h:
                break
            display = line[:w - 1]
            if node is self.current_node:
                self.stdscr.attron(curses.color_pair(3))
                self._safe_addstr(y + i, x + 1, display)
                self.stdscr.attroff(curses.color_pair(3))
            else:
                self._safe_addstr(y + i, x + 1, display)

    def draw_content(self):
        y, x, h, w = self._content_area()
        if not self.current_node:
            return
        content = self.current_node.content
        if not content:
            if self.current_node.is_root() and not self.current_node.children:
                content = "Welcome to Spearhead!\n\nPress Enter to generate the learning overview.\nPress 'f' to generate the full overview with all chapter content."
            else:
                content = "[No content yet. Press Enter to generate.]"
        max_w = w - 1
        wrapped = []
        for p in content.splitlines():
            if p == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(p, width=max_w) or [""])
        total = len(wrapped)
        max_disp = h - 1
        self.content_scroll = max(0, min(self.content_scroll, max(0, total - max_disp)))

        self.stdscr.attron(curses.color_pair(4))
        self._safe_addstr(y, x, f"-- {self.current_node.title} --"[:max_w])
        self.stdscr.attroff(curses.color_pair(4))

        for i in range(max_disp):
            li = self.content_scroll + i
            if li < total:
                self._safe_addstr(y + 1 + i, x, wrapped[li][:max_w])

    def draw_footer(self):
        y, x, h, w = self._footer_area()
        model_str = self.ai.get_model_display()
        help_text = (
            "[↑/↓] nav  [Enter] gen  [f] full gen  [a] add  [d] del  "
            "[r] regen  [h/l] collapse/expand  [←/→] pgup/pgdn content  "
            "[PgUp/PgDn] also scroll  [m] model  [s] save + export  [q] quit"
        )
        full_help = f"{help_text}  |  {model_str}"
        self.stdscr.attron(curses.color_pair(2))
        self._safe_addstr(y, x + 1, full_help[:w - 2])
        self.stdscr.attroff(curses.color_pair(2))

        status_y = y + 1
        if self.generating:
            msg = f" Working... {self.generating_base_msg}"
            self.stdscr.attron(curses.color_pair(2))
            self._safe_addstr(status_y, x + 1, msg[:w - 2])
            self.stdscr.attroff(curses.color_pair(2))
        elif self.status_msg:
            color = curses.color_pair(5) if self.is_error else curses.color_pair(2)
            self.stdscr.attron(color)
            self._safe_addstr(status_y, x + 1, self.status_msg[:w - 2])
            self.stdscr.attroff(color)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    def run(self):
        self.draw_all()
        self.stdscr.timeout(self._stream_timeout)

        while True:
            key = self.stdscr.getch()

            if self.generating:
                if self.ui_needs_update:
                    self.ui_needs_update = False
                    self.draw_all()
                continue

            if key == -1:
                continue

            if key == ord('q'):
                break
            elif key == curses.KEY_RESIZE:
                self.height, self.width = self.stdscr.getmaxyx()
                self.draw_all()
                continue
            elif key == curses.KEY_UP:
                self.move_selection(-1)
            elif key == curses.KEY_DOWN:
                self.move_selection(1)
            elif key == curses.KEY_LEFT or key == ord('h'):
                if key == curses.KEY_LEFT:
                    self.content_scroll -= max(1, self.height - 4)
                    if self.content_scroll < 0:
                        self.content_scroll = 0
                else:
                    self.toggle_collapse(True)
            elif key == curses.KEY_RIGHT or key == ord('l'):
                if key == curses.KEY_RIGHT:
                    self.content_scroll += max(1, self.height - 4)
                else:
                    self.toggle_collapse(False)
            elif key == ord('\n') or key == ord(' '):
                self.ensure_node_content()
            elif key == ord('f'):
                self.full_generation()
            elif key == ord('a'):
                self.add_argument()
            elif key == ord('d'):
                self.delete_node()
            elif key == ord('r'):
                self.regenerate_node()
            elif key == ord('m'):
                self.switch_model()
            elif key == ord('s'):
                self.save_and_export()
            elif key == curses.KEY_PPAGE:
                self.content_scroll -= max(1, self.height - 4)
            elif key == curses.KEY_NPAGE:
                self.content_scroll += max(1, self.height - 4)

            self.draw_all()

    def save_and_export(self):
        self.pm.save()
        self.set_status(f"Saved to {self.pm.filepath}. Exporting PDF...")
        self.draw_all()

        export_script = os.path.join(os.path.dirname(__file__), "spearhead_export.py")
        if not os.path.exists(export_script):
            self.set_status("Warning: spearhead_export.py not found. Only saved.", is_error=True)
            self.draw_all()
            return

        def run_export():
            try:
                proc = subprocess.run(
                    [sys.executable, export_script, str(self.pm.filepath)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if proc.returncode == 0:
                    self.set_status("PDF exported successfully (local + Drive).")
                else:
                    error = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
                    if len(error) > 80:
                        error = error[:80] + "..."
                    self.set_status(f"Export failed: {error}", is_error=True)
            except subprocess.TimeoutExpired:
                self.set_status("Export timed out (60s).", is_error=True)
            except Exception as e:
                self.set_status(f"Export error: {e}", is_error=True)
            self.draw_all()

        threading.Thread(target=run_export, daemon=True).start()

    # ------------------------------------------------------------------
    # Tree navigation (unchanged)
    # ------------------------------------------------------------------
    def move_selection(self, delta: int):
        visible = self._collect_visible_nodes()
        if not visible:
            return
        try:
            idx = visible.index(self.current_node)
        except ValueError:
            idx = 0
        new = idx + delta
        if 0 <= new < len(visible):
            self.current_node = visible[new]
            self._ensure_visible_in_tree()

    def _collect_visible_nodes(self) -> List["Node"]:
        result = []
        def walk(node: "Node"):
            result.append(node)
            if not node.collapsed:
                for child in node.children:
                    walk(child)
        walk(self.root)
        return result

    def _ensure_visible_in_tree(self):
        lines = self._get_tree_lines()
        for i, (node, _) in enumerate(lines):
            if node is self.current_node:
                h = self._tree_area()[2]
                if i < self.tree_offset:
                    self.tree_offset = i
                elif i >= self.tree_offset + h:
                    self.tree_offset = i - h + 1
                break

    def toggle_collapse(self, collapse: bool):
        if self.current_node and self.current_node.children:
            self.current_node.collapsed = collapse

    # ------------------------------------------------------------------
    # AI generation – dual mode
    # ------------------------------------------------------------------
    def _write_ai_log(self, raw_text: str):
        if not self.verbose:
            return
        log_path = str(self.pm.filepath).replace('.json', '.ai.log')
        try:
            with open(log_path, 'w', encoding='utf-8') as lf:
                lf.write(raw_text)
        except Exception:
            pass

    def generate_initial_overview_old(self):
        """Old method: introduction + chapter titles only."""
        self.generating = True
        self.generating_base_msg = "Generating overview (introduction + chapter titles)..."
        self.set_status(self.generating_base_msg)
        self.draw_all()

        def task():
            retrying = False
            try:
                intro, chapters = self.ai.generate_overview(self.root.title)
                self._write_ai_log(self.ai.last_raw_response)
                chap_list = "\n".join(f"• {ch}" for ch in chapters)
                self.root.content = f"{intro}\n\n**Chapters:**\n{chap_list}"
                for ch_title in chapters:
                    chap_node = Node(ch_title, node_type="chapter", parent=self.root)
                    # content remains empty – user must expand individually
                    self.root.children.append(chap_node)
                self.current_node = self.root
                self.pm.save()
                self.set_status("Overview generated. Select a chapter and press Enter to expand.")
            except Exception as e:
                retrying = self._handle_ai_error(e, "overview")
            finally:
                if not retrying:
                    self.generating = False
                    self.draw_all()

        threading.Thread(target=task, daemon=True).start()

    def full_generation(self):
        """'f' key – generate introduction + all chapter contents in one request."""
        if self.current_node != self.root:
            self.set_status("Full generation only available on the root node.", is_error=True)
            self.draw_all()
            return

        self.generating = True
        self.generating_base_msg = "Generating full overview (introduction + all chapter content)..."
        self.set_status(self.generating_base_msg)
        self.draw_all()

        def task():
            retrying = False
            try:
                intro, chapters_data = self.ai.generate_full_overview(self.root.title)
                self._write_ai_log(self.ai.last_raw_response)
                chap_list = "\n".join(f"• {ch['title']}" for ch in chapters_data)
                self.root.content = f"{intro}\n\n**Chapters:**\n{chap_list}"
                # Clear existing children (if any)
                self.root.children.clear()
                for ch in chapters_data:
                    chap_node = Node(ch['title'], node_type="chapter", parent=self.root)
                    chap_node.content = ch['content']
                    self.root.children.append(chap_node)
                self.current_node = self.root
                self.pm.save()
                self.set_status("Full overview generated. All chapters ready.")
            except Exception as e:
                retrying = self._handle_ai_error(e, "overview")
            finally:
                if not retrying:
                    self.generating = False
                    self.draw_all()

        threading.Thread(target=task, daemon=True).start()

    def ensure_node_content(self):
        """Enter / space: generate content for the current node if empty.
        For the root, uses the old overview (titles only) if it has no children."""
        if self.current_node.is_root() and not self.current_node.children:
            # New project – old method
            self.generate_initial_overview_old()
            return
        if self.current_node.content:
            self.set_status("Node already has content. Press 'r' to regenerate.")
            self.draw_all()
            return
        # For any other empty node, use streaming generation
        self._start_streaming_generation()

    def regenerate_node(self):
        if self.current_node.is_root():
            self.set_status("Cannot regenerate root overview. Start a new project instead.")
            self.draw_all()
            return
        self._start_streaming_generation()

    def _start_streaming_generation(self):
        self.generating = True
        self.ui_needs_update = False
        self.generating_base_msg = f"Generating content for '{self.current_node.title}'..."
        self.set_status(self.generating_base_msg)
        self.draw_all()

        node = self.current_node
        root_title = self.root.title

        def callback(text_so_far: str):
            node.content = text_so_far
            self.ui_needs_update = True

        def task():
            retrying = False
            try:
                final = self.ai.generate_node_content_stream(node, root_title, callback)
                node.content = final
                self.pm.save()
                self._write_ai_log(self.ai.last_raw_response)
                self.set_status("Content generated.")
            except Exception as e:
                retrying = self._handle_ai_error(e, "generation")
            finally:
                if not retrying:
                    self.generating = False
                    self.draw_all()

        threading.Thread(target=task, daemon=True).start()

    def _handle_ai_error(self, error: Exception, action: str) -> bool:
        msg = str(error)
        if not isinstance(self.ai, OllamaClient) and (
            "quota" in msg.lower() or "429" in msg or "api error" in msg.lower()
        ):
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=2)
                if resp.status_code == 200:
                    old_tone = self.ai.tone
                    self.ai = OllamaClient(model="gemma3:1b")
                    self.ai.set_tone(old_tone)
                    self.set_status(f"Quota exceeded. Switched to local model ({self.ai.get_model_display()}). Retrying...")
                    if action == "overview":
                        self.generate_initial_overview_old()
                    else:
                        self._start_streaming_generation()
                    return True
            except Exception:
                pass
        self.set_status(f"Error: {msg}", is_error=True)
        return False

    # ------------------------------------------------------------------
    # Model switcher (unchanged)
    # ------------------------------------------------------------------
    def _get_ollama_models(self) -> List[str]:
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return [m["name"] for m in models if "name" in m]
        except Exception:
            pass
        return []

    def switch_model(self):
        if self.generating:
            self.set_status("Cannot switch models while generating.", is_error=True)
            return

        old_timeout = self._stream_timeout
        self.stdscr.timeout(-1)
        curses.flushinp()
        self.stdscr.clear()

        options = []
        if os.environ.get("GEMINI_API_KEY"):
            gemini_models = [
                ("gemini-2.5-flash",      "Gemini 2.5 Flash (recommended, generous quota)"),
                ("gemini-2.0-flash-lite", "Gemini 2.0 Flash‑Lite (high free quota)"),
                ("gemini-2.0-flash",      "Gemini 2.0 Flash (fast)"),
                ("gemini-2.5-pro",        "Gemini 2.5 Pro"),
            ]
            for model_id, label in gemini_models:
                options.append((f"Gemini: {label}", lambda mid=model_id: self._set_gemini_model(mid)))
        else:
            options.append(("Gemini: not available (set GEMINI_API_KEY)", None))

        local_models = self._get_ollama_models()
        if local_models:
            for model_name in sorted(local_models):
                options.append((f"Ollama: {model_name}", lambda mn=model_name: self._set_ollama_model(mn)))
        else:
            options.append(("Ollama: enter model name manually", self._prompt_ollama_custom))
            options.append(("Ollama: default (gemma3:1b)", lambda: self._set_ollama_model("gemma3:1b")))

        options.append(("Cancel", None))

        for i, (text, _) in enumerate(options):
            self._safe_addstr(i, 0, f"{i+1}. {text}")
        prompt_y = len(options) + 1
        self._safe_addstr(prompt_y, 0, "Enter choice number: ")
        self.stdscr.refresh()

        curses.echo()
        curses.curs_set(1)
        try:
            choice_str = self.stdscr.getstr(prompt_y, 21, 3).decode('utf-8').strip()
        except Exception:
            choice_str = ""
        finally:
            curses.noecho()
            curses.curs_set(0)
            self.stdscr.timeout(old_timeout)

        try:
            idx = int(choice_str) - 1
            if 0 <= idx < len(options):
                _, action = options[idx]
                if action:
                    action()
                    self.set_status(f"Model switched to {self.ai.get_model_display()}.")
                else:
                    self.set_status("Model switch cancelled.")
            else:
                self.set_status("Invalid choice. Switch cancelled.")
        except ValueError:
            self.set_status("Invalid input. Switch cancelled.")

        self.draw_all()

    def _set_gemini_model(self, model_name: str):
        tone = self.ai.tone
        self.ai = GeminiClient(os.environ["GEMINI_API_KEY"])
        self.ai.set_tone(tone)
        self.ai.model = model_name

    def _set_ollama_model(self, model_name: str):
        tone = self.ai.tone
        self.ai = OllamaClient(model=model_name)
        self.ai.set_tone(tone)

    def _prompt_ollama_custom(self):
        old_timeout = self._stream_timeout
        self.stdscr.timeout(-1)
        curses.flushinp()
        self.stdscr.clear()
        self._safe_addstr(0, 0, "Enter Ollama model name (e.g. llama3.2:1b): ")
        self.stdscr.refresh()
        curses.echo()
        curses.curs_set(1)
        try:
            model = self.stdscr.getstr(0, 50, 40).decode('utf-8').strip()
        except Exception:
            model = ""
        finally:
            curses.noecho()
            curses.curs_set(0)
            self.stdscr.timeout(old_timeout)
        if model:
            self._set_ollama_model(model)
        else:
            self.set_status("No model entered. Switch cancelled.")

    # ------------------------------------------------------------------
    # Node manipulation (unchanged)
    # ------------------------------------------------------------------
    def add_argument(self):
        old_timeout = self._stream_timeout
        curses.echo()
        curses.curs_set(1)
        self.stdscr.clear()
        self._safe_addstr(0, 0, "Enter argument title: ")
        self.stdscr.refresh()
        self.stdscr.timeout(-1)
        curses.flushinp()

        title = ""
        try:
            raw = self.stdscr.getstr(0, 22, 60)
            title = raw.decode('utf-8').strip()
        except Exception:
            pass
        finally:
            curses.noecho()
            curses.curs_set(0)
            self.stdscr.timeout(old_timeout)

        if not title:
            self.set_status("Argument title cannot be empty.", is_error=True)
            self.draw_all()
            return

        new_node = Node(title, node_type="argument", parent=self.current_node)
        self.current_node.children.append(new_node)
        self.current_node.collapsed = False
        self.current_node = new_node
        self.pm.save()
        self.set_status(f"Added argument '{title}'. Press Enter to generate content.")
        self.draw_all()

    def delete_node(self):
        if self.current_node.is_root():
            self.set_status("Cannot delete root node.", is_error=True)
            return
        parent = self.current_node.parent
        if parent:
            parent.children.remove(self.current_node)
            self.current_node = parent
            self.pm.save()
            self.set_status("Deleted node.")


# ----------------------------------------------------------------------
# Terminal project browser (unchanged)
# ----------------------------------------------------------------------
def list_projects(save_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(save_dir):
        return []
    projects = []
    for f in sorted(os.listdir(save_dir)):
        if f.endswith(".json"):
            path = os.path.join(save_dir, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                name = data.get("project_name", f)
            except Exception:
                name = f
            projects.append((name, path))
    return projects

def browse_existing_project() -> Optional[str]:
    save_dir = os.path.join(str(Path.home()), "Documents", "spearhead_projects")
    projects = list_projects(save_dir)
    if not projects:
        print("No saved projects found in", save_dir)
        return None
    print("\n-- Saved Projects --")
    for i, (name, _) in enumerate(projects, 1):
        print(f"  {i}. {name}")
    print("  0. Go back")
    while True:
        choice = input("\nSelect a project number (0 to cancel): ").strip()
        if choice == "0":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(projects):
                return projects[idx][1]
        except ValueError:
            pass
        print("Invalid choice.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def load_tone_file(tone_path: str) -> Optional[str]:
    if not tone_path:
        return None
    path = Path(tone_path).expanduser()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main(stdscr, project_file, topic, local, local_model, tone_text, verbose):
    if local or not os.environ.get("GEMINI_API_KEY"):
        ai = OllamaClient(model=local_model or "gemma3:1b")
    else:
        ai = GeminiClient(os.environ["GEMINI_API_KEY"])

    if tone_text:
        ai.set_tone(tone_text)

    if project_file:
        pm = ProjectManager(project_file)
        pm.load()
    else:
        pm = ProjectManager.new_project(topic)

    app = SpearheadApp(stdscr, pm, ai, verbose=verbose)
    app.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Spearhead - interactive learning with argument trees")
    parser.add_argument("project_file", nargs="?", help="Path to existing project JSON file")
    parser.add_argument("--topic", help="Create a new project with the given topic")
    parser.add_argument("--local", action="store_true", help="Force use of local Ollama model")
    parser.add_argument("--local-model", default="gemma3:1b", help="Ollama model name (default: gemma3:1b)")
    parser.add_argument("--tone-file", help="Path to a file containing tone/style instructions for the AI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print AI and export logs after quitting")
    args = parser.parse_args()

    tone_text = None
    tone_file_path = args.tone_file or os.path.expanduser("~/.spearhead_tone")
    if tone_file_path:
        tone_text = load_tone_file(tone_file_path)
        if args.tone_file and not tone_text:
            print(f"Tone file not found: {args.tone_file}", file=sys.stderr)
            sys.exit(1)

    project_file = args.project_file
    topic = args.topic

    if not project_file and not topic:
        print("Welcome to Spearhead!")
        print("1. Open an existing project")
        print("2. Create a new project")
        choice = input("Choose (1/2): ").strip()
        if choice == "1":
            project_file = browse_existing_project()
            if not project_file:
                print("No project selected. Exiting.")
                sys.exit(0)
        elif choice == "2":
            topic = input("Enter a topic for your new project: ").strip()
            if not topic:
                print("No topic entered. Exiting.")
                sys.exit(0)
        else:
            print("Invalid choice.")
            sys.exit(0)

    curses.wrapper(main, project_file, topic, args.local, args.local_model, tone_text, args.verbose)

    # If verbose, print export and AI logs
    if args.verbose and project_file:
        log_path = project_file.replace('.json', '.export.log')
        if os.path.exists(log_path):
            print("\n===== EXPORT LOG =====")
            with open(log_path, 'r', encoding='utf-8') as lf:
                print(lf.read())

        ai_log_path = project_file.replace('.json', '.ai.log')
        if os.path.exists(ai_log_path):
            print("\n===== RAW AI OUTPUT =====")
            with open(ai_log_path, 'r', encoding='utf-8') as lf:
                print(lf.read())
