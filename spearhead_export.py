#!/usr/bin/env python3
"""
spearhead_export.py - Called by spearhead on save to generate PDF and upload to Google Drive.
Saves PDFs locally in ~/Documents/spearhead_projects/pdf/
and uploads them to a dedicated "Spearhead PDFs" folder on Google Drive.
If a PDF with the same name already exists in that folder, it is updated (not duplicated).

Usage: python spearhead_export.py <project_file.json>
"""

import sys, os, json, subprocess, tempfile, shutil
from pathlib import Path

# Google Drive
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

LOCAL_PDF_DIR = os.path.join(str(Path.home()), "Documents", "spearhead_projects", "pdf")
os.makedirs(LOCAL_PDF_DIR, exist_ok=True)

DRIVE_FOLDER_NAME = "Spearhead PDFs"

def authenticate():
    creds = None
    token_file = os.path.join(os.path.dirname(__file__), 'token.json')
    creds_file = os.path.join(os.path.dirname(__file__), 'credentials.json')
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = response.get('files', [])
    if folders:
        return folders[0]['id']
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')

def find_file_in_folder(service, folder_id, filename):
    """Return file ID if a file with the given name exists in the folder, else None."""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = response.get('files', [])
    return files[0]['id'] if files else None

def upload_pdf(filepath, filename=None):
    """
    Upload or update a PDF in the "Spearhead PDFs" folder.
    If a file with the same name exists, its content is replaced; otherwise a new file is created.
    """
    service = authenticate()
    folder_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)
    name = filename or os.path.basename(filepath)

    existing_id = find_file_in_folder(service, folder_id, name)
    media = MediaFileUpload(filepath, mimetype='application/pdf')

    if existing_id:
        # Update existing file
        service.files().update(fileId=existing_id, media_body=media).execute()
        print(f"Updated PDF on Drive (ID: {existing_id})")
    else:
        # Create new file
        file_metadata = {
            'name': name,
            'parents': [folder_id]
        }
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded PDF with ID: {file.get('id')}")

def build_latex_recursive(node, depth=0):
    latex = ""
    title = node['title']
    title = title.replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('^', '\\^{}').replace('~', '\\textasciitilde{}')
    content = node.get('content', '')
    content = content.replace('\\', '\\textbackslash{}').replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}').replace('^', '\\^{}').replace('~', '\\textasciitilde{}')
    if depth == 1:
        latex += f"\\section{{{title}}}\n"
        latex += content + "\n\n"
    elif depth >= 2:
        if depth == 2:
            latex += f"\\subsection{{{title}}}\n"
        else:
            latex += f"\\subsubsection*{{{title}}}\n"
        latex += content + "\n\n"
    for child in node.get('children', []):
        latex += build_latex_recursive(child, depth + 1)
    return latex

def render_template(data):
    template_path = os.path.join(os.path.dirname(__file__), 'template.tex')
    if not os.path.exists(template_path):
        raise RuntimeError(f"Template file not found: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    root = data['root']
    title = root['title']
    introduction = root.get('content', '')
    content = ""
    for child in root.get('children', []):
        content += build_latex_recursive(child, depth=1)
    return (template.replace('{{ title }}', title)
                   .replace('{{ introduction }}', introduction)
                   .replace('{{ content }}', content))

def check_pdflatex():
    try:
        subprocess.run(['pdflatex', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "pdflatex not found. Please install a LaTeX distribution.\n"
            "Ubuntu/Debian: sudo apt install texlive-latex-base\n"
            "macOS: brew install mactex\n"
            "Windows: install MiKTeX and ensure pdflatex is in your PATH."
        )

def compile_and_upload(project_file):
    check_pdflatex()

    with open(project_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    latex_code = render_template(data)

    project_name = os.path.splitext(os.path.basename(project_file))[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, 'doc.tex')
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(latex_code)

        pdflatex_output = []
        for run in [1, 2]:
            proc = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'doc.tex'],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            pdflatex_output.append(f"=== Run {run} ===\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n")

        pdf_file = os.path.join(tmpdir, 'doc.pdf')
        if not os.path.exists(pdf_file):
            raise RuntimeError(
                "pdflatex failed to produce PDF.\n\n"
                "Output from pdflatex:\n" + "\n".join(pdflatex_output)
            )

        # Save local copy
        local_pdf_path = os.path.join(LOCAL_PDF_DIR, f"{project_name}.pdf")
        shutil.copy(pdf_file, local_pdf_path)
        print(f"Local PDF saved to {local_pdf_path}")

        # Upload (or update) to Google Drive
        try:
            upload_pdf(local_pdf_path, f"{project_name}.pdf")
        except Exception as e:
            print(f"Drive upload failed: {e}", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python spearhead_export.py <project.json>", file=sys.stderr)
        sys.exit(1)
    try:
        compile_and_upload(sys.argv[1])
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
