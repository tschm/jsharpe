"""Utilities to export Marimo notebooks and generate an index page.

This script discovers Python-based Marimo notebooks in book/marimo, exports
each to HTML using marimo export, and renders an index page via Jinja2.
It is designed to be invoked from CI or the Makefile and avoids using os in
favor of pathlib for portability and clarity.
"""

import dataclasses
import subprocess
import sys
from pathlib import Path

import jinja2

# Get the base directory (assuming script is in .github/scripts/)
base_dir = Path(__file__).resolve().parents[3]
notebooks_dir = base_dir / "book" / "marimo"
output_dir = base_dir / "_marimushka"
template_file = base_dir / ".github" / "templates" / "tailwind.html.j2"


@dataclasses.dataclass(frozen=True)
class Notebook:
    """Represents a marimo notebook.

    This class encapsulates a marimo notebook (.py file) and provides methods
    for exporting it to HTML/WebAssembly format.

    Attributes:
        py_path (Path): Path to the marimo notebook (.py file)
    """

    py_path: Path

    def __post_init__(self):
        """Validate the notebook path after initialization.

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the path is not a file or not a Python file
        """
        if not self.py_path.exists():
            raise FileNotFoundError(f"File not found: {self.py_path}")
        if not self.py_path.is_file():
            raise ValueError(f"Path is not a file: {self.py_path}")
        if not self.py_path.suffix == ".py":
            raise ValueError(f"File is not a Python file: {self.py_path}")

    @property
    def py_file(self) -> Path:
        """Return the underlying Python file path of the notebook."""
        return self.py_path

    @property
    def display_name(self) -> str:
        """Return the display name for the notebook."""
        return self.py_path.stem.replace("_", " ").title()

    @property
    def html_path(self) -> Path:
        """Return the path to the exported HTML file."""
        return output_dir / "notebooks" / f"{self.py_file.stem}.html"

    @property
    def html_url(self) -> str:
        """Return the relative URL for the HTML file."""
        return f"{self.py_file.stem}.html"


def folder_to_notebooks(folder: Path | str | None) -> list[Notebook]:
    """Find all marimo notebooks in a directory."""
    if folder is None or folder == "":
        return []

    folder_path = Path(folder) if isinstance(folder, str) else folder
    if not folder_path.exists():
        print(f"Warning: Notebooks directory not found: {folder_path}")
        return []

    notebooks = []
    for nb_path in folder_path.glob("*.py"):
        try:
            notebook = Notebook(py_path=nb_path)
            notebooks.append(notebook)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {nb_path}: {e}")

    return notebooks


def export_notebook(notebook: Notebook) -> bool:
    """Export a single notebook to HTML format.

    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        print(f"Exporting: {notebook.py_path.name} -> {notebook.html_path.name}")

        result = subprocess.run(
            [
                "uv",
                "run",
                "marimo",
                "export",
                "html",
                "--force",
                "--no-sandbox",
                str(notebook.py_path),
                "-o",
                str(notebook.html_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.returncode == 0:
            print(f"✓ Successfully exported {notebook.py_path.name}")
            return True
        else:
            print(f"✗ Failed to export {notebook.py_path.name}: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Error exporting {notebook.py_path.name}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error exporting {notebook.py_path.name}: {e}")
        return False


def generate_index(notebooks: list[Notebook]) -> bool:
    """Generate the index.html file from the template.

    Returns:
        bool: True if generation was successful, False otherwise
    """
    try:
        index_path = output_dir / "index.html"

        # Ensure template directory exists
        if not template_file.exists():
            print(f"✗ Template file not found: {template_file}")
            return False

        template_dir = template_file.parent
        template_name = template_file.name

        # Create Jinja2 environment and load template
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir), autoescape=jinja2.select_autoescape(["html", "xml"])
        )
        template = env.get_template(template_name)

        # Sort notebooks by display name for consistent ordering
        sorted_notebooks = sorted(notebooks, key=lambda x: x.display_name)

        # Render the template with notebook data
        rendered_html = template.render(notebooks=sorted_notebooks)

        # Write the rendered HTML to the index.html file
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(rendered_html, encoding="utf-8")

        print(f"✓ Generated index at: {index_path}")
        return True

    except jinja2.TemplateError as e:
        print(f"✗ Template error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error generating index: {e}")
        return False


def main() -> int:
    """Main function to export notebooks and generate index.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("Marimo Notebook Exporter")
    print("=" * 60)

    # Create output directories
    (output_dir / "notebooks").mkdir(parents=True, exist_ok=True)

    # Find all notebooks
    notebooks = folder_to_notebooks(notebooks_dir)

    if not notebooks:
        print("No Marimo notebooks found to export.")
        print(f"Expected notebooks in: {notebooks_dir}")
        return 1

    print(f"Found {len(notebooks)} notebook(s):")
    for nb in notebooks:
        print(f"  • {nb.display_name} ({nb.py_path.name})")
    print()

    # Export each notebook
    successful_exports = 0
    for notebook in notebooks:
        if export_notebook(notebook):
            successful_exports += 1

    print(f"\nSuccessfully exported {successful_exports}/{len(notebooks)} notebook(s)")

    # Generate index page if we have at least one successful export
    if successful_exports > 0:
        if generate_index(notebooks):
            print("\n✓ Export process completed successfully")
            return 0
        else:
            print("\n✗ Failed to generate index page")
            return 1
    else:
        print("\n✗ No notebooks were successfully exported")
        return 1


if __name__ == "__main__":
    sys.exit(main())
