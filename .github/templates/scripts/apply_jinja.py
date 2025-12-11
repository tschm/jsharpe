import subprocess
import dataclasses
from pathlib import Path

import jinja2

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
        path (Path): Path to the marimo notebook (.py file)


    """

    py_path: Path
    html_path: Path

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
    def py_file(self):
        return self.py_path

    @property
    def display_name(self) -> str:
        """Return the display name for the notebook."""
        return self.py_path.stem.replace("_", " ")

    @property
    def html(self) -> Path:
        """Return the path to the exported HTML file."""
        return self.html_path / f"{self.py_file.stem}.html"


def folder2notebooks(folder: Path | str | None, html_path = output_dir / "notebooks") -> list[Notebook]:
    """Find all marimo notebooks in a directory."""
    if folder is None or folder == "":
        return []

    return [Notebook(py_path=nb, html_path=html_path) for nb in list(Path(folder).glob("*.py"))]


if __name__ == "__main__":
    (output_dir / "notebooks").mkdir(parents=True, exist_ok=True)

    notebooks = folder2notebooks(notebooks_dir, html_path=output_dir / "notebooks")
    print(notebooks)

    # Iterate over Python notebooks using pathlib; avoid os.listdir
    for notebook in notebooks:
        print(notebook.py_path)
        print(notebook.html)

        # export file with marimo
        #out_file = output_dir / "notebooks" / f"{notebook.stem}.html"
        subprocess.run([
            "uv",
            "run",
            "marimo",
            "export",
            "html",
            "--force",
            "--no-sandbox",
            str(notebook.py_path),
            "-o",
            str(notebook.html),
        ])

    # Create the full path for the index.html file
    index_path: Path = Path(output_dir) / "index.html"

    # Set up Jinja2 environment and load template
    template_dir = template_file.parent
    template_name = template_file.name

    rendered_html = ""

    # Create Jinja2 environment and load template
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir), autoescape=jinja2.select_autoescape(["html", "xml"])
    )
    template = env.get_template(template_name)

    # Render the template with notebook and app data
    rendered_html = template.render(
        notebooks=notebooks
    )

    # Write the rendered HTML to the index.html file
    with Path.open(index_path, "w") as f:
        f.write(rendered_html)




