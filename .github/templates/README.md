# Marimushka Templates

This directory contains the HTML templates used by Marimushka to generate the index
page for exported marimo notebooks.

## Available Templates

### tailwind.html.j2

A lean template based on [Tailwind CSS](https://tailwindcss.com/).
This template uses Tailwind's utility
classes for styling, resulting in a modern, responsive design.
The Tailwind CSS is loaded via CDN, so no additional installation is required.

## Using a Custom Template

You can specify which template to use when running Marimushka:

### Command Line

```bash
# Use the default template (default)
marimushka export

# Use the Tailwind template
marimushka export --template /path/to/marimushka/templates/tailwind.html.j2
```

### Python API

```python
from marimushka.export import main
from pathlib import Path

# Use the default template (default)
main()

# Use the Tailwind template
main(template=Path(__file__).parent / "templates" / "tailwind.html.j2")
```

## Creating Your Own Template

You can create your own custom template by using the existing templates
as a reference. The template should be a Jinja2 template
file with the `.html.j2` extension.

The template has access to the following variables:

- `notebooks`: List of Notebook objects for HTML notebooks
- `notebooks_wasm`: List of Notebook objects for WebAssembly notebooks
- `apps`: List of Notebook objects for apps

Each Notebook object has the following properties:

- `display_name`: The display name of the notebook
- `html_path`: The path to the exported HTML file
