# Templates for exporting Marimo notebooks

This directory contains the Jinja2 template(s) used to generate the index page
for exported Marimo notebooks produced by the project’s scripts.

## Available template

### tailwind.html.j2

A lean template based on Tailwind CSS loaded via CDN. It uses utility classes for
simple, responsive styling without additional build steps.

## How templates are used in this repository

Exports are performed by the helper script:

- .github/templates/scripts/apply_jinja.py

That script:
- finds all .py notebooks in book/marimo,
- exports each to HTML using: uv run marimo export html --no-sandbox,
- renders an index page using this Jinja2 template.

You can run it via the Makefile target:

```bash
make marimushka
```

or directly:

```bash
uv run python .github/templates/scripts/apply_jinja.py
```

The generated files are written to the _marimushka directory.

## Overriding the template

By default, apply_jinja.py uses:

- .github/templates/tailwind.html.j2

If you want to customize the index, edit that file or change template_file in
apply_jinja.py to point to a different Jinja2 template.

## Template context (variables available to Jinja2)

The template receives the following variables:

- notebooks: list of Notebook objects (one per exported .py notebook)

Each Notebook object exposes:
- display_name: human‑friendly title derived from the filename
- html_path: filesystem Path of the exported HTML file
- html_url: relative URL (e.g., "psr.html") suitable for links in the index

Example usage inside a template:

```jinja2
<ul>
  {% for nb in notebooks %}
    <li><a href="{{ nb.html_url }}">{{ nb.display_name }}</a></li>
  {% endfor %}
</ul>
```

## Creating your own template

Create a new .html.j2 file and ensure apply_jinja.py points to it. You can
copy tailwind.html.j2 as a starting point and adjust HTML/CSS as needed.
