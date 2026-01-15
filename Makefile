# Makefile for building documentation with MkDocs

build:
	mkdocs build

serve:
	mkdocs serve --dev-addr localhost:8000 --livereload

deploy:
	mkdocs gh-deploy

clean:
	rm -rf site/

help:
	@echo "Makefile commands:"
	@echo "  build   - Build the documentation site"
	@echo "  serve   - Serve the documentation site locally"
	@echo "  deploy  - Deploy the documentation site to GitHub Pages"
	@echo "  clean   - Remove the built site directory"
	
.PHONY: build serve deploy clean