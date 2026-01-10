# Write a makefile for building documentation in Sphinx.

build-html:
	sphinx-build -b html . _build/html

build-latexpdf:
	sphinx-build -b latex . _build/latex
	cd _build/latex && make all-pdf

build-all: build-html build-latexpdf

view-html:
	python3 -m http.server --directory _build/html 8000

live-html:
	sphinx-autobuild . _build/html --open-browser
	
clean:
	rm -rf _build/*

.PHONY: build-html build-latexpdf build-all clean