SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build

.PHONY: build
build: sp-html

.PHONY: clean
clean:
	rm -rf site

live:
	npx nodemon --ext txt --exec "$(MAKE) build" --ignore site

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
.PHONY: ALWAYS
sp-%: ALWAYS
	@$(SPHINXBUILD) -M $* . site $(SPHINXOPTS) $(O)
