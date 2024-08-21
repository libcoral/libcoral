# Deploy the documentation on Github pages
docs:
  #!/usr/bin/env bash
  pip uninstall --yes libcoral
  pushd py-libcoral
  maturin develop --release
  popd
  pushd docs-src
  quarto publish gh-pages --no-prompt
