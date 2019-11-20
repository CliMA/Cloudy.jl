# Cloudy

| **Documentation**                             | **Build Status**                                                                                                     |
|:--------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------|
| [![latest][docs-latest-img]][docs-latest-url] | [![travis][travis-img]][travis-url] [![codecov][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://climate-machine.github.io/cloudy/latest/

[travis-img]: https://travis-ci.org/climate-machine/cloudy.svg?branch=master
[travis-url]: https://travis-ci.org/climate-machine/cloudy

[codecov-img]: https://codecov.io/gh/climate-machine/cloudy/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/cloudy

A multi-moment cloud microphysics toy model.

Currently Cloudy only supports collisions with simple kernels.

Examples can be found in the examples folder.

Installation hints can be found in .travis.yml file.

To build the docs locally:

- ```julia --project=docs```

- ```include(docs/make.jl)```

MAKE IT YOUR OWN!
