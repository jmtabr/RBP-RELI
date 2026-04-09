# Build patched RELI binary from submodule source.
# This image includes fixes for:
#   - Bug 5: uninitialized max_diff causing Z-score saturation (CRITICAL)
#   - Bug 7: atoi() overflow on large genomic coordinates
#
# Build:  docker build -f Dockerfile.cpp -t reli:patched .
# Test:   docker run --rm reli:patched /reli/RELI --help

FROM alpine

RUN apk update && apk add g++ make gsl gsl-dev bzip2 bash which libcurl

RUN mkdir -p /reli/src
WORKDIR /reli
COPY RELI/src src/
COPY RELI/Makefile .

RUN make
RUN ln -s RELI reli
ENV PATH=$PATH:/reli

RUN apk del g++ && apk add libstdc++ libgcc
