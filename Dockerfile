# TensorDex — SOSP '26 Artifact Evaluation container
#
#   docker build -t tensordex-ae .
#   docker run --rm tensordex-ae                       # kick-the-tires (= make check)
#   docker run -it --rm -v tensordex-cache:/tensordex/ae/cache tensordex-ae bash
#     # inside:  make ae-cache && make reproduce-all
#
# The named volume keeps the ~9 GB cache across container runs.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git make \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain — builds the tensordex._ops extension.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /tensordex

# CPU-only torch first: the pipeline never touches a GPU, and the default CUDA
# wheels would add ~5 GB to the image for nothing.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Build WITH the optional FM++ codec (vendored FM-Delta lib, x86_64-linux) so
# `make verify` re-derives the 70.5% `fratio` bit-exact out of the box;
# fall back to the pure-Rust build on other architectures.
RUN pip install --no-cache-dir --config-settings=build-args="--features fmpp" . \
    || pip install --no-cache-dir . \
    && pip install --no-cache-dir -r ae/requirements-ae.txt

# Fail the build loudly if the FM++ codec didn't make it in (x86_64 images
# must be able to re-derive the 70.5% result; on other arches remove this).
RUN python -c "from tensordex import _ops; \
    assert hasattr(_ops, 'compress_fmpp_rust'), 'FM++ codec missing from build'; \
    print('OK  FM++ codec present')"

CMD ["make", "check"]
