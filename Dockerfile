# Base image setup
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy lock file
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# run commands
RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"
