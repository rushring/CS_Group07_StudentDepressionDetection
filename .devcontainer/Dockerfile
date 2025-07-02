# Use Miniconda as base image for Mamba support
FROM continuumio/miniconda3

# Install mamba in base environment
RUN conda install -n base -c conda-forge mamba && \
    conda clean -a

# Set working directory
WORKDIR /workspace

# Copy the environment.yml file
COPY environment.yml .

# Create the environment using mamba
RUN mamba env create -f environment.yml

# Activate the environment by default
SHELL ["conda", "run", "-n", "ml-env", "/bin/bash", "-c"]

# Set entrypoint to bash so users start in the environment
ENTRYPOINT ["conda", "run", "-n", "ml-env", "/bin/bash"]
