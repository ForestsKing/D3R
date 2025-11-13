# 1. Base Image:
#    - PyTorch 1.11.0
#    - CUDA 11.3 (The corresponding version for torch 1.11.0)
#    - Python 3.9.12 (Closest available to 3.9.13)
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# 2. Set Working Directory
WORKDIR /app

# 3. (CRITICAL!) Package Installation
#    We are avoiding installing directly from a local 'requirements.txt'
#    because it often causes Python version conflicts (specifically with Python 3.9)
#    due to pre-compiled binaries or version lock issues.
#    Instead, we directly install the core packages needed for the project,
#    allowing pip to automatically select compatible versions for this environment.
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib

# 4. Copy all project files
COPY . .

# 5. Default command: Start bash shell
CMD ["/bin/bash"]
